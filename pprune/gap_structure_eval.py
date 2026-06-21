"""
gap_structure_eval.py
----------------------
Synthetic gap-structure ablation: isolates causal-gap *geometry* from token
*selection*, addressing the review concern that the structural-corruption
mechanism (§6) is supported only by comparisons that vary selection method
and gap presence together (naive vs. streaming; SnapKV vs. SnapKV-Select).

For a fixed retention fraction r, four content-blind (no scoring, no query)
position geometries are generated purely from sequence length T:

  block_end   — one contiguous block at the tail.       1 gap  (before it)
  block_mid   — one contiguous block, centered.          2 gaps (before/after)
  clustered4  — four evenly-spaced contiguous blocks.    ~4 gaps
  scattered   — evenly-spaced single-token positions.    ~budget gaps (maximal)

Each geometry is evaluated under two presentations on the *same* retained
position set:

  gapless — the retained positions are gathered into a new, shorter,
            contiguous prompt with no KV patching (no causal gaps).
  gapped  — the full-length prompt is processed with KV pruning that keeps
            exactly the same original-index positions (real causal gaps
            wherever a position was evicted).

KL faithfulness is measured teacher-forced on the shared y* sequence, same
as kl_faith_eval_ystar.py. The (gapped - gapless) delta for each geometry,
holding the retained token set fixed, isolates the cost of the gap itself;
comparing that delta across geometries tests whether gap *structure*
(count, position, scatter) modulates the severity of the corruption.

Usage:
    python gap_structure_eval.py \
        --model meta-llama/Llama-3.1-8B \
        --tasks 2wikimqa,multifieldqa_en,qasper,qmsum,repobench-p,triviaqa \
        --output lb_results_base/gap_structure.json \
        --ystar_cache lb_results_base/ystar_cache_v3.pt \
        --fraction 0.65 \
        --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_pruned import PrunedLlamaConfig
from kl_faith_eval import (
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_COMP,
    DEFAULT_MAX_SEQ_FULL,
    TASK_MAX_SEQ_COMP,
    TASK_MAX_SEQ_FULL,
    patch_model,
    unpatch_model,
    load_ckpt,
    save_ckpt,
    fmt_duration,
    YSTAR_STOP_TOKENS,
)
from kl_faith_eval_ystar import (
    generate_ystar,
    get_comp_log_probs,
    kl_divergence,
    load_ystar_cache,
    save_ystar_cache,
    get_from_cache,
    put_in_cache,
)

GEOMETRIES = ("block_end", "block_mid", "clustered4", "scattered")
PRESENTATIONS = ("gapless", "gapped")


# ---------------------------------------------------------------------------
# Geometry generators — pure functions of (T, fraction). No scoring, no query,
# no content. Each returns a sorted LongTensor of retained original positions.
# ---------------------------------------------------------------------------

def _budget(T: int, fraction: float) -> int:
    return max(1, int(T * fraction))


def block_end_positions(T: int, fraction: float) -> torch.Tensor:
    budget = _budget(T, fraction)
    start = T - budget
    return torch.arange(start, T, dtype=torch.long)


def block_mid_positions(T: int, fraction: float) -> torch.Tensor:
    budget = _budget(T, fraction)
    start = max(0, (T - budget) // 2)
    end = min(T, start + budget)
    return torch.arange(start, end, dtype=torch.long)


def clustered4_positions(T: int, fraction: float, n_clusters: int = 4) -> torch.Tensor:
    budget = _budget(T, fraction)
    per_cluster = max(1, budget // n_clusters)
    seg_len = T / n_clusters
    positions = []
    for c in range(n_clusters):
        seg_start = c * seg_len
        seg_center = seg_start + seg_len / 2
        start = int(round(seg_center - per_cluster / 2))
        start = max(int(seg_start), min(start, int(seg_start + seg_len - per_cluster)))
        start = max(0, min(start, T - per_cluster))
        positions.extend(range(start, start + per_cluster))
    pos = sorted(set(positions))
    return torch.tensor(pos[:budget] if len(pos) > budget else pos, dtype=torch.long)


def scattered_positions(T: int, fraction: float) -> torch.Tensor:
    budget = _budget(T, fraction)
    # `budget` evenly-spaced indices across [0, T) — deterministic, content-blind.
    idx = sorted(set(int(round(j * T / budget)) for j in range(budget)))
    idx = [min(i, T - 1) for i in idx]
    return torch.tensor(sorted(set(idx)), dtype=torch.long)


GEOMETRY_FNS = {
    "block_end": block_end_positions,
    "block_mid": block_mid_positions,
    "clustered4": clustered4_positions,
    "scattered": scattered_positions,
}


# ---------------------------------------------------------------------------
# Comp-id construction for the two presentations
# ---------------------------------------------------------------------------

def gapless_comp_ids(full_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather retained positions into a new, shorter, contiguous prompt. No KV patching."""
    return full_ids[:, positions]


def gapped_pcfg(positions: torch.Tensor, fraction: float) -> PrunedLlamaConfig:
    """KV-prune to exactly `positions` at their original indices. Real causal gaps elsewhere."""
    return PrunedLlamaConfig(
        score_mode="fixed_positions",
        fixed_positions=positions,
        budget_fraction=fraction,
        always_keep_first=0,
        always_keep_last=0,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    tasks: List[str],
    geometries: List[str],
    fraction: float,
    data_dir: Path,
    output_path: Path,
    max_examples: int,
    device: str,
    ystar_cache_path: Optional[Path] = None,
):
    ckpt = load_ckpt(output_path)
    cache = load_ystar_cache(ystar_cache_path) if ystar_cache_path else {}

    def _is_done(key):
        return key in ckpt and (ckpt[key] is None or "kl" in ckpt[key])

    n_methods = len(geometries) * len(PRESENTATIONS)
    total_work = len(tasks) * max_examples * n_methods
    completed_at_start = sum(1 for k, v in ckpt.items() if _is_done(k))
    grand_done = completed_at_start
    grand_start = time.time()

    print(f"\n{'='*72}\nGap-structure ablation  (fraction={fraction})")
    print(f"  Tasks      : {tasks}")
    print(f"  Geometries : {geometries}")
    print(f"  Total      : {total_work} entries  ({completed_at_start} already done)")
    print(f"{'='*72}\n")

    for task in tasks:
        data_file = data_dir / f"{task}.jsonl"
        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template = DATASET2PROMPT[task]
        max_new = DATASET2MAXLEN.get(task, 64)
        max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        stop_toks = YSTAR_STOP_TOKENS.get(task)

        n_ex = min(max_examples, len(dataset))
        print(f"[{task}]  n={n_ex}")

        for idx in range(n_ex):
            ex = dataset[idx]
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            context_text = ex.get("context", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(context=context_text, input=question_text)
            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if full_ids.shape[1] > max_seq_full:
                full_ids = full_ids[:, -max_seq_full:]
            T = full_ids.shape[1]

            ystar, log_p_full = get_from_cache(cache, task, idx)
            if ystar is None:
                ystar, log_p_full, _ = generate_ystar(
                    model, full_ids, max_new, max_seq_full, device,
                    stop_token_ids=stop_toks,
                )
                if ystar is not None and ystar_cache_path:
                    put_in_cache(cache, task, idx, ystar, log_p_full)
                    save_ystar_cache(cache, ystar_cache_path)
            if ystar is None:
                grand_done += n_methods
                continue

            for geom in geometries:
                positions = GEOMETRY_FNS[geom](T, fraction)

                for presentation in PRESENTATIONS:
                    ck_key = f"{task}|{geom}_{presentation}|{idx}"
                    if _is_done(ck_key):
                        grand_done += 1
                        continue

                    if presentation == "gapless":
                        comp_ids = gapless_comp_ids(full_ids, positions)
                        pcfg = None
                    else:
                        comp_ids = full_ids
                        pcfg = gapped_pcfg(positions, fraction)

                    log_p_comp = get_comp_log_probs(model, comp_ids, ystar, pcfg, device)
                    if log_p_comp is not None:
                        kl = kl_divergence(log_p_full, log_p_comp)
                        ckpt[ck_key] = {"kl": kl, "n_gen": ystar.shape[0],
                                        "n_retained": int(positions.shape[0]), "T": T}
                    else:
                        ckpt[ck_key] = None

                    save_ckpt(ckpt, output_path)
                    grand_done += 1

                    elapsed = time.time() - grand_start
                    rate = (grand_done - completed_at_start) / max(elapsed, 1)
                    remaining = (total_work - grand_done) / max(rate, 1e-9)
                    kl_str = f"{kl:.4f}" if log_p_comp is not None else "OOM"
                    print(f"  ex {idx+1:>3}/{n_ex}  {geom:<11} {presentation:<8}"
                          f"  KL={kl_str}  [ETA {fmt_duration(remaining)}]", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir", default="lb_data_raw/data", type=Path)
    p.add_argument("--tasks", default="2wikimqa,multifieldqa_en,qasper,qmsum,repobench-p,triviaqa")
    p.add_argument("--geometries", default=",".join(GEOMETRIES))
    p.add_argument("--fraction", type=float, default=0.65)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ystar_cache", type=Path, default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]

    run_eval(
        model, tokenizer, tasks, geometries, args.fraction,
        args.data_dir, args.output, args.n, args.device,
        ystar_cache_path=args.ystar_cache,
    )


if __name__ == "__main__":
    main()

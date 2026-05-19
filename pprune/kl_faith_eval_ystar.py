"""
kl_faith_eval_ystar.py
----------------------
KL faithfulness evaluation using y* (full-model tokens) as the shared prefix.

For each example:
  1. Generate y* from the uncompressed full model (output_scores=True captures
     P_full at each step for free — no second forward pass needed).
  2. For each compression method, teacher-force the compressed model on
     [comp_prompt + y*] and extract P_comp at the y* positions.
  3. Compute mean KL(P_full(· | full_ctx, y*_{<t}) || P_comp(· | comp_ctx, y*_{<t})).

All methods are scored on the same token sequence y*, eliminating the
path-dependence issue in kl_faith_eval.py where each method generated its own ŷ.

Loop order: task × example × method
  y* and P_full computed once per example; P_comp computed once per method.

Usage:
    python kl_faith_eval_ystar.py \\
        --model meta-llama/Llama-3.1-8B \\
        --data_dir lb_data_raw/data \\
        --output lb_results_base/kl_ystar.json \\
        --tasks 2wikimqa,narrativeqa \\
        --methods naive_65pct,kq_post_rope,snapkv,phrase_w64_t25 \\
        --n 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_pruned import (
    HeadFilterConfig,
    PerHeadFilter,
    PrunedLlamaAttention,
    PrunedLlamaConfig,
)

# ---------------------------------------------------------------------------
# Import shared config from kl_faith_eval (tasks, prompts, method configs)
# ---------------------------------------------------------------------------

from kl_faith_eval import (
    ENGLISH_TASKS,
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_COMP,
    DEFAULT_MAX_SEQ_FULL,
    TASK_MAX_SEQ_COMP,
    TASK_MAX_SEQ_FULL,
    METHOD_CONFIGS,
    CHUNK_CONFIGS,
    patch_model,
    unpatch_model,
    naive_truncate,
    chunk_truncate,
    sent_truncate,
    load_ckpt,
    save_ckpt,
    fmt_duration,
    fmt_eta,
)

# ---------------------------------------------------------------------------
# SnapKV-scored input selection (snapkv_select family)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def scored_select_ids(
    model,
    full_ids: torch.Tensor,
    device: str,
    score_mode: str,
    fraction: float = 0.65,
    min_decay: float = 0.7,
    always_keep_first: int = 16,
    always_keep_last: int = 16,
) -> Optional[torch.Tensor]:
    """
    Score tokens via a forward pass using the given score_mode (averaged across
    all layers), then return the top-k as a clean input tensor for honest prefill.

    Ablation for the structural-corruption claim: same scoring as KV-pruning methods
    but applied as input truncation — no KV cache corruption.  Returns None on OOM.

    score_mode: "snapkv" or "kq_post_rope" (RADAR)
    """
    T = full_ids.shape[1]
    budget = max(1, int(T * fraction))

    score_capture: list = []
    pcfg = PrunedLlamaConfig(
        score_mode=score_mode,
        snapkv_window=32,
        min_decay=min_decay,
        budget_fraction=1.0,       # score_capture triggers scoring even though budget >= T
        always_keep_first=0,
        always_keep_last=0,
        q_buffer_size=128,
        score_capture=score_capture,
    )
    originals = patch_model(model, pcfg, device)

    try:
        _ = model(
            input_ids=full_ids.to(device),
            attention_mask=torch.ones(1, T, device=device, dtype=torch.long),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        unpatch_model(model, originals)

    if not score_capture:
        return None

    # Average global_scores across layers: each entry is (T,) cpu float32
    avg_scores = torch.stack(score_capture, dim=0).mean(dim=0)   # (T,)

    # Select tokens using the same always-keep + top-k logic as _run_filter_prefill
    head_end   = min(always_keep_first, T)
    tail_start = max(head_end, T - always_keep_last)
    always_set = set(range(0, head_end)) | set(range(tail_start, T))

    mid_start     = head_end
    mid_end       = tail_start
    mid_n         = max(0, mid_end - mid_start)
    middle_budget = max(0, budget - len(always_set))

    if middle_budget > 0 and mid_n > 0:
        mid_scores = avg_scores[mid_start:mid_end]
        topk       = min(middle_budget, mid_n)
        _, top_idx = mid_scores.topk(topk)
        keep_set   = always_set | {mid_start + i for i in top_idx.tolist()}
    else:
        keep_set = always_set

    retained = torch.tensor(sorted(keep_set), dtype=torch.long)
    return full_ids[:, retained]


# ---------------------------------------------------------------------------
# Compressed prompt construction (mirrors kl_faith_eval.kl_for_example)
# ---------------------------------------------------------------------------

# Maps select method name → (score_mode, fraction)
_SELECT_METHODS = {}
for _name, _mode in [("snapkv_select", "snapkv"), ("radar_select", "kq_post_rope")]:
    _SELECT_METHODS[_name] = (_mode, 0.65)
    for _frac in (0.50, 0.40, 0.35):
        _fpct = int(_frac * 100)
        _SELECT_METHODS[f"{_name}_f{_fpct}"] = (_mode, _frac)


def make_comp_ids(
    full_ids: torch.Tensor,
    method: str,
    tokenizer,
    question_text: str,
    max_seq_comp: int,
    model=None,
    device: str = "cuda",
) -> torch.Tensor:
    """Return the compressed prompt token ids for a given method."""
    if method == "naive_65pct":
        comp_ids = naive_truncate(full_ids, head_frac=0.10)
    elif method.startswith("naive_") and method.endswith("pct") and method != "naive_tail":
        frac = int(method[6:-3]) / 100.0
        comp_ids = naive_truncate(full_ids, fraction=frac, head_frac=0.10)
    elif method == "naive_tail":
        comp_ids = naive_truncate(full_ids, head_frac=0.0)
    elif method == "chunk_sent":
        q_ids = (tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt")
                 if question_text else None)
        comp_ids = sent_truncate(full_ids, q_ids, tokenizer=tokenizer)
    elif method in CHUNK_CONFIGS:
        q_ids = (tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt")
                 if question_text else None)
        comp_ids = chunk_truncate(full_ids, q_ids, tokenizer=tokenizer, **CHUNK_CONFIGS[method])
    elif method in _SELECT_METHODS and model is not None:
        score_mode, frac = _SELECT_METHODS[method]
        ids = scored_select_ids(model, full_ids, device, score_mode=score_mode, fraction=frac)
        comp_ids = ids if ids is not None else naive_truncate(full_ids, fraction=frac, head_frac=0.10)
    else:
        comp_ids = full_ids.clone()

    if comp_ids.shape[1] > max_seq_comp:
        comp_ids = comp_ids[:, -max_seq_comp:]
    return comp_ids

# ---------------------------------------------------------------------------
# Per-example core functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_ystar(
    model,
    full_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_full: int,
    device: str,
):
    """Generate y* from the uncompressed model.

    Returns (gen_tokens, log_p_full) where:
      gen_tokens  : (n_gen,)        int64 — greedy tokens
      log_p_full  : (n_gen, vocab)  float32 — log-softmax at each step

    Returns (None, None) on OOM or empty generation.
    """
    prompt_ids = full_ids
    if prompt_ids.shape[1] > max_seq_full:
        prompt_ids = prompt_ids[:, -max_seq_full:]
    prompt_ids = prompt_ids.to(device)
    n_prompt = prompt_ids.shape[1]

    try:
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None

    if not out.scores:
        return None, None

    gen_tokens = out.sequences[0, n_prompt:].cpu()            # (n_gen,)
    raw_logits = torch.stack([s[0] for s in out.scores], dim=0)  # (n_gen, vocab)
    log_p_full = F.log_softmax(raw_logits.float(), dim=-1).cpu()

    del out
    torch.cuda.empty_cache()

    if gen_tokens.shape[0] == 0:
        return None, None
    return gen_tokens, log_p_full


@torch.inference_mode()
def get_comp_log_probs(
    model,
    comp_ids: torch.Tensor,
    ystar: torch.Tensor,
    pcfg: Optional[PrunedLlamaConfig],
    device: str,
) -> Optional[torch.Tensor]:
    """Teacher-force compressed model on [comp_ids + ystar].

    Returns log_p_comp (n_gen, vocab) float32, or None on OOM/shape mismatch.
    """
    n_gen = ystar.shape[0]
    full_input = torch.cat([comp_ids.to(device), ystar.unsqueeze(0).to(device)], dim=1)
    n_prompt_comp = comp_ids.shape[1]

    originals = None
    if pcfg is not None:
        originals = patch_model(model, pcfg, device)

    try:
        out = model(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input),
        )
        # Positions [n_prompt_comp-1 .. -1] predict ystar[0..n_gen-1]
        comp_logits = out.logits[0, n_prompt_comp - 1 : -1, :].float()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        comp_logits = None
    finally:
        if originals is not None:
            unpatch_model(model, originals)

    if comp_logits is None or comp_logits.shape[0] != n_gen:
        return None

    return F.log_softmax(comp_logits, dim=-1).cpu()


def kl_divergence(log_p_full: torch.Tensor, log_p_comp: torch.Tensor) -> float:
    """Mean KL(P_full || P_comp) over generation steps, in nats."""
    p_full = log_p_full.exp()
    kl_steps = (p_full * (log_p_full - log_p_comp)).sum(dim=-1)  # (n_gen,)
    return kl_steps.mean().item()

# ---------------------------------------------------------------------------
# y* cache helpers
# ---------------------------------------------------------------------------

def load_ystar_cache(path: Path) -> dict:
    """Load y* cache from a .pt file. Returns {} if not found."""
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return {}


def save_ystar_cache(cache: dict, path: Path):
    """Atomically save y* cache to a .pt file."""
    tmp = path.with_suffix(".tmp.pt")
    torch.save(cache, tmp)
    tmp.replace(path)


def cache_key(task: str, idx: int) -> str:
    return f"{task}|{idx}"


def get_from_cache(cache: dict, task: str, idx: int):
    """Return (gen_tokens, log_p_full) from cache, or (None, None) if missing."""
    entry = cache.get(cache_key(task, idx))
    if entry is None:
        return None, None
    tokens     = entry["tokens"]                          # int64 (n_gen,)
    log_p_full = entry["log_p_full"].float()              # float16 → float32
    return tokens, log_p_full


def put_in_cache(cache: dict, task: str, idx: int,
                 gen_tokens: torch.Tensor, log_p_full: torch.Tensor):
    """Store y* in cache, compressing log_p_full to float16."""
    cache[cache_key(task, idx)] = {
        "tokens":     gen_tokens.cpu(),
        "log_p_full": log_p_full.cpu().half(),   # float32 → float16 to save space
    }

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    tasks: List[str],
    methods: List[str],
    data_dir: Path,
    output_path: Path,
    max_examples: int,
    device: str,
    max_new_tokens_override: Optional[int] = None,
    ystar_cache_path: Optional[Path] = None,
):
    ckpt  = load_ckpt(output_path)
    cache = load_ystar_cache(ystar_cache_path) if ystar_cache_path else {}
    if ystar_cache_path:
        print(f"y* cache: {ystar_cache_path}  ({len(cache)} entries loaded)")

    n_tasks   = len(tasks)
    n_methods = len(methods)
    total_work = n_tasks * max_examples * n_methods
    completed_at_start = sum(
        1 for k, v in ckpt.items()
        if isinstance(v, float) and k not in ("_summary",)
    )

    print(f"\n{'='*72}")
    print(f"KL faithfulness (y* shared prefix)")
    print(f"  Tasks  : {n_tasks}  ({', '.join(tasks[:4])}{'...' if n_tasks > 4 else ''})")
    print(f"  Methods: {n_methods}  ({', '.join(methods)})")
    print(f"  N/task : {max_examples}")
    print(f"  Total  : {total_work} scores  ({completed_at_start} already done)")
    print(f"{'='*72}\n")

    grand_done  = completed_at_start
    grand_start = time.time()

    for t_idx, task in enumerate(tasks):
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file")
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template  = DATASET2PROMPT.get(task, "{context}{input}")
        max_new   = max_new_tokens_override or DATASET2MAXLEN.get(task, 64)
        max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex      = min(max_examples, len(dataset))

        task_start = time.time()
        task_done  = 0

        print(f"[{task}  {t_idx+1}/{n_tasks}]")

        for idx in range(n_ex):
            # Skip if all methods already done for this example
            if all(f"{task}|{m}|{idx}" in ckpt for m in methods):
                grand_done += n_methods
                task_done  += n_methods
                continue

            ex = dataset[idx]
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=question_text,
            )

            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

            # --- Step 1: generate y* and P_full (once per example, cache if possible) ---
            ystar, log_p_full = get_from_cache(cache, task, idx)
            if ystar is None:
                ystar, log_p_full = generate_ystar(model, full_ids, max_new, max_seq_full, device)
                if ystar is not None and ystar_cache_path:
                    put_in_cache(cache, task, idx, ystar, log_p_full)
                    save_ystar_cache(cache, ystar_cache_path)
                    print(f"  [cached y* for {task}|{idx}]", flush=True)
            else:
                print(f"  [y* cache hit: {task}|{idx}  n_gen={ystar.shape[0]}]", flush=True)

            if ystar is None:
                for m in methods:
                    ckpt.setdefault(f"{task}|{m}|{idx}", None)
                save_ckpt(ckpt, output_path)
                grand_done += n_methods
                task_done  += n_methods
                continue

            n_gen = ystar.shape[0]

            # --- Step 2: for each method, teacher-force compressed model on y* ---
            for method in methods:
                ck_key = f"{task}|{method}|{idx}"
                if ck_key in ckpt:
                    grand_done += 1
                    task_done  += 1
                    continue

                pcfg     = METHOD_CONFIGS.get(method)
                comp_ids = make_comp_ids(full_ids, method, tokenizer, question_text, max_seq_comp,
                                         model=model, device=device)

                log_p_comp = get_comp_log_probs(model, comp_ids, ystar, pcfg, device)

                if log_p_comp is not None:
                    kl = kl_divergence(log_p_full, log_p_comp)
                    ckpt[ck_key] = kl
                else:
                    ckpt[ck_key] = None
                    kl = None

                save_ckpt(ckpt, output_path)
                grand_done += 1
                task_done  += 1

                elapsed = time.time() - grand_start
                grand_eta = fmt_eta(grand_done - completed_at_start,
                                    total_work - completed_at_start, elapsed)
                kl_str = f"{kl:.4f}" if kl is not None else " OOM"
                print(
                    f"  ex {idx+1:>4}/{n_ex}  {method:<22}"
                    f"  n_gen={n_gen}  KL={kl_str}"
                    f"  [{grand_eta} total]",
                    flush=True,
                )

        task_elapsed = time.time() - task_start
        # Task summary per method
        print(f"\n  --- {task} summary ---")
        per_method = {}
        for m in methods:
            scores = [ckpt[f"{task}|{m}|{i}"] for i in range(n_ex)
                      if isinstance(ckpt.get(f"{task}|{m}|{i}"), float)]
            mean = np.mean(scores) if scores else float("nan")
            per_method[m] = mean
            print(f"  {m:<24}  n={len(scores)}  mean_KL={mean:.4f}")
        print(f"  [{fmt_duration(task_elapsed)}]\n")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"SUMMARY — KL(P_full || P_comp) with y* shared prefix (nats)")
    print(f"{'='*72}")

    col_w = 12
    header = f"{'Task':<26}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    per_method_all: Dict[str, List[float]] = {m: [] for m in methods}
    for task in tasks:
        row_vals = {}
        for m in methods:
            scores = [ckpt[f"{task}|{m}|{i}"] for i in range(max_examples)
                      if isinstance(ckpt.get(f"{task}|{m}|{i}"), float)]
            mean = np.mean(scores) if scores else float("nan")
            row_vals[m] = mean
            if not math.isnan(mean):
                per_method_all[m].append(mean)
        row = f"{task:<26}  " + "  ".join(
            f"{row_vals[m]:>{col_w}.4f}" if not math.isnan(row_vals[m]) else f"{'—':>{col_w}}"
            for m in methods
        )
        print(row)

    print("-" * len(header))
    avg = f"{'AVERAGE':<26}  " + "  ".join(
        f"{np.mean(per_method_all[m]):>{col_w}.4f}" if per_method_all[m] else f"{'—':>{col_w}}"
        for m in methods
    )
    print(avg)

    total_elapsed = time.time() - grand_start
    print(f"\nTotal wall time: {fmt_duration(total_elapsed)}")
    print(f"Results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, file_path: Path):
        self._file   = open(file_path, "a")
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg); self._file.write(msg); self._file.flush()
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def fileno(self):
        return self._stdout.fileno()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir",       default="lb_data_raw/data")
    p.add_argument("--output",         default="lb_results_base/kl_ystar.json")
    p.add_argument("--log",            default=None)
    p.add_argument("--tasks",          default=",".join(ENGLISH_TASKS))
    p.add_argument("--methods",        default="naive_65pct,kq_post_rope,snapkv,phrase_w64_t25")
    p.add_argument("--n",              type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--ystar_cache",    default="lb_results_base/ystar_cache.pt",
                   help="Path to cache y* tokens and full-model logits across runs")
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    tasks   = [t.strip() for t in args.tasks.split(",")   if t.strip() in ENGLISH_TASKS]
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHOD_CONFIGS]

    if not tasks:
        print("No valid tasks."); return
    if not methods:
        print("No valid methods."); return

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    run_eval(
        model, tokenizer,
        tasks=tasks,
        methods=methods,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        max_examples=args.n,
        device=args.device,
        max_new_tokens_override=args.max_new_tokens,
        ystar_cache_path=Path(args.ystar_cache) if args.ystar_cache else None,
    )


if __name__ == "__main__":
    main()

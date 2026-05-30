"""
calibrate_token_salience.py
---------------------------
Build a static per-token salience lookup table by running the model's
attention scoring mechanism over a text corpus and recording how much
attention each vocabulary token accumulates across many appearances
and contexts.

For each corpus sequence:
  1. Run a full forward pass with score_capture (all layers, no pruning).
  2. Average per-layer global_scores → (T,) position scores.
  3. Append each position's score to a per-token-id accumulator.

After all sequences, compute per token-id:
  - mean score  (stable central estimate)
  - max score   (ever highly important)
  - median score (robust to outliers)

Tokens absent from the corpus fall back to the global mean.

Output .pt file holds three (vocab_size,) float32 tensors + metadata.
At inference time: scores = table['mean'][token_ids]  — pure indexing.

Usage:
    python calibrate_token_salience.py \\
        --model meta-llama/Llama-3.1-8B \\
        --data_dir lb_data_raw/data \\
        --output lb_results_base/token_salience_llama31.pt \\
        --tasks all \\
        --n 100 \\
        --score_mode snapkv \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_pruned import PrunedLlamaConfig
from kl_faith_eval import (
    ENGLISH_TASKS,
    DATASET2PROMPT,
    TASK_MAX_SEQ_COMP,
    DEFAULT_MAX_SEQ_COMP,
    patch_model,
    unpatch_model,
)


# ---------------------------------------------------------------------------
# Scoring forward pass — captures per-position scores, no pruning
# ---------------------------------------------------------------------------

@torch.inference_mode()
def score_sequence(
    model,
    input_ids: torch.Tensor,   # (1, T)
    device: str,
    score_mode: str = "snapkv",
    snapkv_window: int = 32,
    n_score_layers: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Run one scoring forward pass and return (T,) mean-across-layers scores.

    Uses budget_fraction=1.0 and score_capture so the model scores every
    token without pruning anything.  Returns None on OOM.
    """
    T             = input_ids.shape[1]
    score_capture: list = []

    pcfg = PrunedLlamaConfig(
        score_mode       = score_mode,
        snapkv_window    = snapkv_window,
        min_decay        = 1.0,          # no positional bias in calibration
        budget_fraction  = 1.0,          # no pruning
        always_keep_first= 0,
        always_keep_last = 0,
        q_buffer_size    = 128,
        score_capture    = score_capture,
    )
    originals = patch_model(model, pcfg, device)

    if n_score_layers is not None:
        for i, layer in enumerate(model.model.layers):
            if i >= n_score_layers:
                layer.self_attn.head_filter = None

    try:
        _ = model(
            input_ids      = input_ids.to(device),
            attention_mask = torch.ones(1, T, device=device, dtype=torch.long),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        unpatch_model(model, originals)

    if not score_capture:
        return None

    # Average global_scores across captured layers → (T,) cpu float32
    return torch.stack(score_capture, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate per-token salience from corpus attention statistics."
    )
    p.add_argument("--model",       required=True)
    p.add_argument("--data_dir",    default="lb_data_raw/data")
    p.add_argument("--output",      required=True,
                   help="Output .pt file")
    p.add_argument("--tasks",       default="all",
                   help="Comma-separated task names or 'all'")
    p.add_argument("--n",           type=int, default=100,
                   help="Examples per task (default: 100)")
    p.add_argument("--score_mode",  default="snapkv",
                   choices=["snapkv", "kq_post_rope"],
                   help="Attention scoring mode (default: snapkv)")
    p.add_argument("--snapkv_window", type=int, default=32)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args     = parse_args()
    device   = args.device
    data_dir = Path(args.data_dir)

    if args.tasks.strip().lower() == "all":
        tasks = ENGLISH_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device,
    )
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"Vocab size: {vocab_size}")
    print(f"Tasks: {tasks}")
    print(f"Examples/task: {args.n}  Score mode: {args.score_mode}\n")

    # Per-token accumulator: token_id → list of float scores
    # Total data: ~16 tasks × 100 ex × ~4K tokens = ~6.4M scores, ~25 MB — fine in memory
    accum: Dict[int, List[float]] = defaultdict(list)

    total_seqs   = 0
    total_tokens = 0
    skipped      = 0

    for task in tasks:
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file")
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template    = DATASET2PROMPT.get(task, "{context}{input}")
        max_seq     = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        n_ex        = min(args.n, len(dataset))

        print(f"[{task}]  {n_ex} examples  max_seq={max_seq}")

        for idx in range(n_ex):
            ex      = dataset[idx]
            prompt  = template.format(
                context = ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input   = ex.get("input",   "").replace("NEWLINE_CHAR", "\n"),
            )
            full_ids = tokenizer.encode(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
            if full_ids.shape[1] > max_seq:
                full_ids = full_ids[:, -max_seq:]

            scores = score_sequence(
                model, full_ids, device,
                score_mode    = args.score_mode,
                snapkv_window = args.snapkv_window,
            )

            if scores is None:
                skipped += 1
                continue

            # Map each position's score back to its token id
            ids_list    = full_ids[0].tolist()
            scores_list = scores.tolist()
            for tok_id, score in zip(ids_list, scores_list):
                accum[tok_id].append(score)

            total_seqs   += 1
            total_tokens += len(ids_list)

            if (idx + 1) % 20 == 0 or idx == n_ex - 1:
                print(f"  ex {idx+1}/{n_ex}  "
                      f"tokens_so_far={total_tokens:,}  "
                      f"vocab_covered={len(accum):,}",
                      flush=True)

    print(f"\nDone.  sequences={total_seqs}  tokens={total_tokens:,}  "
          f"vocab_covered={len(accum):,}/{vocab_size}  skipped={skipped}")

    # ---------------------------------------------------------------------------
    # Compute statistics — mean, max, median per token-id
    # ---------------------------------------------------------------------------
    print("Computing statistics ...")

    scores_mean   = np.zeros(vocab_size, dtype=np.float32)
    scores_max    = np.zeros(vocab_size, dtype=np.float32)
    scores_median = np.zeros(vocab_size, dtype=np.float32)
    counts        = np.zeros(vocab_size, dtype=np.int32)

    for tok_id, vals in accum.items():
        if tok_id >= vocab_size:
            continue
        arr = np.array(vals, dtype=np.float32)
        scores_mean  [tok_id] = arr.mean()
        scores_max   [tok_id] = arr.max()
        scores_median[tok_id] = np.median(arr)
        counts       [tok_id] = len(arr)

    # Tokens not seen in corpus are flagged with -1.0 (sentinel).
    # All real scores are in [0, 1] so -1.0 is unambiguous.
    # Inference code can treat unseen tokens however it wants:
    #   - always keep (score = +inf), assign global mean, assign 0, etc.
    # Global statistics of seen tokens are saved separately for use as fallbacks.
    UNSEEN = -1.0
    seen_mask = counts > 0

    scores_mean  [~seen_mask] = UNSEEN
    scores_max   [~seen_mask] = UNSEEN
    scores_median[~seen_mask] = UNSEEN

    global_mean   = float(scores_mean  [seen_mask].mean()) if seen_mask.any() else 0.0
    global_max    = float(scores_max   [seen_mask].mean()) if seen_mask.any() else 0.0
    global_median = float(scores_median[seen_mask].mean()) if seen_mask.any() else 0.0

    n_seen    = int(seen_mask.sum())
    n_missing = vocab_size - n_seen
    print(f"  Seen: {n_seen:,}  Unseen (flagged -1.0): {n_missing:,}")
    print(f"  Mean   range (seen): [{scores_mean[seen_mask].min():.5f}, {scores_mean[seen_mask].max():.5f}]  global={global_mean:.5f}")
    print(f"  Max    range (seen): [{scores_max[seen_mask].min():.5f},  {scores_max[seen_mask].max():.5f}]  global={global_max:.5f}")
    print(f"  Median range (seen): [{scores_median[seen_mask].min():.5f}, {scores_median[seen_mask].max():.5f}]  global={global_median:.5f}")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Per-token scores: (vocab_size,) float32.  Unseen tokens = -1.0 (sentinel).
            "mean":          torch.from_numpy(scores_mean),
            "max":           torch.from_numpy(scores_max),
            "median":        torch.from_numpy(scores_median),
            # Appearance count per token (0 = unseen).
            "counts":        torch.from_numpy(counts),
            # Global statistics of seen tokens — use as fallback for unseen.
            "global_mean":   global_mean,
            "global_max":    global_max,
            "global_median": global_median,
            # Metadata
            "unseen_flag":   UNSEEN,
            "model":         args.model,
            "vocab_size":    vocab_size,
            "tasks":         tasks,
            "n_per_task":    args.n,
            "score_mode":    args.score_mode,
            "total_seqs":    total_seqs,
            "total_tokens":  total_tokens,
        },
        out_path,
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

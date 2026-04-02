"""
mode_compare.py
---------------
Quickly compare pruning score_modes on a targeted subset of LongBench tasks.

Loads the pruned model once, then re-runs each task with different score_mode
values by patching pcfg in-place — no model reload needed between modes.

Usage:
    python mode_compare.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --budget 1500 \
        --tasks passage_retrieval_en,hotpotqa,2wikimqa \
        --max_examples 30 \
        --modes vn_decay,kq_only,kq_vn_decay,additive
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import List

import torch
from rouge_score import rouge_scorer

from llama_pruned import PrunedLlamaConfig, build_pruned_model
from longbench_eval import (
    DATASET2MAXLEN, DATASET2SCORER, build_prompt, load_task,
    generate, truncate_to_budget,
)


def run_mode(
    mode: str,
    alpha: float,
    retention: float,
    tasks: List[str],
    pruned_model,
    pcfg: PrunedLlamaConfig,
    tokenizer,
    data_dir: Path,
    max_examples: int,
    max_seq_len: int,
    device: str,
) -> dict:
    """Run all tasks with this score_mode/retention and return per-task scores."""
    pcfg.score_mode = mode
    pcfg.score_alpha = alpha
    pcfg.budget_fraction = retention
    results = {}

    for task in tasks:
        dataset = load_task(task, data_dir)
        scorer_fn = DATASET2SCORER[task]
        max_new_tokens = DATASET2MAXLEN[task]
        n = min(len(dataset), max_examples)
        scores = []

        for idx in range(n):
            ex = dataset[idx]
            prompt = build_prompt(task, ex)
            try:
                pred = generate(pruned_model, tokenizer, prompt,
                                max_new_tokens, max_seq_len, device)
            except Exception as e:
                print(f"  WARNING: {task}[{idx}] failed: {e}")
                pred = ""
            scores.append(scorer_fn(pred, ex["answers"]))

        results[task] = 100.0 * sum(scores) / len(scores) if scores else float("nan")
        print(f"  {task}: {results[task]:.1f}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--tasks",        default="passage_retrieval_en,hotpotqa,2wikimqa")
    p.add_argument("--max_examples", type=int,   default=30)
    p.add_argument("--max_seq_len",  type=int,   default=16000)
    p.add_argument("--modes",        default="vn_decay,kq_only,additive")
    p.add_argument("--alphas",       default="0.3,0.5,0.7,0.9",
                   help="alpha values to test for additive mode")
    p.add_argument("--retentions",   default="0.50,0.65,0.80",
                   help="Fraction of tokens to keep (e.g. 0.5 = keep 50%%)")
    p.add_argument("--data_dir",     default="lb_data_raw/data")
    p.add_argument("--output",       default="mode_compare_results.json")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--min_decay",    type=float, default=0.7)
    p.add_argument("--always_keep_first", type=int, default=16)
    p.add_argument("--always_keep_last",  type=int, default=16)
    p.add_argument("--q_buffer_size",     type=int, default=128)
    args = p.parse_args()

    tasks      = [t.strip() for t in args.tasks.split(",")]
    data_dir   = Path(args.data_dir)
    alphas     = [float(a) for a in args.alphas.split(",")]
    retentions = [float(r) for r in args.retentions.split(",")]

    # Build (label, mode, alpha) list — expand "additive" into one entry per alpha
    base_modes = [m.strip() for m in args.modes.split(",")]
    modes_to_run: List[tuple] = []
    for m in base_modes:
        if m == "additive":
            for a in alphas:
                modes_to_run.append((f"additive(α={a})", "additive", a))
        else:
            modes_to_run.append((m, m, 0.5))

    pcfg = PrunedLlamaConfig(
        total_budget=999999,        # overridden by budget_fraction at runtime
        budget_fraction=0.5,        # will be overridden per run
        q_buffer_size=args.q_buffer_size,
        decay_fn="linear",
        min_decay=args.min_decay,
        always_keep_last=args.always_keep_last,
        always_keep_first=args.always_keep_first,
        score_mode="vn_decay",
    )

    print(f"Loading {args.model} ...")
    pruned_model, tokenizer = build_pruned_model(
        args.model, pcfg, device=args.device, dtype=torch.float16,
    )

    all_results = {}
    for retention in retentions:
        for label, mode, alpha in modes_to_run:
            run_label = f"r={retention:.0%} {label}"
            print(f"\n=== {run_label} ===")
            all_results[run_label] = run_mode(
                mode=mode, alpha=alpha, retention=retention,
                tasks=tasks,
                pruned_model=pruned_model, pcfg=pcfg, tokenizer=tokenizer,
                data_dir=data_dir,
                max_examples=args.max_examples,
                max_seq_len=args.max_seq_len,
                device=args.device,
            )

    # Print summary table grouped by retention
    col_w = 14
    header = f"{'Mode':<28}" + "".join(f"{t[:col_w]:>{col_w}}" for t in tasks)
    for retention in retentions:
        print(f"\n--- Retention {retention:.0%} ---")
        print(header)
        print("-" * len(header))
        for label, mode, alpha in modes_to_run:
            run_label = f"r={retention:.0%} {label}"
            scores = all_results[run_label]
            row = f"{label:<28}" + "".join(f"{scores.get(t, float('nan')):>{col_w}.1f}" for t in tasks)
            print(row)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

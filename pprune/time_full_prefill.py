"""
time_full_prefill.py
--------------------
Measure uncompressed (full-context) TTFT and TPT for LongBench tasks.

Runs the model on each example's full prompt with no compression, recording
wall-clock time to first token (TTFT) and mean time per decode token (TPT).

These serve as the baseline for compression savings tables.

Usage:
    python time_full_prefill.py \\
        --model meta-llama/Llama-3.1-8B \\
        --tasks 2wikimqa,multifieldqa_en,repobench-p,triviaqa \\
        --n 100 \\
        --output lb_results_base/timing_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kl_faith_eval import (
    ENGLISH_TASKS,
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_FULL,
    TASK_MAX_SEQ_FULL,
)


def load_task(task: str, data_dir: Path) -> List[dict]:
    path = data_dir / f"{task}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.inference_mode()
def time_one(
    model,
    input_ids: torch.Tensor,   # (1, T) — full uncompressed prompt
    max_new_tokens: int,
    device: str,
    n_decode: int = 5,
) -> Optional[dict]:
    """Return dict with ttft_ms, tpt_ms, n_tokens, or None on OOM."""
    n_decode = min(n_decode, max_new_tokens)
    ids = input_ids.to(device)

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
        )
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        kv = out.past_key_values
        del out

        step_ms: list = []
        # Generate a few tokens from the last position's predicted token
        next_tok = torch.zeros(1, 1, dtype=torch.long, device=device)
        for _ in range(n_decode):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            step_out = model(input_ids=next_tok, past_key_values=kv, use_cache=True)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - ts) * 1000)
            next_tok = step_out.logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            kv = step_out.past_key_values
        del kv

        tpt_ms = sum(step_ms) / len(step_ms) if step_ms else None
        return {"ttft_ms": ttft_ms, "tpt_ms": tpt_ms, "n_tokens": ids.shape[1]}

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Time full-context prefill on LongBench tasks.")
    p.add_argument("--model",    default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir", default="lb_data_raw/data")
    p.add_argument("--output",   default="lb_results_base/timing_full.json")
    p.add_argument("--tasks",    default="2wikimqa,multifieldqa_en,repobench-p,triviaqa")
    p.add_argument("--n",        type=int, default=100)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def main():
    args     = parse_args()
    tasks    = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results (resumable)
    results: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    for task in tasks:
        dataset      = load_task(task, data_dir)
        n_ex         = min(len(dataset), args.n)
        max_new      = DATASET2MAXLEN.get(task, 64)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        template     = DATASET2PROMPT[task]

        task_results = results.setdefault(task, {})
        done = sum(1 for i in range(n_ex) if str(i) in task_results)
        print(f"\n[{task}]  n={n_ex}  max_seq_full={max_seq_full}  already done={done}", flush=True)

        for idx in range(n_ex):
            if str(idx) in task_results:
                continue

            ex     = dataset[idx]
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
            )
            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if full_ids.shape[1] > max_seq_full:
                full_ids = full_ids[:, -max_seq_full:]

            r = time_one(model, full_ids, max_new, args.device)
            if r is None:
                print(f"  [{task}] ex={idx}  OOM — skipped", flush=True)
                task_results[str(idx)] = None
            else:
                task_results[str(idx)] = r
                print(f"  [{task}] ex={idx:>3}/{n_ex-1}  "
                      f"n_tok={r['n_tokens']:>5}  "
                      f"ttft={r['ttft_ms']:>7.0f}ms  "
                      f"tpt={r['tpt_ms']:>5.1f}ms", flush=True)

            with open(out_path, "w") as f:
                json.dump(results, f)

    # Summary table
    print("\n=== Full-context timing summary ===")
    print(f"{'Task':<26}  {'mean ttft_ms':>12}  {'mean tpt_ms':>11}  {'mean n_tok':>10}  {'N':>4}")
    print("-" * 70)
    for task in tasks:
        task_results = results.get(task, {})
        rows = [v for v in task_results.values() if v is not None]
        if not rows:
            continue
        mean_ttft = sum(r["ttft_ms"] for r in rows) / len(rows)
        mean_tpt  = sum(r["tpt_ms"]  for r in rows if r.get("tpt_ms")) / max(1, sum(1 for r in rows if r.get("tpt_ms")))
        mean_ntok = sum(r["n_tokens"] for r in rows) / len(rows)
        print(f"{task:<26}  {mean_ttft:>12.0f}  {mean_tpt:>11.1f}  {mean_ntok:>10.0f}  {len(rows):>4}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

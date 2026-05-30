"""
time_snapkv_quick.py
--------------------
Quick TTFT/TPT validation for SnapKV at 65% retention.
Runs n=20 examples per task on 5 tasks; takes ~5 min.
Compare results against pyramidkv in timing_kvpress.json.

Usage:
    python time_snapkv_quick.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from kvpress import SnapKVPress
from transformers import AutoModelForCausalLM, AutoTokenizer

from kl_faith_eval import DATASET2PROMPT, TASK_MAX_SEQ_FULL, DEFAULT_MAX_SEQ_FULL

MODEL    = "meta-llama/Llama-3.1-8B"
DATA_DIR = Path("lb_data_raw/data")
OUTPUT   = Path("lb_results_base/timing_snapkv_quick.json")
TASKS    = ["2wikimqa", "multifieldqa_en", "qasper", "repobench-p", "triviaqa"]
N        = 20
DEVICE   = "cuda"

press = SnapKVPress(compression_ratio=0.35)   # keep 65%


def load_task(task: str) -> list:
    with open(DATA_DIR / f"{task}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


@torch.inference_mode()
def time_one(model, input_ids, n_decode=5):
    ids = input_ids.to(DEVICE)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with press(model):
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        kv = out.past_key_values
        del out

        step_ms = []
        next_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        for _ in range(n_decode):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            step_out = model(input_ids=next_tok, past_key_values=kv, use_cache=True)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - ts) * 1000)
            next_tok = step_out.logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            kv = step_out.past_key_values
        del kv

        return {"ttft_ms": ttft_ms, "tpt_ms": sum(step_ms) / len(step_ms), "n_tokens": ids.shape[1]}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            results = json.load(f)

    print(f"Loading {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEVICE)
    model.eval()

    for task in TASKS:
        dataset     = load_task(task)
        max_seq     = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        template    = DATASET2PROMPT[task]
        task_results = results.setdefault(task, {})
        print(f"\n[{task}]")

        for idx in range(N):
            if str(idx) in task_results:
                continue
            ex     = dataset[idx]
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
            )
            ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if ids.shape[1] > max_seq:
                ids = ids[:, -max_seq:]

            r = time_one(model, ids)
            task_results[str(idx)] = r
            if r:
                print(f"  ex={idx:>2}/{N-1}  n_tok={r['n_tokens']:>5}  "
                      f"ttft={r['ttft_ms']:>7.0f}ms  tpt={r['tpt_ms']:>5.1f}ms", flush=True)
            else:
                print(f"  ex={idx:>2}/{N-1}  OOM", flush=True)

        with open(OUTPUT, "w") as f:
            json.dump(results, f)

    # Summary
    print("\n=== SnapKV 65% timing (n=20) ===")
    print(f"  {'Task':<26}  {'mean ttft_ms':>12}  {'mean tpt_ms':>11}  {'N':>3}")
    print("  " + "-" * 57)
    all_ttft, all_tpt = [], []
    for task in TASKS:
        rows = [v for v in results.get(task, {}).values() if v]
        if not rows:
            continue
        mt = sum(r["ttft_ms"] for r in rows) / len(rows)
        mp = sum(r["tpt_ms"]  for r in rows) / len(rows)
        all_ttft.append(mt); all_tpt.append(mp)
        print(f"  {task:<26}  {mt:>12.0f}  {mp:>11.1f}  {len(rows):>3}")
    if all_ttft:
        print(f"  {'Mean':<26}  {sum(all_ttft)/len(all_ttft):>12.0f}  "
              f"{sum(all_tpt)/len(all_tpt):>11.1f}")

    # Compare with pyramidkv from main timing run
    pyr_file = Path("lb_results_base/timing_kvpress.json")
    if pyr_file.exists():
        with open(pyr_file) as f:
            pdata = json.load(f)
        pyr_rows = [v for t in TASKS for v in pdata.get("pyramidkv", {}).get(t, {}).values() if v]
        if pyr_rows:
            p_ttft = sum(r["ttft_ms"] for r in pyr_rows) / len(pyr_rows)
            p_tpt  = sum(r["tpt_ms"]  for r in pyr_rows if r.get("tpt_ms")) / len(pyr_rows)
            print(f"\n  PyramidKV 65% (n={len(pyr_rows)}): ttft={p_ttft:.0f}ms  tpt={p_tpt:.1f}ms")
            if all_ttft:
                snap_ttft = sum(all_ttft) / len(all_ttft)
                print(f"  SnapKV    65% (n={N*len(TASKS)}): ttft={snap_ttft:.0f}ms")
                print(f"  Ratio SnapKV/PyramidKV: {snap_ttft/p_ttft:.2f}x")

    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()

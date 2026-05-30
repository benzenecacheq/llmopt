"""
time_kvpress.py
---------------
Measure TTFT and TPT for kvpress methods (PyramidKV at multiple retention rates)
on LongBench tasks, to complement the full-context baseline in timing_full.json.

TTFT here includes the full prefill pass AND the kvpress in-place compression step
that follows it — i.e., wall-clock time until the first decode token can be generated.
TPT is measured against the compressed KV cache.

Usage:
    python time_kvpress.py \\
        --model meta-llama/Llama-3.1-8B \\
        --tasks 2wikimqa,multifieldqa_en,qasper,repobench-p,triviaqa \\
        --methods pyramidkv,pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \\
        --n 100 \\
        --output lb_results_base/timing_kvpress.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

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
from kl_faith_eval_ystar import _KVPRESS_METHODS


def load_task(task: str, data_dir: Path) -> list:
    path = data_dir / f"{task}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.inference_mode()
def time_one(
    model,
    input_ids: torch.Tensor,
    press,
    device: str,
    n_decode: int = 5,
) -> Optional[dict]:
    """
    Measure TTFT (prefill + kvpress compression) and TPT (decode against
    compressed KV cache). Returns None on OOM.
    """
    ids = input_ids.to(device)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with press(model):
            out = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
            )
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        kv = out.past_key_values
        del out

        step_ms = []
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

        # KV cache size after compression (layer 0 key length)
        compressed_len = None
        try:
            compressed_len = kv.layers[0].keys.shape[-2]
        except Exception:
            pass

        tpt_ms = sum(step_ms) / len(step_ms) if step_ms else None
        return {
            "ttft_ms": ttft_ms,
            "tpt_ms": tpt_ms,
            "n_tokens": ids.shape[1],
            "compressed_len": compressed_len,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir", default="lb_data_raw/data")
    p.add_argument("--output",   default="lb_results_base/timing_kvpress.json")
    p.add_argument("--tasks",    default="2wikimqa,multifieldqa_en,qasper,repobench-p,triviaqa")
    p.add_argument("--methods",  default="pyramidkv,pyramidkv_f50,pyramidkv_f40,pyramidkv_f35")
    p.add_argument("--n",        type=int, default=100)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def main():
    args    = parse_args()
    tasks   = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in _KVPRESS_METHODS]
    if not methods:
        raise SystemExit(f"No valid kvpress methods in: {args.methods}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    for method in methods:
        press = _KVPRESS_METHODS[method]
        print(f"\n{'='*60}\nMethod: {method}")
        method_results = results.setdefault(method, {})

        for task in tasks:
            dataset      = load_task(task, Path(args.data_dir))
            n_ex         = min(len(dataset), args.n)
            max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
            template     = DATASET2PROMPT[task]

            task_results = method_results.setdefault(task, {})
            done = sum(1 for i in range(n_ex) if str(i) in task_results)
            print(f"\n[{task}]  n={n_ex}  max_seq={max_seq_full}  done={done}", flush=True)

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

                r = time_one(model, full_ids, press, args.device)
                if r is None:
                    print(f"  ex={idx}  OOM — skipped", flush=True)
                    task_results[str(idx)] = None
                else:
                    task_results[str(idx)] = r
                    print(f"  ex={idx:>3}/{n_ex-1}  "
                          f"n_tok={r['n_tokens']:>5}  "
                          f"ttft={r['ttft_ms']:>7.0f}ms  "
                          f"tpt={r['tpt_ms']:>5.1f}ms", flush=True)

                with open(out_path, "w") as f:
                    json.dump(results, f)

    # Summary table
    print("\n=== kvpress timing summary ===")
    for method in methods:
        print(f"\n{method}:")
        print(f"  {'Task':<26}  {'mean ttft_ms':>12}  {'mean tpt_ms':>11}  {'mean n_tok':>10}  {'N':>4}")
        print("  " + "-" * 66)
        for task in tasks:
            rows = [v for v in results.get(method, {}).get(task, {}).values() if v]
            if not rows:
                continue
            mean_ttft = sum(r["ttft_ms"] for r in rows) / len(rows)
            mean_tpt  = sum(r["tpt_ms"] for r in rows if r.get("tpt_ms")) / max(1, sum(1 for r in rows if r.get("tpt_ms")))
            mean_ntok = sum(r["n_tokens"] for r in rows) / len(rows)
            print(f"  {task:<26}  {mean_ttft:>12.0f}  {mean_tpt:>11.1f}  {mean_ntok:>10.0f}  {len(rows):>4}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

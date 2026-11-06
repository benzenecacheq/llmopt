"""
evict_first_eval.py
-------------------
Quantify the first-token advantage by comparing F_out for:

  full        -- full model, no compression
  snapkv      -- standard SnapKV (first token from full prefill, then evict)
  evict_first -- SnapKV but evict *before* first token: first token attends
                 to the enriched-but-evicted cache (uses generate_clean_first)

Output: lb_results_base/evict_first_fout.json
  { "task|method|idx": f_out_score, ... }

Usage:
    conda run -n llmopt python evict_first_eval.py [--cr 0.35] [--n 20]
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from kvpress.presses.snapkv_press import SnapKVPress

from gt_eval_compression import generate_clean_first, generate_from_ids
from kl_faith_eval import DATASET2PROMPT, TASK_MAX_SEQ_FULL, DEFAULT_MAX_SEQ_FULL
from longbench_eval import qa_f1_score

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"

TASKS_DEFAULT = [
    "narrativeqa",
    "gov_report",
    "hotpotqa",
    "passage_retrieval_en",
    "trec",
]

MAX_NEW_TOKENS = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS_DEFAULT))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--cr", type=float, default=0.35,
                        help="compression ratio (fraction to EVICT); 0.35 = keep 65%%")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--output", default="lb_results_base/evict_first_fout.json")
    parser.add_argument("--data_dir", default="lb_data_raw/data")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        results: dict = json.loads(out_path.read_text())
        done = set(results.keys())
        print(f"Resuming — {len(done)} entries already saved.")
    else:
        results, done = {}, set()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    press = SnapKVPress(
        compression_ratio=args.cr,
        window_size=args.window_size,
    )

    data_dir = Path(args.data_dir)
    tasks = [t.strip() for t in args.tasks.split(",")]

    for task in tasks:
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — not found"); continue

        dataset = [json.loads(l) for l in data_file.read_text().splitlines() if l.strip()]
        template = DATASET2PROMPT.get(task, "{context}{input}")
        max_seq = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex = min(args.n, len(dataset))
        print(f"\n[{task}]  n={n_ex}  cr={args.cr}  max_seq={max_seq}")

        for idx in range(n_ex):
            keys_needed = [
                f"{task}|full|{idx}",
                f"{task}|snapkv|{idx}",
                f"{task}|evict_first|{idx}",
            ]
            if all(k in done for k in keys_needed):
                continue

            ex = dataset[idx]
            prompt = template.format(**ex)
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_seq
            )["input_ids"]

            torch.cuda.empty_cache()

            # ── full model ─────────────────────────────────────────────────────
            key_full = f"{task}|full|{idx}"
            if key_full not in done:
                pred_full = generate_from_ids(
                    model, tokenizer, input_ids, MAX_NEW_TOKENS, max_seq, DEVICE
                )
                results[key_full] = pred_full
                done.add(key_full)
                print(f"  [{idx}] full: {pred_full[:60]!r}")
                torch.cuda.empty_cache()

            # ── standard SnapKV (first token from full prefill) ────────────────
            key_snap = f"{task}|snapkv|{idx}"
            if key_snap not in done:
                pred_snap = generate_from_ids(
                    model, tokenizer, input_ids, MAX_NEW_TOKENS, max_seq, DEVICE,
                    press=press,
                )
                results[key_snap] = pred_snap
                done.add(key_snap)
                print(f"  [{idx}] snapkv: {pred_snap[:60]!r}")
                torch.cuda.empty_cache()

            # ── evict-first SnapKV (first token from evicted cache) ────────────
            key_ef = f"{task}|evict_first|{idx}"
            if key_ef not in done:
                pred_ef = generate_clean_first(
                    model, tokenizer, input_ids, MAX_NEW_TOKENS, max_seq, DEVICE,
                    press=press,
                )
                results[key_ef] = pred_ef
                done.add(key_ef)
                print(f"  [{idx}] evict_first: {pred_ef[:60]!r}")
                torch.cuda.empty_cache()

            out_path.write_text(json.dumps(results))

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n=== F_out summary (vs full model) ===")
    for method in ("snapkv", "evict_first"):
        scores = []
        for task in tasks:
            for idx in range(args.n):
                kf = f"{task}|full|{idx}"
                km = f"{task}|{method}|{idx}"
                if kf in results and km in results:
                    scores.append(qa_f1_score(results[km], [results[kf]]) * 100)
        if scores:
            print(f"  {method}: {sum(scores)/len(scores):.1f}%  (n={len(scores)})")

    print(f"\nDone. {len(results)} entries → {out_path}")


if __name__ == "__main__":
    main()

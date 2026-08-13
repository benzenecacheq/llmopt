"""
gt_eval_instruct.py
-------------------
Ground-truth LongBench evaluation for instruct (or base) models with absolute token budgets.

Compression ratio is computed per-example as max(0, 1 - budget/T).
Re-rotated methods use a corrected decode loop that properly handles instruct
model EOS tokens (including <|eot_id|> in addition to <|end_of_text|>).

Usage:
    python gt_eval_instruct.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --budget 512 --window_size 32 --instruct \\
        --output lb_results_base/gt_instruct_llama_b512/ \\
        --n 100

    # Score only (no inference):
    python gt_eval_instruct.py \\
        --output lb_results_base/gt_instruct_llama_b512/ --score_only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from kvpress import KeyRerotationPress as _KRP
    _KVPRESS_AVAILABLE = True
except ImportError:
    _KRP = None
    _KVPRESS_AVAILABLE = False

from kl_faith_eval import (
    ENGLISH_TASKS, DATASET2PROMPT, DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_FULL, TASK_MAX_SEQ_FULL,
    load_ckpt, save_ckpt, fmt_duration,
    naive_truncate,
)
from kl_faith_eval_ystar import PyramidKVRerotationPress
from gt_eval_compression import (
    generate_from_ids,
    score_results, print_table, ckpt_key,
)
from kl_faith_eval_instruct import make_press, encode_prompt, _INSTRUCT_METHOD_TYPES


# ---------------------------------------------------------------------------
# EOS helpers
# ---------------------------------------------------------------------------

def _eos_set(model, tokenizer) -> Set[int]:
    """All stop token IDs for this model (handles instruct models with multiple EOS)."""
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, list):
        return set(eos)
    return {tokenizer.eos_token_id}


# ---------------------------------------------------------------------------
# Generation for re-rotated methods (manual decode loop)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate_rerotated(
    model,
    tokenizer,
    full_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
    press,
    eos_ids: Set[int],
) -> str:
    """Generation for KeyRerotationPress / PyramidKVRerotationPress methods.

    Equivalent to gt_eval_compression.generate_rerotated but accepts an explicit
    eos_ids set, so instruct models stop correctly at <|eot_id|> in addition to
    <|end_of_text|>.
    """
    cap = max_seq_len - max_new_tokens
    if full_ids.shape[1] > cap:
        full_ids = full_ids[:, -cap:]
    full_ids = full_ids.to(device)

    for attempt in range(4):
        try:
            T = full_ids.shape[1]

            with press(model):
                out = model(
                    full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=True,
                    return_dict=True,
                )
            compressed_kv = out.past_key_values
            M = T if isinstance(press, PyramidKVRerotationPress) else compressed_kv.get_seq_length()
            T1 = out.logits[0, -1].argmax().reshape(1, 1)
            kv = compressed_kv
            del out
            torch.cuda.empty_cache()

            generated_ids = [T1.item()]
            cur_tok = T1
            for step in range(max_new_tokens - 1):
                cache_pos = torch.tensor([M + step], dtype=torch.long, device=device)
                out_s = model(
                    cur_tok,
                    past_key_values=kv,
                    cache_position=cache_pos,
                    use_cache=True,
                    return_dict=True,
                )
                next_tok = out_s.logits[0, -1].argmax().reshape(1, 1)
                kv = out_s.past_key_values
                del out_s
                tok_id = next_tok.item()
                generated_ids.append(tok_id)
                cur_tok = next_tok
                if tok_id in eos_ids:
                    break

            del kv
            torch.cuda.empty_cache()
            # Trim trailing EOS tokens before decoding
            while generated_ids and generated_ids[-1] in eos_ids:
                generated_ids.pop()
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            full_ids = full_ids[:, full_ids.shape[1] // 2:]
            if attempt == 3 or full_ids.shape[1] < 64:
                raise


# ---------------------------------------------------------------------------
# Inference dispatcher
# ---------------------------------------------------------------------------

def run_one(
    model,
    tokenizer,
    full_ids: torch.Tensor,
    method: str,
    budget: int,
    window_size: int,
    max_seq_full: int,
    max_new_tokens: int,
    device: str,
    eos_ids: Set[int],
) -> str:
    """Dispatch a single (example, method) → prediction string."""
    T = full_ids.shape[1]

    if method == "full":
        return generate_from_ids(model, tokenizer, full_ids, max_new_tokens, max_seq_full, device)

    if method == "naive":
        fraction = min(1.0, budget / max(T, 1))
        comp_ids = naive_truncate(full_ids, fraction=fraction, head_frac=0.10)
        return generate_from_ids(model, tokenizer, comp_ids, max_new_tokens, max_seq_full, device)

    press = make_press(method, budget, T, window_size)

    is_rerotated = isinstance(press, PyramidKVRerotationPress) or (
        _KVPRESS_AVAILABLE and _KRP is not None and isinstance(press, _KRP)
    )
    if is_rerotated:
        return _generate_rerotated(
            model, tokenizer, full_ids, max_new_tokens, max_seq_full, device,
            press, eos_ids,
        )

    return generate_from_ids(model, tokenizer, full_ids, max_new_tokens, max_seq_full, device,
                             press=press)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    tasks: List[str],
    methods: List[str],
    budget: int,
    window_size: int,
    instruct: bool,
    data_dir: Path,
    ckpt_path: Path,
    max_examples: int,
    device: str,
):
    ckpt    = load_ckpt(ckpt_path)
    eos_ids = _eos_set(model, tokenizer)
    print(f"EOS token IDs: {eos_ids}")

    for t_idx, task in enumerate(tasks):
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file")
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template     = DATASET2PROMPT[task]
        max_new      = DATASET2MAXLEN.get(task, 64)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex         = min(max_examples, len(dataset))

        done = sum(
            1 for idx in range(n_ex)
            if all(ckpt_key(task, idx, m) in ckpt for m in methods)
        )
        print(f"\n[{task}  {t_idx+1}/{len(tasks)}]  ({done}/{n_ex} fully done)", flush=True)

        for idx in range(n_ex):
            needed = [m for m in methods if ckpt_key(task, idx, m) not in ckpt]
            if not needed:
                continue

            ex = dataset[idx]
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
            )
            full_ids = encode_prompt(tokenizer, prompt, instruct, max_seq_full=max_seq_full)

            for method in needed:
                t0 = time.perf_counter()
                try:
                    pred = run_one(
                        model, tokenizer, full_ids, method, budget, window_size,
                        max_seq_full, max_new, device, eos_ids,
                    )
                except Exception as e:
                    print(f"  WARNING [{task}][{idx}][{method}]: {e}", flush=True)
                    traceback.print_exc()
                    pred = ""

                elapsed = time.perf_counter() - t0
                ckpt[ckpt_key(task, idx, method)] = {
                    "prediction": pred,
                    "answers":    ex["answers"],
                }
                save_ckpt(ckpt, ckpt_path)
                print(
                    f"  [{task}] ex={idx}/{n_ex-1}  {method:<22}  {elapsed:.1f}s"
                    f"  → {repr(pred[:60])}",
                    flush=True,
                )


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
    p.add_argument("--model",       default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data_dir",    default="lb_data_raw/data")
    p.add_argument("--output",      default="lb_results_base/gt_instruct/",
                   help="Output directory; checkpoint.json goes here")
    p.add_argument("--log",         default=None)
    p.add_argument("--budget",      type=int, default=512,
                   help="Absolute tokens retained per layer (compression_ratio = 1 - budget/T)")
    p.add_argument("--window_size", type=int, default=32,
                   help="SnapKV/PyramidKV observation window")
    p.add_argument("--instruct",    action="store_true",
                   help="Apply chat template (set for instruct models)")
    p.add_argument("--tasks",       default=",".join(ENGLISH_TASKS))
    p.add_argument("--methods",
                   default="snapkv,snapkv_rot,pyramidkv,pyramidkv_rot,"
                           "streaming,streaming_rot,naive,full")
    p.add_argument("--n",           type=int, default=100)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--score_only",  action="store_true",
                   help="Skip inference; just score the checkpoint")
    return p.parse_args()


def main():
    args     = parse_args()
    tasks    = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    methods  = [m.strip() for m in args.methods.split(",") if m.strip() in _INSTRUCT_METHOD_TYPES]
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    print(f"Tasks    : {tasks}")
    print(f"Methods  : {methods}")
    print(f"Budget   : {args.budget} tokens, window_size={args.window_size}, instruct={args.instruct}")
    print(f"Checkpoint: {ckpt_path}")

    if not args.score_only:
        print(f"\nLoading {args.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=args.device,
        )
        model.eval()

        try:
            run_eval(
                model, tokenizer,
                tasks=tasks,
                methods=methods,
                budget=args.budget,
                window_size=args.window_size,
                instruct=args.instruct,
                data_dir=Path(args.data_dir),
                ckpt_path=ckpt_path,
                max_examples=args.n,
                device=args.device,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved. Re-run to resume.")
            sys.exit(0)

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Score and report
    # ------------------------------------------------------------------
    print("\n=== Ground-truth scores ===")
    ckpt = load_ckpt(ckpt_path)
    results = score_results(tasks, ckpt, args.n)
    print_table(results, method_order=methods)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

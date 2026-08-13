"""
kl_faith_eval_instruct.py
-------------------------
KL faithfulness evaluation for instruct (or base) models with absolute token budgets.

Creates press instances dynamically per example:
  compression_ratio = max(0, 1 - budget/T)
so the retained count is exactly `budget` tokens regardless of input length.

Supported methods: snapkv, snapkv_rot, pyramidkv, pyramidkv_rot,
                   streaming, streaming_rot, naive, full

Usage:
    python kl_faith_eval_instruct.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --budget 512 --window_size 32 --instruct \\
        --output lb_results_base/kl_instruct_llama_b512.json \\
        --ystar_cache lb_results_base/ystar_instruct_llama.pt \\
        --n 100
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

try:
    from kvpress import PyramidKVPress as _PyramidKVPress
    from kvpress import SnapKVPress as _SnapKVPress
    from kvpress import StreamingLLMPress as _StreamingLLMPress
    from kvpress import KeyRerotationPress as _KeyRerotationPress
    _KVPRESS_AVAILABLE = True
except ImportError:
    _PyramidKVPress = _SnapKVPress = _StreamingLLMPress = _KeyRerotationPress = None
    _KVPRESS_AVAILABLE = False

from kl_faith_eval import (
    ENGLISH_TASKS, DATASET2PROMPT, DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_FULL, TASK_MAX_SEQ_FULL,
    load_ckpt, save_ckpt, fmt_duration, fmt_eta,
    naive_truncate,
)
from kl_faith_eval_ystar import (
    PyramidKVRerotationPress,
    generate_ystar, get_comp_log_probs, kl_divergence,
    load_ystar_cache, save_ystar_cache,
    get_from_cache, put_in_cache,
    YSTAR_STOP_TOKENS,
)

# ---------------------------------------------------------------------------
# Press factory
# ---------------------------------------------------------------------------

_INSTRUCT_METHOD_TYPES = frozenset({
    "snapkv", "snapkv_rot",
    "pyramidkv", "pyramidkv_rot",
    "streaming", "streaming_rot",
    "naive", "full",
})


def make_press(method: str, budget: int, T: int, window_size: int = 32):
    """Return a fresh press retaining exactly `budget` tokens from T.

    Returns None for non-kvpress methods (naive, full).
    """
    if method in ("full", "naive"):
        return None
    if not _KVPRESS_AVAILABLE:
        raise RuntimeError("kvpress not installed")
    cr = max(0.0, 1.0 - budget / T)
    if method == "snapkv":
        return _SnapKVPress(compression_ratio=cr, window_size=window_size)
    elif method == "snapkv_rot":
        return _KeyRerotationPress(press=_SnapKVPress(compression_ratio=cr, window_size=window_size))
    elif method == "pyramidkv":
        return _PyramidKVPress(compression_ratio=cr, window_size=window_size, beta=20)
    elif method == "pyramidkv_rot":
        return PyramidKVRerotationPress(compression_ratio=cr, window_size=window_size, beta=20)
    elif method == "streaming":
        return _StreamingLLMPress(compression_ratio=cr, n_sink=4)
    elif method == "streaming_rot":
        return _KeyRerotationPress(press=_StreamingLLMPress(compression_ratio=cr, n_sink=4))
    else:
        raise ValueError(f"Unknown method: {method}")


def encode_prompt(tokenizer, prompt: str, instruct: bool, max_seq_full: int = None) -> torch.Tensor:
    """Tokenize prompt, optionally wrapping in the chat template.

    When instruct=True and max_seq_full is given, the document content is
    truncated *before* the template is applied so the header/footer tokens
    are always present even for very long documents.  Left-truncation of the
    fully-formatted sequence would silently drop the user-header tokens.
    """
    if not instruct:
        return tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

    if max_seq_full is not None:
        overhead_ids = tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False, add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        max_content = max(1, max_seq_full - len(overhead_ids))
        content_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(content_ids) > max_content:
            content_ids = content_ids[-max_content:]
            prompt = tokenizer.decode(content_ids, skip_special_tokens=True)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")


# ---------------------------------------------------------------------------
# Eval loop
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
    output_path: Path,
    max_examples: int,
    device: str,
    ystar_cache_path: Optional[Path] = None,
    max_new_tokens_override: Optional[int] = None,
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
        if isinstance(v, float)
    )

    print(f"\n{'='*72}")
    print(f"KL faithfulness — budget={budget} tokens/layer, window_size={window_size}, instruct={instruct}")
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

        template     = DATASET2PROMPT.get(task, "{context}{input}")
        max_new      = max_new_tokens_override or DATASET2MAXLEN.get(task, 64)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex         = min(max_examples, len(dataset))

        print(f"[{task}  {t_idx+1}/{n_tasks}]")

        for idx in range(n_ex):
            if all(f"{task}|{m}|{idx}" in ckpt for m in methods):
                grand_done += n_methods
                continue

            ex = dataset[idx]
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=question_text,
            )

            full_ids = encode_prompt(tokenizer, prompt, instruct, max_seq_full=max_seq_full)

            # y*: generate from full (uncompressed) model; cache across methods
            ystar, log_p_full = get_from_cache(cache, task, idx)
            if ystar is None:
                # Instruct models stop naturally at <|eot_id|>; skip per-task stop tokens
                stop_toks = None if instruct else YSTAR_STOP_TOKENS.get(task)
                ystar, log_p_full, _ = generate_ystar(
                    model, full_ids, max_new, max_seq_full, device,
                    stop_token_ids=stop_toks,
                )
                if ystar is not None and ystar_cache_path:
                    put_in_cache(cache, task, idx, ystar, log_p_full)
                    save_ystar_cache(cache, ystar_cache_path)
                    print(f"  [cached y* for {task}|{idx}  n_gen={ystar.shape[0]}]", flush=True)
            else:
                print(f"  [y* cache hit: {task}|{idx}  n_gen={ystar.shape[0]}]", flush=True)

            if ystar is None:
                for m in methods:
                    ckpt.setdefault(f"{task}|{m}|{idx}", None)
                save_ckpt(ckpt, output_path)
                grand_done += n_methods
                continue

            n_gen = ystar.shape[0]
            T     = full_ids.shape[1]

            for method in methods:
                ck_key = f"{task}|{method}|{idx}"
                if ck_key in ckpt:
                    grand_done += 1
                    continue

                if method == "naive":
                    # Cap to the same window as kvpress methods before selecting,
                    # so naive draws from the same token pool as the reference model.
                    capped = full_ids[:, -max_seq_full:] if full_ids.shape[1] > max_seq_full else full_ids
                    fraction = min(1.0, budget / max(capped.shape[1], 1))
                    comp_ids = naive_truncate(capped, fraction=fraction, head_frac=0.10)
                    press = None
                else:
                    comp_ids = full_ids
                    if comp_ids.shape[1] > max_seq_full:
                        comp_ids = comp_ids[:, -max_seq_full:]
                    T_eff = comp_ids.shape[1]
                    press = make_press(method, budget, T_eff, window_size) if method != "full" else None

                if method == "full":
                    log_p_comp = log_p_full
                else:
                    log_p_comp = get_comp_log_probs(model, comp_ids, ystar, None, device, press=press)

                if log_p_comp is not None:
                    kl = kl_divergence(log_p_full, log_p_comp)
                    ckpt[ck_key] = kl
                else:
                    ckpt[ck_key] = None
                    kl = None

                save_ckpt(ckpt, output_path)
                grand_done += 1

                elapsed   = time.time() - grand_start
                grand_eta = fmt_eta(
                    grand_done - completed_at_start,
                    total_work - completed_at_start, elapsed
                )
                kl_str = f"{kl:.4f}" if kl is not None else " OOM"
                print(
                    f"  ex {idx+1:>4}/{n_ex}  {method:<22}"
                    f"  T={T}  n_gen={n_gen}  KL={kl_str}"
                    f"  [{grand_eta} total]",
                    flush=True,
                )

        # Task summary
        print(f"\n  --- {task} summary ---")
        for m in methods:
            scores = [ckpt[f"{task}|{m}|{i}"] for i in range(n_ex)
                      if isinstance(ckpt.get(f"{task}|{m}|{i}"), float)]
            mean = np.mean(scores) if scores else float("nan")
            print(f"  {m:<24}  n={len(scores)}  mean_KL={mean:.4f}")
        print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"SUMMARY — KL(P_full || P_comp), budget={budget}, window_size={window_size}")
    print(f"{'='*72}")
    col_w = 12
    header = f"{'Task':<26}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(header); print("-" * len(header))

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
            for m in methods)
        print(row)

    print("-" * len(header))
    avg = f"{'AVERAGE':<26}  " + "  ".join(
        f"{np.mean(per_method_all[m]):>{col_w}.4f}" if per_method_all[m] else f"{'—':>{col_w}}"
        for m in methods)
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
    p.add_argument("--model",          default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data_dir",       default="lb_data_raw/data")
    p.add_argument("--output",         default="lb_results_base/kl_instruct.json")
    p.add_argument("--log",            default=None)
    p.add_argument("--budget",         type=int, default=512,
                   help="Absolute tokens retained per layer (compression_ratio = 1 - budget/T per example)")
    p.add_argument("--window_size",    type=int, default=32,
                   help="SnapKV/PyramidKV observation window (paper default = 32)")
    p.add_argument("--instruct",       action="store_true",
                   help="Apply chat template (set for instruct models)")
    p.add_argument("--tasks",          default=",".join(ENGLISH_TASKS))
    p.add_argument("--methods",
                   default="snapkv,snapkv_rot,pyramidkv,pyramidkv_rot,"
                           "streaming,streaming_rot,naive,full")
    p.add_argument("--n",              type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--ystar_cache",    default=None,
                   help="Path to y* cache .pt file (shared across budgets for same model+tasks)")
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    tasks   = [t.strip() for t in args.tasks.split(",")   if t.strip() in ENGLISH_TASKS]
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in _INSTRUCT_METHOD_TYPES]

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
        budget=args.budget,
        window_size=args.window_size,
        instruct=args.instruct,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        max_examples=args.n,
        device=args.device,
        ystar_cache_path=Path(args.ystar_cache) if args.ystar_cache else None,
        max_new_tokens_override=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

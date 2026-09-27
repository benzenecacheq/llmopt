#!/usr/bin/env python3
"""Per-step top-k logit analysis for gov_report.

Reuses kl_faith_eval_ystar.py infrastructure to save per-step top-k (k=20)
token IDs and log-probs from both P_full (uncompressed) and P_comp (compressed)
alongside KL, enabling analysis of why high-KL methods (SnapKV, PyramidKV) still
produce correct tokens more often than low-KL rotated variants.

Target: gov_report, n=30, 4 methods:
  snapkv_press, snapkv_rerotated, pyramidkv, pyramidkv_rerotated

Output: lb_results_base/topk_logits_gov_report.pt
  Dict keyed by "task|method|idx", each entry:
    {
      "kl": float,
      "ystar_toks": tensor (n_gen,) int64,
      "topk_P_ids": tensor (n_gen, k) int64,
      "topk_P_lp":  tensor (n_gen, k) float32,
      "topk_Q_ids": tensor (n_gen, k) int64,
      "topk_Q_lp":  tensor (n_gen, k) float32,
    }
"""

import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse all infrastructure from the existing eval script
from kl_faith_eval_ystar import (
    _KVPRESS_METHODS,
    get_comp_log_probs,
    load_ystar_cache,
    get_from_cache,
    make_comp_ids,
    kl_divergence,
    fmt_duration,
    PyramidKVRerotationPress,
)
from kl_faith_eval import (
    ENGLISH_TASKS,
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_COMP,
    DEFAULT_MAX_SEQ_FULL,
    TASK_MAX_SEQ_COMP,
    TASK_MAX_SEQ_FULL,
    METHOD_CONFIGS,
    load_ckpt,
    save_ckpt,
    fmt_eta,
)

TOPK = 20
TARGET_TASKS   = ["gov_report"]
TARGET_METHODS = ["snapkv_press", "snapkv_rerotated", "pyramidkv", "pyramidkv_rerotated"]
DEFAULT_N      = 30
DEFAULT_OUTPUT = "lb_results_base/topk_logits_gov_report.pt"
DEFAULT_YSTAR  = "lb_results_base/ystar_cache_v3.pt"


def extract_topk(log_probs: torch.Tensor, k: int = TOPK):
    """Return (ids, log_probs) each (n_gen, k) from a (n_gen, vocab) log-prob tensor."""
    topk_lp, topk_ids = torch.topk(log_probs, k, dim=-1)
    return topk_ids.cpu(), topk_lp.cpu()


def load_results(path: Path) -> dict:
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded existing results: {path}  ({len(data)} entries)")
        return data
    return {}


def save_results(data: dict, path: Path):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(data, tmp)
    tmp.replace(path)


def run_topk_eval(
    model,
    tokenizer,
    tasks: List[str],
    methods: List[str],
    data_dir: Path,
    output_path: Path,
    max_examples: int,
    device: str,
    ystar_cache_path: Path,
    topk: int = TOPK,
):
    results = load_results(output_path)
    cache   = load_ystar_cache(ystar_cache_path)
    print(f"y* cache: {ystar_cache_path}  ({len(cache)} entries loaded)")

    total_work = len(tasks) * max_examples * len(methods)
    completed  = len(results)
    grand_start = time.time()
    grand_done  = completed

    print(f"\n{'='*72}")
    print(f"Top-k logit eval (k={topk})")
    print(f"  Tasks  : {tasks}")
    print(f"  Methods: {methods}")
    print(f"  N/task : {max_examples}")
    print(f"  Total  : {total_work} entries  ({completed} already done)")
    print(f"{'='*72}\n")

    for task in tasks:
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file")
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template     = DATASET2PROMPT.get(task, "{context}{input}")
        max_new      = DATASET2MAXLEN.get(task, 512)
        max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex         = min(max_examples, len(dataset))

        for idx in range(n_ex):
            # Load y* and P_full from cache
            ystar, log_p_full = get_from_cache(cache, task, idx)
            if ystar is None:
                print(f"  [{task}|{idx}] SKIP — not in y* cache (run kl_faith_eval_ystar.py first)")
                grand_done += len(methods)
                continue

            n_gen = ystar.shape[0]
            topk_P_ids, topk_P_lp = extract_topk(log_p_full, topk)  # (n_gen, k)

            ex = dataset[idx]
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=question_text,
            )
            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

            for method in methods:
                ck_key = f"{task}|{method}|{idx}"
                if ck_key in results:
                    grand_done += 1
                    continue

                press = _KVPRESS_METHODS.get(method)
                pcfg  = METHOD_CONFIGS.get(method)

                effective_max = max_seq_full if press is not None else max_seq_comp
                comp_ids = make_comp_ids(
                    full_ids, method, tokenizer, question_text, effective_max,
                    model=model, device=device,
                )

                log_p_comp = get_comp_log_probs(model, comp_ids, ystar, pcfg, device, press=press)

                if log_p_comp is not None:
                    kl = kl_divergence(log_p_full, log_p_comp)
                    topk_Q_ids, topk_Q_lp = extract_topk(log_p_comp, topk)
                    results[ck_key] = {
                        "kl":         kl,
                        "ystar_toks": ystar.cpu(),
                        "topk_P_ids": topk_P_ids,
                        "topk_P_lp":  topk_P_lp,
                        "topk_Q_ids": topk_Q_ids,
                        "topk_Q_lp":  topk_Q_lp,
                    }
                else:
                    results[ck_key] = None
                    kl = None

                save_results(results, output_path)
                grand_done += 1

                elapsed   = time.time() - grand_start
                grand_eta = fmt_eta(grand_done - completed, total_work - completed, elapsed)
                kl_str    = f"{kl:.4f}" if kl is not None else " OOM"
                print(
                    f"  {task}  ex {idx+1:>4}/{n_ex}  {method:<22}"
                    f"  n_gen={n_gen}  KL={kl_str}  [{grand_eta} total]",
                    flush=True,
                )

    total_elapsed = time.time() - grand_start
    print(f"\nTotal wall time: {fmt_duration(total_elapsed)}")
    print(f"Results saved → {output_path}")

    # Summary
    print(f"\n{'='*72}")
    print(f"SUMMARY — mean KL(P_full || P_comp)")
    for task in tasks:
        for method in methods:
            scores = [
                results[f"{task}|{method}|{i}"]["kl"]
                for i in range(max_examples)
                if isinstance(results.get(f"{task}|{method}|{i}"), dict)
                and results[f"{task}|{method}|{i}"].get("kl") is not None
            ]
            mean = np.mean(scores) if scores else float("nan")
            print(f"  {task}  {method:<24}  n={len(scores)}  mean_KL={mean:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir",   default="lb_data_raw/data")
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--ystar_cache",default=DEFAULT_YSTAR)
    p.add_argument("--tasks",      default=",".join(TARGET_TASKS))
    p.add_argument("--methods",    default=",".join(TARGET_METHODS))
    p.add_argument("--n",          type=int, default=DEFAULT_N)
    p.add_argument("--topk",       type=int, default=TOPK)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    tasks   = [t.strip() for t in args.tasks.split(",")   if t.strip() in ENGLISH_TASKS]
    _known  = set(_KVPRESS_METHODS) | set(METHOD_CONFIGS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in _known]

    if not tasks:   print("No valid tasks.");   return
    if not methods: print("No valid methods."); return

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    run_topk_eval(
        model         = model,
        tokenizer     = tokenizer,
        tasks         = tasks,
        methods       = methods,
        data_dir      = Path(args.data_dir),
        output_path   = Path(args.output),
        max_examples  = args.n,
        device        = args.device,
        ystar_cache_path = Path(args.ystar_cache),
        topk          = args.topk,
    )


if __name__ == "__main__":
    main()

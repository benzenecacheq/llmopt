"""
distillation_diagnostic.py
--------------------------
Tests the hypothesis that PyramidKV acts as query-focused distillation:
its logit distribution DURING DECODE is more sharply peaked toward the correct
answer than the full model's distribution.

Key insight: kvpress only compresses the *stored* KV cache during prefill —
it does NOT change the prefill logits (which are computed from full dense
attention regardless). The difference shows up at decode step 1 and beyond,
when the compressed cache is used to attend.

Therefore we use model.generate(output_scores=True) and compare decode-step
logits (scores[1+]) not prefill logits (scores[0], which are identical).

For each QA example we compute, at each decode position >= 1:
  - p_full(GT_token)    : full model probability on that GT answer token
  - p_pyr(GT_token)     : PyramidKV probability on that GT answer token
  - entropy_full        : entropy of full model's distribution (nats)
  - entropy_pyr         : entropy of PyramidKV's distribution

Expected under distillation hypothesis:
  p_pyr(GT) > p_full(GT)   (PyramidKV focuses on answer-relevant tokens)
  entropy_pyr < entropy_full  (sharper distribution toward correct answer)

Note: step 0 logits are identical (from prefill) and are excluded from stats.

Usage:
    python distillation_diagnostic.py --tasks triviaqa,2wikimqa,hotpotqa --n 100
    python distillation_diagnostic.py --tasks triviaqa --n 50 --out distill_diag.json
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kl_faith_eval import (
    ENGLISH_TASKS,
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_FULL,
)
from gt_eval_compression import load_task

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

try:
    from kvpress import PyramidKVPress as _PyramidKVPress
    _KVPRESS_AVAILABLE = True
except ImportError:
    _KVPRESS_AVAILABLE = False

# Cap prompt length to fit two generate() passes on a 31.73 GB V100.
# kl_faith_eval uses 4096 for the compressed model — match that cap here.
MAX_SEQ = 4096
MAX_NEW_TOKENS = 10  # compare up to 10 decode positions

PYRAMID_CFGS = {
    "pyramidkv":     {"compression_ratio": 0.35, "window_size": 128, "beta": 20},
    "pyramidkv_f50": {"compression_ratio": 0.50, "window_size": 128, "beta": 20},
    "pyramidkv_f40": {"compression_ratio": 0.60, "window_size": 128, "beta": 20},
    "pyramidkv_f35": {"compression_ratio": 0.65, "window_size": 128, "beta": 20},
}


def entropy_nats(logits: torch.Tensor) -> float:
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum().item()


def get_gt_tokens(answers: List[str], tokenizer) -> List[int]:
    """Return token ids for the first non-empty answer (with leading space)."""
    for ans in answers:
        ans = ans.strip()
        if not ans:
            continue
        toks = tokenizer.encode(" " + ans, add_special_tokens=False)
        if toks:
            return toks
        toks = tokenizer.encode(ans, add_special_tokens=False)
        if toks:
            return toks
    return []


def run_generate(model, input_ids, device, pyr_press=None):
    """
    Run model.generate with output_scores=True.
    Returns list of (vocab_size,) float32 CPU tensors, one per generated token.
    Returns None on OOM.
    """
    try:
        with torch.no_grad():
            if pyr_press is not None:
                with pyr_press(model):
                    out = model.generate(
                        input_ids.to(device),
                        attention_mask=torch.ones_like(input_ids).to(device),
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
            else:
                out = model.generate(
                    input_ids.to(device),
                    attention_mask=torch.ones_like(input_ids).to(device),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        scores = [s[0].float().cpu() for s in out.scores]  # list of (vocab,)
        del out
        torch.cuda.empty_cache()
        return scores
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def analyze(results: list) -> dict:
    from collections import defaultdict
    keys = [
        "p_full_gt", "p_pyr_gt", "entropy_full", "entropy_pyr",
        "top1_prob_full", "top1_prob_pyr",
        "pyr_gt_higher", "pyr_entropy_lower", "pyr_top1_higher",
    ]
    acc = defaultdict(list)
    for r in results:
        for k in keys:
            if k in r:
                acc[k].append(r[k])

    summary = {}
    for k, vals in acc.items():
        summary[f"mean_{k}"] = sum(vals) / len(vals)
        summary[f"n_{k}"] = len(vals)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data_dir", default="lb_data_raw/data")
    parser.add_argument("--tasks", default="triviaqa,2wikimqa,hotpotqa")
    parser.add_argument("--pyramid", default="pyramidkv",
                        help="Which pyramid config (pyramidkv/pyramidkv_f50/f40/f35)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="lb_results_base/distill_diagnostic.json")
    args = parser.parse_args()

    if not _KVPRESS_AVAILABLE:
        sys.exit("kvpress not installed; cannot run PyramidKV")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    if not tasks:
        sys.exit(f"No valid tasks in: {args.tasks}")

    pyr_cfg = PYRAMID_CFGS.get(args.pyramid)
    if not pyr_cfg:
        sys.exit(f"Unknown pyramid config: {args.pyramid}")
    pyr_press = _PyramidKVPress(**pyr_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device
    )
    model.eval()

    all_results = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}  (n={args.n})  pyramid={args.pyramid}")
        print(f"Note: comparing at decode steps >=1 (step 0 is prefill, identical for both)")
        print('='*60, flush=True)

        dataset = load_task(task, Path(args.data_dir))
        n_ex = min(len(dataset), args.n)
        template = DATASET2PROMPT[task]
        task_results = []

        for idx in range(n_ex):
            ex = dataset[idx]
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
            )
            answers = ex.get("answers", [])
            gt_tokens = get_gt_tokens(answers, tokenizer)

            if not gt_tokens:
                continue  # no GT tokens to compare against

            full_ids = tokenizer.encode(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
            if full_ids.shape[1] > MAX_SEQ:
                full_ids = full_ids[:, -MAX_SEQ:]

            # Run full model
            scores_full = run_generate(model, full_ids, device, pyr_press=None)
            if scores_full is None:
                print(f"  ex={idx}: OOM (full model)", flush=True)
                continue

            # Run PyramidKV
            scores_pyr = run_generate(model, full_ids, device, pyr_press=pyr_press)
            if scores_pyr is None:
                print(f"  ex={idx}: OOM (PyramidKV)", flush=True)
                continue

            # scores[0] is prefill logit — IDENTICAL for both, skip it.
            # scores[i] (i>=1) is the i-th decode step using the compressed/full KV cache.
            # We compare:
            #   (a) entropy and top-1 at every decode step (always valid)
            #   (b) p(GT_token) at steps where GT has a token at that position
            #
            # GT comparison: scores[step] predicts what comes after the first `step`
            # generated tokens. Both models generate the SAME first token (step 0 logits
            # are identical), so at step 1 both condition on the same prefix.
            # At step 1 we compare p(gt_tokens[1]); more steps may have diverged.
            n_compare = min(len(scores_full), len(scores_pyr))
            ex_recs = []
            for step in range(1, n_compare):
                lf = scores_full[step]   # (vocab,)
                lp = scores_pyr[step]

                sf = F.softmax(lf, dim=-1)
                sp = F.softmax(lp, dim=-1)

                ent_full = entropy_nats(lf)
                ent_pyr  = entropy_nats(lp)

                rec = {
                    "step":              step,
                    "entropy_full":      ent_full,
                    "entropy_pyr":       ent_pyr,
                    "top1_prob_full":    sf.max().item(),
                    "top1_prob_pyr":     sp.max().item(),
                    "pyr_entropy_lower": float(ent_pyr < ent_full),
                    "pyr_top1_higher":   float(sp.max().item() > sf.max().item()),
                }

                # GT comparison: step positions line up if step < len(gt_tokens)
                gt_tok = gt_tokens[step] if step < len(gt_tokens) else None
                if gt_tok is not None:
                    p_full_gt = sf[gt_tok].item()
                    p_pyr_gt  = sp[gt_tok].item()
                    rec["p_full_gt"]     = p_full_gt
                    rec["p_pyr_gt"]      = p_pyr_gt
                    rec["pyr_gt_higher"] = float(p_pyr_gt > p_full_gt)
                    rec["log_ratio_gt"]  = (math.log(p_pyr_gt / p_full_gt)
                                            if p_pyr_gt > 0 and p_full_gt > 0 else None)

                ex_recs.append(rec)

            if ex_recs and idx % 10 == 0:
                r = ex_recs[0]  # first decode step
                gt_str = (f"p_full_gt={r['p_full_gt']:.4f} p_pyr_gt={r['p_pyr_gt']:.4f}"
                          if "p_full_gt" in r else "no GT token at step 1")
                print(
                    f"  ex={idx:3d} step1 | ent_full={r['entropy_full']:.3f} "
                    f"ent_pyr={r['entropy_pyr']:.3f} | {gt_str}",
                    flush=True,
                )

            task_results.extend(ex_recs)

        summary = analyze(task_results)
        all_results[task] = {"per_step": task_results, "summary": summary}

        print(f"\n--- {task} summary (decode steps>=1, n_points={len(task_results)}) ---")
        print(f"  Entropy:    full={summary.get('mean_entropy_full',0):.3f}  "
              f"pyr={summary.get('mean_entropy_pyr',0):.3f}  "
              f"pyr lower: {summary.get('mean_pyr_entropy_lower',0)*100:.1f}%  "
              f"(n={summary.get('n_entropy_full',0)})")
        print(f"  Top-1 prob: full={summary.get('mean_top1_prob_full',0):.4f}  "
              f"pyr={summary.get('mean_top1_prob_pyr',0):.4f}  "
              f"pyr higher: {summary.get('mean_pyr_top1_higher',0)*100:.1f}%")
        if "mean_p_full_gt" in summary:
            print(f"  p(GT tok):  full={summary['mean_p_full_gt']:.4f}  "
                  f"pyr={summary['mean_p_pyr_gt']:.4f}  "
                  f"pyr higher: {summary['mean_pyr_gt_higher']*100:.1f}%  "
                  f"(n={summary.get('n_p_full_gt',0)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {task: {"summary": v["summary"], "n_points": len(v["per_step"]),
                    "per_step": v["per_step"]}
             for task, v in all_results.items()},
            f, indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

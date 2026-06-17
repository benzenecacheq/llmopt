"""
rank_analysis.py
----------------
Comprehensive per-token rank and probability analysis to understand why
PyramidKV achieves good GT scores despite high KL divergence.

For each token position in y*, records:
  - rank of y*[t] in the compressed model's distribution
  - p_comp(y*[t]) and p_full(y*[t])
  - per-token KL at that position

Aggregates by:
  - Overall (all positions)
  - Position quartile (Q1=first 25%, Q2, Q3, Q4=last 25%)

This tests two hypotheses:
  H1: PyramidKV preserves correct argmax even at high KL
      (rank-1 rate similar to good methods despite 8-10x worse KL)
  H2: Early tokens are more faithful; KL grows with position
      (KL in Q1 << KL in Q4; rank degrades across quartiles)

Usage:
    python rank_analysis.py \
        --tasks all \
        --n 100 \
        --methods naive_65pct,chunk_word128_t20,snapkv_select,snapkv,pyramidkv \
        --ystar_cache lb_results_base/ystar_cache_v3.pt \
        --output lb_results_base/rank_analysis.json \
        --log lb_results_base/rank_analysis.log
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kl_faith_eval import (
    DATASET2PROMPT, DATASET2MAXLEN, ENGLISH_TASKS,
    TASK_MAX_SEQ_COMP, TASK_MAX_SEQ_FULL,
    DEFAULT_MAX_SEQ_COMP, DEFAULT_MAX_SEQ_FULL,
    YSTAR_STOP_TOKENS,
)
from kl_faith_eval_ystar import (
    _KVPRESS_METHODS, generate_ystar, get_comp_log_probs,
    make_comp_ids, load_ystar_cache, save_ystar_cache,
)

ALL_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]


def quartile_label(pos, total):
    """Return Q1/Q2/Q3/Q4 bucket for position pos (0-indexed) in sequence of length total."""
    if total <= 1:
        return "Q1"
    frac = pos / (total - 1)
    if frac < 0.25:
        return "Q1"
    elif frac < 0.50:
        return "Q2"
    elif frac < 0.75:
        return "Q3"
    else:
        return "Q4"


def analyse(log_p_full, log_p_comp, ystar_tokens):
    """Per-token stats with position quartile labels."""
    n = log_p_full.shape[0]
    records = []
    for t in range(n):
        tok = ystar_tokens[t].item()
        # rank: argsort descending, find where tok lands (1-indexed)
        sorted_ids = log_p_comp[t].argsort(descending=True)
        pos = (sorted_ids == tok).nonzero(as_tuple=True)[0]
        rank = pos.item() + 1 if pos.numel() > 0 else -1

        p_f = log_p_full[t, tok].exp().item()
        p_c = log_p_comp[t, tok].exp().item()

        p_full_t = log_p_full[t].exp()
        kl_t = (p_full_t * (log_p_full[t] - log_p_comp[t])).sum().item()

        records.append(dict(
            pos=t,
            first=(t == 0),
            quartile=quartile_label(t, n),
            rank=rank,
            p_full=p_f,
            p_comp=p_c,
            kl=kl_t,
        ))
    return records


def aggregate(records):
    """Compute summary stats from a list of per-token record dicts."""
    if not records:
        return {}
    n = len(records)
    ranks  = [r["rank"]  for r in records]
    p_f    = [r["p_full"] for r in records]
    p_c    = [r["p_comp"] for r in records]
    kls    = [r["kl"]    for r in records]
    return dict(
        n_tokens   = n,
        rank1_pct  = 100.0 * sum(1 for r in ranks if r == 1) / n,
        rank5_pct  = 100.0 * sum(1 for r in ranks if r <= 5) / n,
        mean_rank  = sum(ranks) / n,
        mean_p_full = sum(p_f) / n,
        mean_p_comp = sum(p_c) / n,
        p_ratio    = (sum(p_c) / n) / (sum(p_f) / n) if sum(p_f) > 0 else 0,
        mean_kl    = sum(kls) / n,
    )


def print_stats(label, overall, by_quartile, first_token):
    o = overall
    print(f"\n  {label}  (n_tokens={o['n_tokens']})")
    print(f"    rank-1 (argmax match):    {o['rank1_pct']:5.1f}%")
    print(f"    rank ≤ 5:                 {o['rank5_pct']:5.1f}%")
    print(f"    mean rank:                {o['mean_rank']:8.1f}")
    print(f"    mean p_full(y*[t]):       {o['mean_p_full']:.4f}")
    print(f"    mean p_comp(y*[t]):       {o['mean_p_comp']:.4f}")
    print(f"    p_comp/p_full:            {o['p_ratio']:.4f}")
    print(f"    mean KL (nats):           {o['mean_kl']:.4f}")
    if first_token:
        ft = first_token
        print(f"    --- first token only (n={ft['n_tokens']}) ---")
        print(f"    rank-1:                   {ft['rank1_pct']:5.1f}%")
        print(f"    rank ≤ 5:                 {ft['rank5_pct']:5.1f}%")
        print(f"    mean p_comp(y*[0]):       {ft['mean_p_comp']:.4f}")
        print(f"    p_comp/p_full:            {ft['p_ratio']:.4f}")
        print(f"    mean KL (nats):           {ft['mean_kl']:.4f}")
    if by_quartile:
        print(f"    KL by quartile:  ", end="")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            qs = by_quartile.get(q, {})
            print(f"  {q}={qs.get('mean_kl', float('nan')):.3f}", end="")
        print()
        print(f"    rank-1 by quartile:", end="")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            qs = by_quartile.get(q, {})
            print(f"  {q}={qs.get('rank1_pct', float('nan')):.1f}%", end="")
        print()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tasks",       default="all")
    p.add_argument("--n",           type=int, default=25)
    p.add_argument("--methods",     default="naive_65pct,chunk_word128_t20,snapkv_select,snapkv,pyramidkv")
    p.add_argument("--ystar_cache", default="lb_results_base/ystar_cache_v3.pt")
    p.add_argument("--output",      default="lb_results_base/rank_analysis.json")
    p.add_argument("--log",         default="lb_results_base/rank_analysis.log")
    p.add_argument("--data_dir",    default="lb_data_raw/data")
    args = p.parse_args()

    import time
    t_start = time.time()

    log_fh = open(args.log, "a")
    def log(msg):
        print(msg, flush=True)
        print(msg, file=log_fh, flush=True)

    tasks   = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    methods = args.methods.split(",")
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"=== rank_analysis.py started ===")
    log(f"  model:   {args.model}")
    log(f"  tasks:   {tasks}")
    log(f"  methods: {methods}")
    log(f"  n:       {args.n}")

    log(f"\nLoading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    ystar_cache = load_ystar_cache(Path(args.ystar_cache))
    cache_dirty = False

    # results[task][method] = {"overall": {...}, "by_quartile": {"Q1": {...}, ...}}
    results = {}

    # Load existing output if present (resume support)
    out_path = Path(args.output)
    if out_path.exists():
        results = json.load(open(out_path))
        log(f"  Resuming: {sum(len(v) for v in results.values())} task-method pairs already done")

    total_tasks = len(tasks)
    for ti, task in enumerate(tasks):
        prompt_fmt = DATASET2PROMPT[task]
        max_new    = DATASET2MAXLEN[task]
        max_seq_c  = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_f  = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        stop_ids   = YSTAR_STOP_TOKENS.get(task)

        # Check which methods still need to run for this task
        todo_methods = [m for m in methods
                        if task not in results or m not in results[task]]
        if not todo_methods:
            log(f"\n[{ti+1}/{total_tasks}] {task}: all methods done, skipping")
            continue

        log(f"\n[{ti+1}/{total_tasks}] {task}  (methods: {todo_methods})")

        import json as _json
        rows = []
        for candidate in [
            Path(args.data_dir) / task / "test.jsonl",
            Path(args.data_dir) / f"{task}.jsonl",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    for line in f:
                        rows.append(_json.loads(line))
                        if len(rows) == args.n:
                            break
                break

        if not rows:
            log(f"  WARNING: no data found for {task}, skipping")
            continue

        # per-method accumulator: list of per-token records
        method_records = {m: [] for m in todo_methods}

        for i, row in enumerate(rows):
            ctx    = row.get("context", row.get("input", ""))
            prompt = prompt_fmt.format_map({"input": ctx, **row})
            full_ids = tok(prompt, return_tensors="pt",
                           truncation=True, max_length=max_seq_f).input_ids

            cache_key = f"{task}|{i}"
            if cache_key in ystar_cache:
                entry   = ystar_cache[cache_key]
                ystar   = entry["tokens"]
                log_p_f = entry["log_p_full"]
            else:
                ystar, log_p_f, _ = generate_ystar(
                    model, full_ids, max_new, max_seq_f, device, stop_ids)
                ystar_cache[cache_key] = {"tokens": ystar, "log_p_full": log_p_f}
                cache_dirty = True

            if ystar.shape[0] == 0:
                continue

            for method in todo_methods:
                press    = _KVPRESS_METHODS.get(method)
                max_seq  = max_seq_f if press is not None else max_seq_c
                comp_ids = make_comp_ids(full_ids, method, tok, "", max_seq,
                                         model=model, device=device)
                log_p_c  = get_comp_log_probs(model, comp_ids, ystar,
                                               None, device, press=press)
                if log_p_c is None:
                    continue
                records = analyse(log_p_f, log_p_c, ystar)
                method_records[method].extend(records)

            if (i + 1) % 10 == 0:
                elapsed = (time.time() - t_start) / 3600
                log(f"  ex {i+1}/{len(rows)}  [{elapsed:.1f}h elapsed]")

        # Aggregate and store
        if task not in results:
            results[task] = {}

        for method in todo_methods:
            recs = method_records[method]
            if not recs:
                continue
            overall = aggregate(recs)
            by_quartile = {}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                q_recs = [r for r in recs if r["quartile"] == q]
                if q_recs:
                    by_quartile[q] = aggregate(q_recs)
            first_recs = [r for r in recs if r["first"]]
            first_token = aggregate(first_recs) if first_recs else {}
            results[task][method] = {
                "overall": overall,
                "first_token": first_token,
                "by_quartile": by_quartile,
            }
            print_stats(f"{task}/{method}", overall, by_quartile, first_token)

        # Save after each task
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  Saved → {out_path}")

    if cache_dirty:
        save_ystar_cache(ystar_cache, Path(args.ystar_cache))
        log(f"y* cache updated → {args.ystar_cache}")

    elapsed = (time.time() - t_start) / 3600
    log(f"\n=== rank_analysis.py complete  ({elapsed:.2f}h) ===")
    log_fh.close()


if __name__ == "__main__":
    main()

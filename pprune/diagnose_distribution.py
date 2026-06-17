"""
diagnose_distribution.py
------------------------
Tests whether PyramidKV's high KL with preserved rank-1 comes from a
flatter/reshaped distribution that still keeps the correct token on top.

At each token position in y*, records:
  - rank of y*[t] in p_comp
  - p_comp(y*[t]) and p_full(y*[t])
  - entropy of p_comp and p_full
  - margin: p_comp(rank-1) - p_comp(rank-2)
  - what the rank-1 token actually is (when wrong)

Aggregates separately for:
  - all positions
  - first token only (t=0)
  - positions where rank-1 is correct vs incorrect

Usage:
    python diagnose_distribution.py \
        --tasks trec,lcc,narrativeqa \
        --n 25 \
        --methods naive_65pct,chunk_word128_t20,snapkv,pyramidkv \
        --ystar_cache lb_results_base/ystar_cache_v3.pt \
        --output lb_results_base/diag_distribution.json
"""
import argparse
import json
import os
import math
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kl_faith_eval import (
    DATASET2PROMPT, DATASET2MAXLEN, YSTAR_STOP_TOKENS,
    TASK_MAX_SEQ_COMP, TASK_MAX_SEQ_FULL,
    DEFAULT_MAX_SEQ_COMP, DEFAULT_MAX_SEQ_FULL,
)
from kl_faith_eval_ystar import (
    _KVPRESS_METHODS, generate_ystar, get_comp_log_probs,
    make_comp_ids, load_ystar_cache, save_ystar_cache,
)

ALL_TASKS = [
    "narrativeqa","qasper","multifieldqa_en","hotpotqa","2wikimqa","musique",
    "gov_report","qmsum","multi_news","trec","triviaqa","samsum",
    "passage_count","passage_retrieval_en","lcc","repobench-p",
]


def analyse_position(log_p_full, log_p_comp, tok_id):
    """Full distribution stats at one token position."""
    p_full = log_p_full.exp()
    p_comp = log_p_comp.exp()

    # rank of correct token in compressed distribution
    sorted_comp = p_comp.argsort(descending=True)
    pos = (sorted_comp == tok_id).nonzero(as_tuple=True)[0]
    rank = pos.item() + 1 if pos.numel() > 0 else -1

    # rank-1 token
    rank1_id  = sorted_comp[0].item()
    rank1_prob = p_comp[rank1_id].item()
    rank2_prob = p_comp[sorted_comp[1]].item() if len(sorted_comp) > 1 else 0.0
    margin = rank1_prob - rank2_prob

    p_f = p_full[tok_id].item()
    p_c = p_comp[tok_id].item()

    # entropy (clip for numerical safety)
    def entropy(p):
        p = p.clamp(min=1e-10)
        return -(p * p.log()).sum().item() / math.log(2)   # bits

    ent_full = entropy(p_full)
    ent_comp = entropy(p_comp)

    kl = (p_full * (log_p_full - log_p_comp)).sum().item()

    return dict(
        rank=rank,
        correct=(rank == 1),
        rank1_id=rank1_id,
        rank1_prob=rank1_prob,
        rank2_prob=rank2_prob,
        margin=margin,
        p_full=p_f,
        p_comp=p_c,
        p_ratio=p_c / p_f if p_f > 0 else 0,
        ent_full=ent_full,
        ent_comp=ent_comp,
        kl=kl,
    )


def agg(records, label=""):
    if not records:
        return {}
    n = len(records)
    correct = [r for r in records if r["correct"]]
    wrong   = [r for r in records if not r["correct"]]

    def means(recs, keys):
        return {k: sum(r[k] for r in recs)/len(recs) for k in keys} if recs else {}

    keys = ["rank","margin","p_full","p_comp","p_ratio","ent_full","ent_comp","kl"]
    overall = means(records, keys)
    overall["rank1_pct"] = 100.0 * len(correct) / n
    overall["n"] = n

    result = {"overall": overall}
    if correct:
        c = means(correct, ["margin","p_comp","p_ratio","ent_comp","kl"])
        c["n"] = len(correct)
        result["when_correct"] = c
    if wrong:
        w = means(wrong, ["rank","p_comp","p_ratio","ent_comp","kl","margin"])
        w["n"] = len(wrong)
        result["when_wrong"] = w

    return result


def print_summary(task, method, first_tok, all_tok):
    print(f"\n  {task}/{method}")
    o = first_tok.get("overall", {})
    c = first_tok.get("when_correct", {})
    w = first_tok.get("when_wrong", {})
    print(f"    First token (n={o.get('n','?')}):")
    print(f"      rank-1 %:      {o.get('rank1_pct','?'):5.1f}%")
    print(f"      ent_full:      {o.get('ent_full','?'):6.2f} bits")
    print(f"      ent_comp:      {o.get('ent_comp','?'):6.2f} bits  "
          f"({'flatter' if o.get('ent_comp',0)>o.get('ent_full',0) else 'sharper'})")
    def f4(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
    def f1(v): return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"
    print(f"      margin (r1-r2): {f4(o.get('margin','?'))}  "
          f"when correct: {f4(c.get('margin','?')) if c else 'N/A'}  "
          f"when wrong: {f4(w.get('margin','?')) if w else 'N/A'}")
    print(f"      p_comp(y*[0]): {f4(o.get('p_comp','?'))}  "
          f"(p_ratio={f4(o.get('p_ratio','?'))})")
    print(f"      KL:            {f4(o.get('kl','?'))} nats")
    if c:
        print(f"      when CORRECT: KL={f4(c.get('kl','?'))}  p_comp={f4(c.get('p_comp','?'))}")
    if w:
        print(f"      when WRONG:   KL={f4(w.get('kl','?'))}  p_comp={f4(w.get('p_comp','?'))}  rank={f1(w.get('rank','?'))}")

    ao = all_tok.get("overall", {})
    print(f"    All positions (n={ao.get('n','?')}):")
    print(f"      rank-1 %:      {ao.get('rank1_pct','?'):5.1f}%")
    print(f"      ent_comp:      {ao.get('ent_comp','?'):6.2f} bits  "
          f"(full={ao.get('ent_full','?'):.2f})")
    print(f"      KL:            {ao.get('kl','?'):.3f} nats")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tasks",       default="trec,lcc,narrativeqa")
    p.add_argument("--n",           type=int, default=25)
    p.add_argument("--methods",     default="naive_65pct,chunk_word128_t20,snapkv,pyramidkv")
    p.add_argument("--ystar_cache", default="lb_results_base/ystar_cache_v3.pt")
    p.add_argument("--output",      default="lb_results_base/diag_distribution.json")
    p.add_argument("--data_dir",    default="lb_data_raw/data")
    args = p.parse_args()

    tasks   = args.tasks.split(",")
    methods = args.methods.split(",")
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    ystar_cache = load_ystar_cache(Path(args.ystar_cache))
    cache_dirty = False
    results = {}

    for task in tasks:
        prompt_fmt = DATASET2PROMPT[task]
        max_new    = DATASET2MAXLEN[task]
        max_seq_c  = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_f  = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        stop_ids   = YSTAR_STOP_TOKENS.get(task)

        rows = []
        for cand in [Path(args.data_dir)/task/"test.jsonl",
                     Path(args.data_dir)/f"{task}.jsonl"]:
            if cand.exists():
                with open(cand) as f:
                    for line in f:
                        rows.append(json.loads(line))
                        if len(rows) == args.n: break
                break

        print(f"\n{'='*60}\nTask: {task}  (n={len(rows)})\n{'='*60}")
        results[task] = {}
        method_recs  = {m: {"first": [], "all": []} for m in methods}
        # per-example KL sequences: list of dicts with within-example stats
        method_ex_kl = {m: [] for m in methods}

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

            for method in methods:
                press   = _KVPRESS_METHODS.get(method)
                max_seq = max_seq_f if press is not None else max_seq_c
                comp_ids = make_comp_ids(full_ids, method, tok, "", max_seq,
                                         model=model, device=device)
                log_p_c = get_comp_log_probs(model, comp_ids, ystar,
                                              None, device, press=press)
                if log_p_c is None:
                    continue

                ex_kls = []
                for t in range(ystar.shape[0]):
                    rec = analyse_position(log_p_f[t], log_p_c[t], ystar[t].item())
                    method_recs[method]["all"].append(rec)
                    if t == 0:
                        method_recs[method]["first"].append(rec)
                    ex_kls.append(rec["kl"])

                # within-example KL distribution
                n_tok = len(ex_kls)
                ex_kls_sorted = sorted(ex_kls)
                ex_mean   = sum(ex_kls) / n_tok
                ex_median = ex_kls_sorted[n_tok // 2] if n_tok % 2 == 1 \
                            else (ex_kls_sorted[n_tok//2-1] + ex_kls_sorted[n_tok//2]) / 2
                ex_std    = (sum((x - ex_mean)**2 for x in ex_kls) / n_tok) ** 0.5
                first_correct = method_recs[method]["first"][-1]["correct"] \
                                if method_recs[method]["first"] else None
                method_ex_kl[method].append(dict(
                    n_tokens    = n_tok,
                    mean_kl     = ex_mean,
                    median_kl   = ex_median,
                    std_kl      = ex_std,
                    first_correct = first_correct,
                ))

            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(rows)} done", flush=True)

        for method in methods:
            first_agg = agg(method_recs[method]["first"])
            all_agg   = agg(method_recs[method]["all"])

            # aggregate within-example stats across examples
            ex_stats = method_ex_kl[method]
            def ex_agg(subset):
                if not subset: return {}
                n = len(subset)
                means   = sorted(e["mean_kl"]   for e in subset)
                medians = sorted(e["median_kl"] for e in subset)
                med_of_means   = means[n//2] if n%2==1 else (means[n//2-1]+means[n//2])/2
                med_of_medians = medians[n//2] if n%2==1 else (medians[n//2-1]+medians[n//2])/2
                return dict(
                    n            = n,
                    mean_of_means   = sum(e["mean_kl"]   for e in subset) / n,
                    mean_of_medians = sum(e["median_kl"] for e in subset) / n,
                    median_of_means   = med_of_means,
                    median_of_medians = med_of_medians,
                    mean_n_tokens = sum(e["n_tokens"] for e in subset) / n,
                )

            correct_ex = [e for e in ex_stats if e.get("first_correct")]
            wrong_ex   = [e for e in ex_stats if e.get("first_correct") == False]

            ex_summary = {
                "all":     ex_agg(ex_stats),
                "correct": ex_agg(correct_ex),
                "wrong":   ex_agg(wrong_ex),
            }

            results[task][method] = {
                "first_token":  first_agg,
                "all_positions": all_agg,
                "per_example_kl": ex_summary,
            }

            # print per-example KL summary
            print_summary(task, method, first_agg, all_agg)
            ea = ex_summary["all"]; ec = ex_summary.get("correct",{}); ew = ex_summary.get("wrong",{})
            def f3(v): return f"{v:.3f}" if isinstance(v,(int,float)) else "N/A"
            print(f"    Per-example KL (avg {ea.get('mean_n_tokens','?'):.1f} tokens/ex):")
            print(f"      mean-of-means:    {f3(ea.get('mean_of_means','?'))}  "
                  f"correct={f3(ec.get('mean_of_means','?'))}  wrong={f3(ew.get('mean_of_means','?'))}")
            print(f"      median-of-means:  {f3(ea.get('median_of_means','?'))}  "
                  f"correct={f3(ec.get('median_of_means','?'))}  wrong={f3(ew.get('median_of_means','?'))}")
            print(f"      mean-of-medians:  {f3(ea.get('mean_of_medians','?'))}  "
                  f"correct={f3(ec.get('mean_of_medians','?'))}  wrong={f3(ew.get('mean_of_medians','?'))}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

    if cache_dirty:
        save_ystar_cache(ystar_cache, Path(args.ystar_cache))

    print("Done.")


if __name__ == "__main__":
    main()

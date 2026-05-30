"""
diagnose_pyramid_ranks.py
-------------------------
For each y* token position, record:
  - rank of y*[t] in the compressed model's distribution (1 = correct argmax)
  - probability the compressed model assigns to y*[t]
  - probability the full model assigns to y*[t]
  - per-token KL at that position

Compares naive_65pct vs pyramidkv to understand why KL is high for pyramid
despite good GT scores.  If pyramid has rank-1 most of the time but high KL,
it means the argmax is preserved but the probability mass is distributed very
differently from the full model.

Usage:
    python diagnose_pyramid_ranks.py \
        --tasks hotpotqa,trec \
        --n 50 \
        --ystar_cache lb_results_base/ystar_cache_v3.pt
"""
import argparse
import os
import sys
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

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets"); sys.exit(1)


def load_task_data(task, n, data_dir="lb_data_raw/data"):
    import json
    for candidate in [
        Path(data_dir) / task / "test.jsonl",
        Path(data_dir) / f"{task}.jsonl",
        Path(data_dir) / f"{task}_e.jsonl",
    ]:
        if candidate.exists():
            rows = []
            with open(candidate) as f:
                for line in f:
                    rows.append(json.loads(line))
                    if len(rows) == n:
                        break
            return rows
    raise FileNotFoundError(f"No data file found for task '{task}' in {data_dir}")


def rank_of(token_id: int, log_probs: torch.Tensor) -> int:
    """1-based rank of token_id in log_probs (1 = argmax)."""
    sorted_ids = log_probs.argsort(descending=True)
    pos = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    return pos.item() + 1 if pos.numel() > 0 else -1


def analyse(log_p_full, log_p_comp, ystar_tokens):
    """Return per-token stats dict."""
    n = log_p_full.shape[0]
    ranks, p_full_at_y, p_comp_at_y, kl_per_tok = [], [], [], []
    for t in range(n):
        tok = ystar_tokens[t].item()
        r = rank_of(tok, log_p_comp[t])
        ranks.append(r)
        p_full_at_y.append(log_p_full[t, tok].exp().item())
        p_comp_at_y.append(log_p_comp[t, tok].exp().item())
        p_full_t = log_p_full[t].exp()
        kl_t = (p_full_t * (log_p_full[t] - log_p_comp[t])).sum().item()
        kl_per_tok.append(kl_t)
    return dict(ranks=ranks, p_full=p_full_at_y, p_comp=p_comp_at_y, kl=kl_per_tok)


def print_stats(label, all_stats):
    ranks = [r for s in all_stats for r in s["ranks"]]
    p_full = [p for s in all_stats for p in s["p_full"]]
    p_comp = [p for s in all_stats for p in s["p_comp"]]
    kl     = [k for s in all_stats for k in s["kl"]]
    n = len(ranks)
    rank1_frac = sum(1 for r in ranks if r == 1) / n
    rank5_frac = sum(1 for r in ranks if r <= 5) / n
    mean_rank  = sum(ranks) / n
    mean_p_full = sum(p_full) / n
    mean_p_comp = sum(p_comp) / n
    mean_kl     = sum(kl) / n

    print(f"\n  {label}  (n_tokens={n})")
    print(f"    rank-1 (argmax correct):  {rank1_frac*100:5.1f}%")
    print(f"    rank ≤ 5:                 {rank5_frac*100:5.1f}%")
    print(f"    mean rank:                {mean_rank:7.1f}")
    print(f"    mean p_full(y*[t]):       {mean_p_full:.4f}")
    print(f"    mean p_comp(y*[t]):       {mean_p_comp:.4f}")
    print(f"    p_comp / p_full ratio:    {mean_p_comp/mean_p_full:.4f}")
    print(f"    mean per-token KL (nats): {mean_kl:.4f}")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tasks",       default="hotpotqa,trec")
    p.add_argument("--n",           type=int, default=50)
    p.add_argument("--methods",     default="naive_65pct,pyramidkv")
    p.add_argument("--ystar_cache", default="lb_results_base/ystar_cache_v3.pt")
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

    for task in tasks:
        prompt_fmt  = DATASET2PROMPT[task]
        max_new     = DATASET2MAXLEN[task]
        max_seq_c   = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_f   = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        stop_ids    = YSTAR_STOP_TOKENS.get(task)

        rows = load_task_data(task, args.n, args.data_dir)

        print(f"\n{'='*60}")
        print(f"Task: {task}  (n={len(rows)})")
        print(f"{'='*60}")

        method_stats = {m: [] for m in methods}

        for i, row in enumerate(rows):
            ctx   = row.get("context", row.get("input", ""))
            prompt = prompt_fmt.format_map({"input": ctx, **row})
            full_ids = tok(prompt, return_tensors="pt",
                           truncation=True, max_length=max_seq_f).input_ids

            # y* from cache or generate
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
                press = _KVPRESS_METHODS.get(method)
                max_seq = max_seq_f if press is not None else max_seq_c
                comp_ids = make_comp_ids(full_ids, method, tok, "", max_seq,
                                         model=model, device=device)
                log_p_c = get_comp_log_probs(model, comp_ids, ystar,
                                             None, device, press=press)
                if log_p_c is None:
                    continue
                stats = analyse(log_p_f, log_p_c, ystar)
                method_stats[method].append(stats)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)} examples processed", flush=True)

        for method in methods:
            if method_stats[method]:
                print_stats(method, method_stats[method])

    if cache_dirty:
        save_ystar_cache(ystar_cache, Path(args.ystar_cache))
        print(f"\ny* cache updated → {args.ystar_cache}")

    print("\nDone.")


if __name__ == "__main__":
    main()

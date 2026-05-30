"""
truncate_ystar_cache.py
-----------------------
Post-process the existing ystar_cache.pt by truncating each entry at the first
stop token (\n=198, \n\n=271) for the short-answer tasks.

The old cache was generated without stopping criteria, so y* sequences continue
past the answer into the model regurgitating the next few-shot context block.
This script applies the same stopping logic as the updated generate_ystar() so
the cache is consistent with the new code — without needing to re-run the
(expensive) reference model.

Usage:
    python truncate_ystar_cache.py \
        --input  lb_results_base/ystar_cache.pt \
        --output lb_results_base/ystar_cache_v2.pt
"""
import argparse
import torch
from kl_faith_eval import YSTAR_STOP_TOKENS


def truncate_entry(tokens: torch.Tensor, log_p_full: torch.Tensor,
                   stop_ids: list) -> tuple:
    """Truncate tokens/log_p_full at the first stop token (inclusive)."""
    stop_set = set(stop_ids)
    for i, t in enumerate(tokens.tolist()):
        if t in stop_set:
            # Keep up to and including the stop token so it matches what
            # model.generate returns (the stop token IS included in output).
            cut = i + 1
            return tokens[:cut], log_p_full[:cut]
    return tokens, log_p_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="lb_results_base/ystar_cache.pt")
    p.add_argument("--output", default="lb_results_base/ystar_cache_v2.pt")
    args = p.parse_args()

    cache = torch.load(args.input, map_location="cpu", weights_only=True)
    print(f"Loaded {len(cache)} entries from {args.input}")

    new_cache = {}
    stats = {}
    for key, entry in cache.items():
        task = key.split("|")[0]
        stop_ids = YSTAR_STOP_TOKENS.get(task)

        tokens   = entry["tokens"]
        log_p    = entry["log_p_full"]
        old_len  = tokens.shape[0]

        if stop_ids:
            tokens, log_p = truncate_entry(tokens, log_p, stop_ids)

        new_len = tokens.shape[0]
        new_cache[key] = {"tokens": tokens, "log_p_full": log_p}

        task_stats = stats.setdefault(task, {"old": [], "new": []})
        task_stats["old"].append(old_len)
        task_stats["new"].append(new_len)

    torch.save(new_cache, args.output)
    print(f"Saved {len(new_cache)} entries to {args.output}\n")

    print(f"{'Task':<26}  {'old n_gen':>10}  {'new n_gen':>10}  {'reduction':>10}")
    print("-" * 62)
    for task in sorted(stats):
        s = stats[task]
        old_mean = sum(s["old"]) / len(s["old"])
        new_mean = sum(s["new"]) / len(s["new"])
        pct = 100 * (1 - new_mean / old_mean) if old_mean else 0
        print(f"{task:<26}  {old_mean:>10.1f}  {new_mean:>10.1f}  {pct:>9.0f}%")


if __name__ == "__main__":
    main()

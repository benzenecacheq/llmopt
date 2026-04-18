"""
benchmark_efficiency.py
-----------------------
Measures generation throughput and KV cache size for full-context vs KV-pruned Llama.

KV cache pruning trades prefill overhead for generation savings. The relevant metrics are:
  - KV cache size after prefill (MB) — determines how much VRAM context consumes
  - Generation tokens/sec — benefits from smaller KV cache per attention step
  - Maximum feasible context length — how long a prompt can be before OOM

Usage:
    python benchmark_efficiency.py \\
        --model meta-llama/Llama-3.1-8B \\
        --budget_fraction 0.65 \\
        --context_lens 2048,4096,8192 \\
        --gen_tokens 64 \\
        --warmup 1 \\
        --repeats 3 \\
        --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from llama_pruned import PrunedLlamaConfig, build_pruned_model


def kv_cache_mb(model, context_len: int, dtype: torch.dtype) -> float:
    """Theoretical KV cache size in MB for a given context length."""
    cfg = model.config
    n_layers    = cfg.num_hidden_layers
    kv_heads    = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim    = cfg.hidden_size // cfg.num_attention_heads
    bytes_per_el = 2 if dtype == torch.float16 else 4
    # 2 for K and V
    return 2 * n_layers * kv_heads * context_len * head_dim * bytes_per_el / 1e6


@torch.inference_mode()
def measure_generation(
    model,
    context_len: int,
    gen_tokens: int,
    warmup: int,
    repeats: int,
    vocab_size: int,
    device: str,
) -> dict:
    """Prefill then generate gen_tokens; return peak VRAM (MB) and tokens/sec."""
    input_ids = torch.randint(100, vocab_size - 1, (1, context_len), device=device)

    def run_once():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model.generate(
            input_ids,
            max_new_tokens=gen_tokens,
            do_sample=False,
            pad_token_id=vocab_size - 1,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        tokens_sec = gen_tokens / elapsed
        return peak_mb, tokens_sec

    # Warmup
    for _ in range(warmup):
        run_once()

    # Timed runs
    peak_mbs, tpss = [], []
    for _ in range(repeats):
        peak_mb, tps = run_once()
        peak_mbs.append(peak_mb)
        tpss.append(tps)

    peak_mbs.sort()
    tpss.sort()
    return {
        "peak_vram_mb":  round(peak_mbs[len(peak_mbs) // 2], 1),
        "tokens_per_sec": round(tpss[len(tpss) // 2], 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="meta-llama/Llama-3.1-8B")
    p.add_argument("--budget_fraction", type=float, default=0.65)
    p.add_argument("--context_lens",    default="2048,4096,8192")
    p.add_argument("--gen_tokens",      type=int, default=64)
    p.add_argument("--warmup",          type=int, default=1)
    p.add_argument("--repeats",         type=int, default=3)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--score_mode",      default="additive")
    p.add_argument("--score_alpha",     type=float, default=0.65)
    p.add_argument("--min_decay",       type=float, default=0.7)
    p.add_argument("--q_buffer_size",   type=int,   default=128)
    p.add_argument("--always_keep_first", type=int, default=16)
    p.add_argument("--always_keep_last",  type=int, default=16)
    p.add_argument("--output",          default="benchmark_results.json")
    args = p.parse_args()

    context_lens = [int(x) for x in args.context_lens.split(",")]

    pcfg = PrunedLlamaConfig(
        budget_fraction=args.budget_fraction,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        min_decay=args.min_decay,
        q_buffer_size=args.q_buffer_size,
        always_keep_first=args.always_keep_first,
        always_keep_last=args.always_keep_last,
    )
    model, _ = build_pruned_model(args.model, pcfg, device=args.device, dtype=torch.float16)
    vocab_size = model.config.vocab_size

    print(f"\nModel: {args.model}")
    print(f"Budget fraction: {args.budget_fraction:.0%}  |  "
          f"Gen tokens: {args.gen_tokens}  |  Repeats: {args.repeats}\n")

    # Theoretical KV cache sizes
    print(f"{'Context':>10}  {'KV cache (full)':>18}  {'KV cache (pruned)':>18}  {'Reduction':>10}")
    print("-" * 64)
    for ctx in context_lens:
        full_mb   = kv_cache_mb(model, ctx,    torch.float16)
        pruned_mb = kv_cache_mb(model, int(ctx * args.budget_fraction), torch.float16)
        print(f"{ctx:>10}  {full_mb:>18.0f}  {pruned_mb:>18.0f}  "
              f"{100*(full_mb-pruned_mb)/full_mb:>9.1f}%")

    # Measured generation throughput
    results = []
    print(f"\n{'Context':>10}  {'Method':<8}  {'Peak VRAM (MB)':>16}  {'Tokens/sec':>12}")
    print("-" * 52)

    for ctx in context_lens:
        for method, fraction in [("full", 0.0), ("pruned", args.budget_fraction)]:
            pcfg.budget_fraction = fraction
            try:
                m = measure_generation(
                    model, ctx, args.gen_tokens, args.warmup, args.repeats,
                    vocab_size, args.device,
                )
                results.append({"context_len": ctx, "method": method, **m})
                print(f"{ctx:>10}  {method:<8}  "
                      f"{m['peak_vram_mb']:>16.1f}  {m['tokens_per_sec']:>12.1f}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results.append({"context_len": ctx, "method": method,
                                 "peak_vram_mb": None, "tokens_per_sec": None})
                print(f"{ctx:>10}  {method:<8}  {'OOM':>16}  {'OOM':>12}")

    # Speedup summary
    print(f"\n{'Context':>10}  {'VRAM full':>12}  {'VRAM pruned':>12}  "
          f"{'VRAM saved':>10}  {'Speedup':>10}")
    print("-" * 60)
    by_ctx: dict = {}
    for r in results:
        by_ctx.setdefault(r["context_len"], {})[r["method"]] = r
    for ctx in sorted(by_ctx):
        f  = by_ctx[ctx].get("full",   {})
        pr = by_ctx[ctx].get("pruned", {})
        fv, pv = f.get("peak_vram_mb"), pr.get("peak_vram_mb")
        ft, pt = f.get("tokens_per_sec"), pr.get("tokens_per_sec")
        vram_saved = f"{100*(fv-pv)/fv:.1f}%" if fv and pv else "N/A"
        speedup    = f"{pt/ft:.2f}x"           if ft and pt else "N/A"
        fv_s = f"{fv:.0f}" if fv else "OOM"
        pv_s = f"{pv:.0f}" if pv else "OOM"
        print(f"{ctx:>10}  {fv_s:>12}  {pv_s:>12}  {vram_saved:>10}  {speedup:>10}")

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

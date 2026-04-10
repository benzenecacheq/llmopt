"""
fa_benchmark.py
---------------
Efficiency benchmark comparing full-context, AdditiveScorerPress, and SnapKV
at long context lengths using Flash Attention 2.

Uses synthetic random-token inputs of fixed length — content is irrelevant
for efficiency measurement since the compute graph is identical regardless
of token semantics.

Measures per method and context length:
  - Prefill time (s)        — time to process the input sequence
  - Peak VRAM during prefill (MB)
  - Post-prefill KV cache size (MB)  — what remains in memory during generation
  - Generation throughput (tok/s)    — autoregressive decoding speed

The FA2 / non-FA2 comparison isolates the Flash Attention compatibility
advantage: SnapKV requires materializing attention weights (incompatible with
FA2 without a separate pass), while AdditiveScorerPress uses only K/V tensors
and runs natively under FA2.

Usage
-----
    python fa_benchmark.py \\
        --model meta-llama/Llama-3.1-8B \\
        --context_lens 8192,16384,32768 \\
        --compression_ratio 0.35 \\
        --gen_tokens 64 \\
        --repeats 3 \\
        --output fa_benchmark_results.json

    # Without Flash Attention (baseline comparison):
    python fa_benchmark.py --no_flash_attn ...
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager, nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import SnapKVPress
from additive_scorer_press import AdditiveScorerPress


# ---------------------------------------------------------------------------
# KV cache size measurement
# ---------------------------------------------------------------------------

def kv_cache_mb(past_key_values, dtype: torch.dtype) -> float:
    """Sum the actual allocated KV tensors from a past_key_values object."""
    if past_key_values is None:
        return 0.0
    bytes_per_el = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total = 0
    for layer in past_key_values:
        for tensor in layer:
            if tensor is not None:
                total += tensor.numel()
    return total * bytes_per_el / 1e6


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@torch.inference_mode()
def measure(
    model,
    press,
    context_len: int,
    gen_tokens: int,
    repeats: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (1, context_len), device=device)

    prefill_times, gen_tps, peak_vrams, kv_mbs = [], [], [], []

    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Create a fresh press context each repeat so hooks are clean
        press_ctx = press(model) if press is not None else nullcontext()

        # --- Prefill ---
        t0 = time.perf_counter()
        try:
            with press_ctx:
                out = model(input_ids, use_cache=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return {"error": str(e)[:120]}
        except Exception as e:
            return {"error": str(e)[:120]}
        torch.cuda.synchronize(device)
        prefill_time = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e6
        kv_mb = kv_cache_mb(out.past_key_values, dtype)

        prefill_times.append(prefill_time)
        peak_vrams.append(peak_vram)
        kv_mbs.append(kv_mb)

        # --- Generation (short, to measure decode throughput) ---
        if gen_tokens > 0:
            pkv = out.past_key_values
            last_tok = input_ids[:, -1:]
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            for _ in range(gen_tokens):
                gen_out = model(last_tok, past_key_values=pkv, use_cache=True)
                pkv = gen_out.past_key_values
                last_tok = gen_out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize(device)
            gen_tps.append(gen_tokens / (time.perf_counter() - t1))
            del pkv, last_tok, gen_out

        del out
        torch.cuda.empty_cache()

    def median(lst):
        s = sorted(lst)
        return s[len(s) // 2]

    result = {
        "prefill_time_s":   round(median(prefill_times), 3),
        "peak_vram_mb":     round(median(peak_vrams), 1),
        "kv_cache_mb":      round(median(kv_mbs), 1),
    }
    if gen_tps:
        result["gen_tok_per_s"] = round(median(gen_tps), 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",            default="meta-llama/Llama-3.1-8B")
    p.add_argument("--context_lens",     default="8192,16384,32768")
    p.add_argument("--compression_ratio",type=float, default=0.35)
    p.add_argument("--q_buffer_size",    type=int,   default=128)
    p.add_argument("--gen_tokens",       type=int,   default=64)
    p.add_argument("--repeats",          type=int,   default=3)
    p.add_argument("--attn_impl",        default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="auto: try fa2, fall back to sdpa")
    p.add_argument("--output",           default="fa_benchmark_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    context_lens = [int(x) for x in args.context_lens.split(",")]
    device = "cuda"
    dtype = torch.float16
    # Resolve attention implementation
    if args.attn_impl == "auto":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    else:
        attn_impl = args.attn_impl

    print(f"Model     : {args.model}")
    print(f"Attention : {attn_impl}")
    print(f"Contexts  : {context_lens}")
    print(f"Keep      : {100*(1-args.compression_ratio):.0f}%  (compression_ratio={args.compression_ratio})")
    print(f"Gen tokens: {args.gen_tokens}")

    print(f"\nLoading model ({attn_impl}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()

    presses = {
        "full":     None,
        "additive": AdditiveScorerPress(
            compression_ratio=args.compression_ratio,
            q_buffer_size=args.q_buffer_size,
        ),
        "snapkv":   SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.q_buffer_size,
        ),
    }

    all_results = {}
    header = (f"\n{'Context':>10}  {'Method':<10}  {'Prefill(s)':>10}  "
              f"{'PeakVRAM(MB)':>13}  {'KV(MB)':>8}  {'Gen(tok/s)':>11}")
    print(header)
    print("-" * 72)

    for ctx in context_lens:
        all_results[ctx] = {}
        for name, press in presses.items():
            r = measure(model, press, ctx, args.gen_tokens, args.repeats,
                        device, dtype)
            all_results[ctx][name] = r
            if "error" in r:
                print(f"{ctx:>10}  {name:<10}  ERROR: {r['error'][:50]}")
            else:
                gen_s = f"{r.get('gen_tok_per_s', 0):>11.1f}" if 'gen_tok_per_s' in r else f"{'N/A':>11}"
                print(f"{ctx:>10}  {name:<10}  {r['prefill_time_s']:>10.3f}  "
                      f"{r['peak_vram_mb']:>13.1f}  {r['kv_cache_mb']:>8.1f}  {gen_s}")
            torch.cuda.empty_cache()

    # Summary: additive vs full speedup and VRAM reduction
    print(f"\n{'Context':>10}  {'Method':<10}  {'Prefill vs full':>16}  "
          f"{'KV vs full':>12}  {'Gen speedup':>12}")
    print("-" * 64)
    for ctx in context_lens:
        full = all_results[ctx].get("full", {})
        for name in ("additive", "snapkv"):
            r = all_results[ctx].get(name, {})
            if "error" in r or "error" in full:
                continue
            pt_ratio = f"{r['prefill_time_s']/full['prefill_time_s']:.2f}x"
            kv_ratio = f"{100*(1 - r['kv_cache_mb']/full['kv_cache_mb']):.1f}% saved"
            gen_spd  = (f"{r['gen_tok_per_s']/full['gen_tok_per_s']:.2f}x"
                        if 'gen_tok_per_s' in r and 'gen_tok_per_s' in full else "N/A")
            print(f"{ctx:>10}  {name:<10}  {pt_ratio:>16}  {kv_ratio:>12}  {gen_spd:>12}")

    out = {
        "results": all_results,
        "args": {**vars(args), "attn_impl": attn_impl},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

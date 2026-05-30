"""
precompute_token_scores.py
--------------------------
Compute a static per-token semantic salience lookup table from a model's
Q/K weight geometry.  No input sequence required — scores reflect how much
each vocabulary token is attended to when the *entire vocabulary* is used as
the query set.  The result is a position-independent importance prior.

Algorithm (per selected layer):
  1. Apply the layer's input_layernorm to every token embedding (no RoPE).
  2. Project through q_proj / k_proj  →  pre-RoPE Q and K for all V tokens.
  3. Expand K for GQA (repeat_interleave).
  4. For each head, compute the V×V attention matrix in query chunks:
       logits  = Q_chunk @ K_all^T / sqrt(head_dim)
       mask diagonal (no token attends to itself)
       attn    = softmax(logits, dim=-1)          # over key dimension
       accumulate column sums  →  (V,) per head
  5. Divide by V  →  mean attention received per token.
  6. Aggregate across heads (--head_agg max|mean)  →  (V,) per layer.
Average across layers (--layer_agg mean|max)  →  final (V,) lookup table.

At inference time:  scores = table[token_ids]  — pure indexing, zero compute.

Usage:
    python precompute_token_scores.py \\
        --model meta-llama/Llama-3.1-8B \\
        --output lb_results_base/token_salience_llama31.pt \\
        --layers all                 # or e.g. 0,8,16,23,31
        --head_agg max               # max | mean over heads
        --layer_agg mean             # mean | max over layers
        --query_chunk 512            # queries per chunk (tune for VRAM)
        --device cuda

Supports Llama and Mistral architectures (auto-detected via model config).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Per-layer score computation
# ---------------------------------------------------------------------------

def _layer_scores(
    embed_weight: torch.Tensor,    # (V, hidden_dim) — shared across layers
    input_norm,                    # RMSNorm for this layer (module)
    q_weight: torch.Tensor,        # (n_heads*head_dim, hidden_dim)
    k_weight: torch.Tensor,        # (n_kv_heads*head_dim, hidden_dim)
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_agg: str,
    query_chunk: int,
    device: str,
) -> torch.Tensor:
    """Return (V,) salience scores for one transformer layer."""
    V        = embed_weight.shape[0]
    q_per_kv = num_heads // num_kv_heads
    scale    = 1.0 / math.sqrt(head_dim)

    with torch.no_grad():
        normed = input_norm(embed_weight)                          # (V, hidden_dim)
        Q_all  = (normed @ q_weight.T).reshape(V, num_heads,    head_dim)
        K_all  = (normed @ k_weight.T).reshape(V, num_kv_heads, head_dim)
        # Expand K for GQA: each KV-head serves q_per_kv Q-heads
        K_all  = K_all.repeat_interleave(q_per_kv, dim=1)         # (V, num_heads, head_dim)

    # Head scores accumulator: (num_heads, V)
    head_accum = torch.zeros(num_heads, V, device=device, dtype=torch.float32)
    n_chunks   = math.ceil(V / query_chunk)

    # Outer loop over heads so k_h is converted from fp16 once per head,
    # not once per (head × chunk).  Each k_h is (V, head_dim) fp32 = ~65 MB
    # for Llama; recomputing it 251× per head per layer was the bottleneck.
    for h in range(num_heads):
        k_h = K_all[:, h, :].float()                    # (V, head_dim) — once per head
        q_h_all = Q_all[:, h, :].float()                # (V, head_dim) — once per head

        for c in range(n_chunks):
            q_start  = c * query_chunk
            q_end    = min(q_start + query_chunk, V)
            chunk_sz = q_end - q_start

            logits = (q_h_all[q_start:q_end] @ k_h.T) * scale   # (chunk_sz, V)

            # Mask self-attention (query token attending to itself)
            i_idx = torch.arange(chunk_sz, device=device)
            logits[i_idx, q_start + i_idx] = float("-inf")

            attn = F.softmax(logits, dim=-1)             # (chunk_sz, V)
            head_accum[h] += attn.sum(dim=0)             # accumulate column sums

        if (h + 1) % max(1, num_heads // 4) == 0 or h == num_heads - 1:
            print(f"      head {h+1}/{num_heads}", flush=True)

    head_accum /= V                                      # mean attention per token

    if head_agg == "max":
        layer_scores = head_accum.max(dim=0).values      # (V,)
    else:
        layer_scores = head_accum.mean(dim=0)            # (V,)

    return layer_scores.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute static per-token salience scores from model Q/K geometry."
    )
    p.add_argument("--model",       required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--output",      required=True,
                   help="Output .pt file (saves scores tensor + metadata)")
    p.add_argument("--layers",      default="all",
                   help="Comma-separated layer indices or 'all' (default: all)")
    p.add_argument("--head_agg",    default="max", choices=["max", "mean"],
                   help="How to aggregate scores across attention heads (default: max)")
    p.add_argument("--layer_agg",   default="mean", choices=["mean", "max"],
                   help="How to aggregate scores across layers (default: mean)")
    p.add_argument("--query_chunk", type=int, default=512,
                   help="Number of query tokens processed per chunk (tune for VRAM, default: 512)")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device,
    )
    model.eval()

    cfg          = model.config
    num_layers   = cfg.num_hidden_layers
    num_heads    = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim     = cfg.hidden_size // num_heads
    vocab_size   = cfg.vocab_size

    print(f"Model type : {cfg.model_type}")
    print(f"Vocab size : {vocab_size}")
    print(f"Layers     : {num_layers}")
    print(f"Heads      : {num_heads}  (KV heads: {num_kv_heads},  head_dim: {head_dim})")
    print(f"Head agg   : {args.head_agg}")
    print(f"Layer agg  : {args.layer_agg}")
    print(f"Chunk size : {args.query_chunk}")

    if args.layers.strip().lower() == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    print(f"Scoring {len(layer_indices)} layer(s): {layer_indices}\n")

    # Embedding matrix — shared across all layers
    embed_weight = model.model.embed_tokens.weight.detach().to(device)   # (V, hidden_dim)
    print(f"Embedding matrix: {embed_weight.shape}  {embed_weight.dtype}")

    all_layer_scores: list = []

    for l_idx in layer_indices:
        layer = model.model.layers[l_idx]
        attn  = layer.self_attn
        norm  = layer.input_layernorm

        q_w = attn.q_proj.weight.detach().to(device)   # (n_heads*head_dim, hidden_dim)
        k_w = attn.k_proj.weight.detach().to(device)   # (n_kv_heads*head_dim, hidden_dim)

        n_chunks = math.ceil(vocab_size / args.query_chunk)
        print(f"Layer {l_idx:>2d}  ({n_chunks} query chunks × {num_heads} heads) ...")

        scores = _layer_scores(
            embed_weight=embed_weight,
            input_norm=norm,
            q_weight=q_w,
            k_weight=k_w,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_agg=args.head_agg,
            query_chunk=args.query_chunk,
            device=device,
        )
        all_layer_scores.append(scores)
        print(f"  scores: min={scores.min():.5f}  max={scores.max():.5f}"
              f"  mean={scores.mean():.5f}\n")

        del q_w, k_w
        torch.cuda.empty_cache()

    # Aggregate across layers
    stacked = torch.stack(all_layer_scores, dim=0)    # (n_layers, V)
    if args.layer_agg == "max":
        final_scores = stacked.max(dim=0).values
    else:
        final_scores = stacked.mean(dim=0)

    print(f"Final scores: {final_scores.shape}  "
          f"min={final_scores.min():.5f}  max={final_scores.max():.5f}  "
          f"mean={final_scores.mean():.5f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores":      final_scores.float(),   # (V,) float32
            "model":       args.model,
            "vocab_size":  vocab_size,
            "num_layers":  num_layers,
            "layers":      layer_indices,
            "head_agg":    args.head_agg,
            "layer_agg":   args.layer_agg,
        },
        out_path,
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

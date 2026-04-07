"""
llama_pruned.py
---------------
Drop-in replacement for HuggingFace LlamaAttention that injects
per-head KV pruning via PerHeadFilter.

Supports Llama 2 and Llama 3 (transformers >= 4.36 Cache API detected
automatically at import time).

The key change to the forward pass:

  Standard Llama:
    1. Project Q, K, V for all tokens               O(n · d)
    2. Compute attention: softmax(QKᵀ/√d) · V       O(n² · d)

  Pruned Llama (prefill of long context):
    1. Stream tokens through PerHeadFilter, building
       per-head importance scores                    O(n · m · d), m << n
    2. For each head, gather only retained K, V      O(n_h · d)
    3. Compute attention against retained KV         O(n_q · n_h · d)

Usage:
    from llama_pruned import build_pruned_model, PrunedLlamaConfig

    pcfg = PrunedLlamaConfig(
        total_budget=512,
        q_buffer_size=64,
        budget_strategy="entropy",   # or "equal" or "both"
        decay_fn="exponential",      # or "linear"
        decay_rate=0.002,
        always_keep_last=16,
    )
    model, tokenizer = build_pruned_model(
        model_name="meta-llama/Llama-2-7b-hf",   # or Meta-Llama-3-8B, etc.
        pruned_cfg=pcfg,
        device="cuda",
        dtype=torch.float16,
    )

    # Standard HuggingFace generate API works unchanged
    inputs = tokenizer("Your long prompt here", return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=200)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

# transformers >= 4.36 (required for Llama 3) uses a Cache object for
# past_key_value instead of a raw (key, value) tuple.
try:
    from transformers.cache_utils import Cache as HFCache
    _HF_CACHE_API = True
except ImportError:
    HFCache = None
    _HF_CACHE_API = False

# transformers 5.x: LlamaDecoderLayer unpacks self_attn output as
# (hidden_states, present_key_value) — 2 values, no attn_weights slot.
# transformers 4.x: (hidden_states, attn_weights, present_key_value) — 3 values.
import transformers as _transformers
_TRANSFORMERS_V5 = int(_transformers.__version__.split(".")[0]) >= 5

from head_filter import HeadFilterConfig, PerHeadFilter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PrunedLlamaConfig:
    total_budget: int = 512
    q_buffer_size: int = 64
    budget_strategy: str = "entropy"    # "equal" | "entropy" | "both"
    decay_fn: str = "linear"            # "linear" | "exponential"
    decay_rate: float = 0.0             # if 0.0, derived from min_decay and actual T
    min_decay: float = 0.7              # decay value applied to the very first token;
                                        # rate is auto-computed as (1-min_decay)/T  (linear)
                                        # or -ln(min_decay)/T  (exponential).
                                        # Ignored when decay_rate is set explicitly (> 0).
    always_keep_last: int = 16          # unconditionally keep this many tail tokens
    always_keep_first: int = 16         # unconditionally keep this many head tokens
    score_mode: str = "additive"        # "kq_vn_decay" | "kq_only" | "vn_only" | "vn_decay"
                                        # | "additive"  (alpha*kq + (1-alpha)*vn*decay)
                                        # | "streaming" (first sink_size + recent window)
                                        # | "kq_post_rope" (kq_only but with rotated K/Q)
                                        # | "snapkv"    (pooled attention weights, post-RoPE)
    score_alpha: float = 0.65           # weight on kq in "additive" mode
    sink_size: int = 4                  # attention sink tokens kept in "streaming" mode
    snapkv_window: int = 32             # observation window size for snapkv scoring
    budget_fraction: float = 0.0        # if > 0, overrides total_budget with int(T * fraction)
    # If True, run the filter during prefill only; generation uses standard KV cache
    filter_prefill_only: bool = True


# ---------------------------------------------------------------------------
# Pruned attention module
# ---------------------------------------------------------------------------

class PrunedLlamaAttention(nn.Module):
    """
    Replaces LlamaAttention. Shares all weights with the original module —
    no retraining required.

    Compatible with Llama 2 (tuple KV cache) and Llama 3 (Cache object).
    """

    def __init__(
        self,
        original: LlamaAttention,
        pruned_cfg: PrunedLlamaConfig,
        layer_idx: int,
        device: torch.device,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.pcfg = pruned_cfg
        self.device = device

        # Copy config scalars.
        # transformers 5.x removed direct attributes on LlamaAttention;
        # they live on the module's config instead.
        cfg = original.config
        self.hidden_size          = cfg.hidden_size
        self.num_heads            = cfg.num_attention_heads
        self.head_dim             = getattr(cfg, "head_dim",
                                        cfg.hidden_size // cfg.num_attention_heads)
        self.num_key_value_heads  = getattr(cfg, "num_key_value_heads",
                                        cfg.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = cfg.max_position_embeddings

        # Share weight tensors (no copy — same storage)
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        # rotary_emb moved to the decoder layer in transformers 5.x;
        # use getattr so we don't crash on new versions
        self.rotary_emb = getattr(original, "rotary_emb", None)

        # Per-layer filter — will be set by the model wrapper
        self.head_filter: Optional[PerHeadFilter] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """(B, T, d) -> (B, num_heads, T, head_dim)"""
        B, T, _ = x.shape
        return x.view(B, T, num_heads, self.head_dim).transpose(1, 2)

    def _aggregate_head_scores(self, all_scores: torch.Tensor) -> torch.Tensor:
        """
        Combine per-head importance scores into a single global score per token.

        all_scores : (num_heads, T)  — each row is a head, each column a token.
                     Values are in [0, 1] (normalised within each head).

        Returns : (T,) global score tensor.

        Strategies to try:
          "max"  — a token is retained if *any* head finds it important.
                   Conservative: nothing is discarded unless all heads agree it
                   is unimportant.  Good default.
          "mean" — average importance across heads.  Tokens that are broadly
                   useful rank higher; tokens needed by only one head may be
                   under-valued.
          "sum"  — equivalent to mean (up to scaling) but amplifies tokens
                   attended by many heads simultaneously.
        """
        # Change this line to experiment with other strategies:
        return all_scores.max(dim=0).values

    def _run_filter_prefill(
        self,
        q_raw: torch.Tensor,          # pre-RoPE query states  (B, num_heads,    T, head_dim)
        k_raw: torch.Tensor,          # pre-RoPE key states    (B, num_kv_heads, T, head_dim)
        key_states: torch.Tensor,     # post-RoPE key states   (B, num_kv_heads, T, head_dim)
        value_states: torch.Tensor,   # post-RoPE value states (B, num_kv_heads, T, head_dim)
        budget: int = 0,              # effective budget (already resolved from fraction/total)
        q_post: Optional[torch.Tensor] = None,  # post-RoPE query states (B, num_heads, T, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Score every token globally (aggregating across all Q-heads), keep the
        top `total_budget` positions, and gather the SAME retained set for every
        KV head.

        Scoring uses pre-RoPE K and Q so that the KQ dot products reflect
        position-independent semantic similarity.  The actual K/V that get
        gathered and returned are the post-RoPE tensors needed for attention.

        Returns (gathered_k, gathered_v, retained_positions) where
        retained_positions is a LongTensor of the original sequence positions
        that were kept (needed to build the correct causal mask in forward).
        On failure / fallback returns (key_states, value_states, None).

        Global scoring:
          1. Per Q-head importance score: kq_alignment × v_norm × distance_decay
          2. Aggregate per token: max over all Q-heads  (each score in [0,1])
          3. Always-keep tail: protect the last `always_keep_last` tokens
          4. From the remaining early tokens select the top
             (total_budget - always_keep) by global score
          5. Gather the unified kept set from every KV head

        Batch size > 1 falls back to standard attention (masks would differ).
        """
        B, _kv_h, T, D = key_states.shape
        if B != 1:
            return key_states, value_states, None

        num_q    = self.num_heads
        q_per_kv = self.num_key_value_groups
        cfg      = self.head_filter.cfg if self.head_filter is not None else None
        if cfg is None:
            return key_states, value_states, None

        # StreamingLLM: keep first sink_size tokens + a recent window. No scoring needed.
        if self.pcfg.score_mode == "streaming":
            sink        = min(self.pcfg.sink_size, T)
            window      = max(0, budget - sink)
            win_start   = max(sink, T - window)
            keep_set    = set(range(sink)) | set(range(win_start, T))
            retained    = torch.tensor(sorted(keep_set), dtype=torch.long,
                                       device=self.device)
            gathered_k  = key_states[  :, :, retained, :]
            gathered_v  = value_states[:, :, retained, :]
            return gathered_k, gathered_v, retained

        # --- Fully batched scoring ---
        #
        # For pre-RoPE modes: use q_raw / k_raw so that KQ dot products reflect
        # position-independent semantic similarity.  Post-RoPE vectors penalise
        # early tokens simply because their rotational encoding is far from the
        # Q-buffer's encoding.
        #
        # For kq_post_rope: use rotated Q and K to measure the same similarity
        # the actual attention kernel will see, as an ablation of the pre-RoPE choice.

        q_buf_sz = min(cfg.q_buffer_size, T)

        mode = self.pcfg.score_mode

        if mode == "kq_post_rope":
            # Post-RoPE ablation: score using rotated Q and K (same vectors the
            # attention kernel uses), to isolate the effect of pre-RoPE scoring.
            _q = q_post if q_post is not None else q_raw
            q_buf_p  = _q[0, :, -q_buf_sz:, :]                           # (num_q, q_buf_sz, D)
            key_exp_p = key_states[0].repeat_interleave(q_per_kv, dim=0) # (num_q, T, D)
            dots_p   = torch.bmm(q_buf_p.float(),
                                 key_exp_p.float().transpose(1, 2)) / math.sqrt(D)
            kq_p     = dots_p.max(dim=1).values                          # (num_q, T)
            kq_min   = kq_p.min(dim=1, keepdim=True).values
            kq_max   = kq_p.max(dim=1, keepdim=True).values
            head_scores = (kq_p - kq_min) / (kq_max - kq_min + cfg.score_eps)

        elif mode == "snapkv":
            # SnapKV [Li et al., 2024]: pool causally-masked attention weights from
            # the last snapkv_window query positions over all key positions.
            # Uses post-RoPE Q and K so scores match the actual attention distribution.
            _q     = q_post if q_post is not None else q_raw
            window = min(self.pcfg.snapkv_window, T)
            dev    = key_states.device

            q_win    = _q[0, :, -window:, :]                              # (num_q, window, D)
            key_ep   = key_states[0].repeat_interleave(q_per_kv, dim=0)  # (num_q, T, D)
            logits   = torch.bmm(q_win.float(),
                                 key_ep.float().transpose(1, 2)) / math.sqrt(D)
                                                                           # (num_q, window, T)

            # Causal mask: query at seq-position (T-window+j) may only attend to
            # keys at positions <= T-window+j.
            q_pos = torch.arange(T - window, T, device=dev).unsqueeze(1)  # (window, 1)
            k_pos = torch.arange(T,          device=dev).unsqueeze(0)     # (1, T)
            causal = torch.where(k_pos <= q_pos,
                                 torch.zeros((), device=dev),
                                 torch.full((), float('-inf'), device=dev))
            logits = logits + causal.unsqueeze(0)                          # broadcast over heads

            attn_w    = F.softmax(logits, dim=-1)                          # (num_q, window, T)
            head_scores = attn_w.max(dim=1).values                         # (num_q, T)

        else:
            # Pre-RoPE path: compute KQ using unrotated Q and K so that dot
            # products reflect content similarity, not positional distance.
            q_buf   = q_raw[0, :, -q_buf_sz:, :]              # (num_q, q_buf_sz, D)
            key_exp = k_raw[0].repeat_interleave(q_per_kv, dim=0)  # (num_q, T, D)
            dots    = torch.bmm(q_buf.float(),
                                key_exp.float().transpose(1, 2)) / math.sqrt(D)
            kq      = dots.max(dim=1).values                   # (num_q, T)

            # Normalize KQ per head to [0,1]
            kq_min = kq.min(dim=1, keepdim=True).values
            kq_max = kq.max(dim=1, keepdim=True).values
            kq = (kq - kq_min) / (kq_max - kq_min + cfg.score_eps)

            # V-norm: (kv_heads, T) -> (num_q, T)
            vn = value_states[0].norm(dim=-1)
            vn = vn.repeat_interleave(q_per_kv, dim=0)
            vn = vn / (vn.max(dim=1, keepdim=True).values + cfg.score_eps)

            # Distance decay — rate auto-derived from min_decay and T
            pos_idx = torch.arange(T, device=key_states.device, dtype=torch.float32)
            dist    = (T - 1) - pos_idx                       # (T,)  0 = last token
            pcfg    = self.pcfg
            if pcfg.decay_rate > 0.0:
                rate = pcfg.decay_rate
            elif T > 1:
                if pcfg.decay_fn == "linear":
                    rate = (1.0 - pcfg.min_decay) / (T - 1)
                else:
                    rate = -math.log(max(pcfg.min_decay, 1e-9)) / (T - 1)
            else:
                rate = 0.0
            if pcfg.decay_fn == "linear":
                decay = (1.0 - rate * dist).clamp(min=0.0)
            else:
                decay = torch.exp(-rate * dist)

            decay_b = decay.unsqueeze(0)  # (1, T) broadcast-ready
            if mode == "kq_only":
                head_scores = kq
            elif mode == "vn_only":
                head_scores = vn
            elif mode == "vn_decay":
                head_scores = vn * decay_b
            elif mode == "additive":
                a = self.pcfg.score_alpha
                head_scores = a * kq + (1.0 - a) * vn * decay_b
            else:  # "kq_vn_decay"
                head_scores = kq * vn * decay_b

        # Global aggregation
        global_scores = self._aggregate_head_scores(head_scores)  # (T,)

        # Always-keep head and tail tokens
        if budget <= 0:
            budget = self.pcfg.total_budget
        keep_last         = self.pcfg.always_keep_last
        keep_first        = self.pcfg.always_keep_first
        head_end          = min(keep_first, T)
        tail_start        = max(head_end, T - keep_last)
        always_set        = set(range(0, head_end)) | set(range(tail_start, T))

        # Top-k from the middle region
        middle_budget = max(0, budget - len(always_set))
        mid_start     = head_end
        mid_end       = tail_start   # exclusive
        mid_n         = max(0, mid_end - mid_start)

        if middle_budget > 0 and mid_n > 0:
            mid_scores = global_scores[mid_start:mid_end].clone()
            topk       = min(middle_budget, mid_n)
            _, top_idx = mid_scores.topk(topk)
            keep_set   = always_set | {mid_start + i for i in top_idx.tolist()}
        else:
            keep_set = always_set

        # retained: sorted original positions of kept tokens  (n_keep,)
        retained = torch.tensor(sorted(keep_set), dtype=torch.long,
                                device=self.device)

        # Gather the SAME positions for every KV head
        gathered_k = key_states[  :, :, retained, :]   # (B, kv_heads, n_keep, D)
        gathered_v = value_states[:, :, retained, :]

        return gathered_k, gathered_v, retained

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,         # transformers 4.x: Tuple[Tensor,Tensor] or Cache
        past_key_values=None,        # transformers 5.x: Cache (plural name)
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        B, T, _ = hidden_states.shape

        # Normalise cache: transformers 5.x passes past_key_values (plural).
        cache = past_key_values if past_key_values is not None else past_key_value

        # Standard Q/K/V projections
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states, self.num_heads)
        key_states   = self._split_heads(key_states,   self.num_key_value_heads)
        value_states = self._split_heads(value_states, self.num_key_value_heads)

        # Save pre-RoPE references for the importance filter.
        # apply_rotary_pos_emb returns new tensors (no in-place ops), so these
        # variables remain valid as pre-RoPE copies after the call below.
        q_raw = query_states
        k_raw = key_states

        # RoPE.
        # transformers 5.x pre-computes (cos, sin) at the model level and passes
        # them in as position_embeddings.  Older transformers: each attention
        # module owns a rotary_emb and computes them itself.
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # KV cache update.
        # transformers 5.x DynamicCache is updated in-place; sin/cos/cache_position
        # must be included in cache_kwargs for StaticCache compatibility.
        if cache is not None:
            if _HF_CACHE_API and isinstance(cache, HFCache):
                ckwargs: Dict = {"cache_position": cache_position,
                                 "sin": sin, "cos": cos}
                key_states, value_states = cache.update(
                    key_states, value_states, self.layer_idx, ckwargs
                )
            else:
                # Legacy tuple cache (transformers < 4.36)
                key_states   = torch.cat([cache[0], key_states],   dim=2)
                value_states = torch.cat([cache[1], value_states], dim=2)

        # past_kv_out is only meaningful for the legacy tuple path; for the Cache
        # API the object is mutated in-place and the decoder layer ignores the
        # return value anyway.
        if use_cache and not (_HF_CACHE_API and isinstance(cache, HFCache)):
            past_kv_out = (key_states, value_states)
        else:
            past_kv_out = cache

        # Apply per-head filter only during prefill (T > 1) if configured.
        # Skip when total_budget >= T: the budget already covers the full sequence,
        # so filtering would only discard tokens needlessly (sanity-check condition).
        retained_pos: Optional[torch.Tensor] = None   # original positions of kept KV tokens
        is_prefill = (T > 1)
        effective_budget = (int(T * self.pcfg.budget_fraction)
                            if self.pcfg.budget_fraction > 0.0
                            else self.pcfg.total_budget)
        if (is_prefill
                and self.pcfg.filter_prefill_only
                and self.head_filter is not None
                and effective_budget < T):
            key_states, value_states, retained_pos = self._run_filter_prefill(
                q_raw, k_raw, key_states, value_states, effective_budget,
                q_post=query_states,
            )

        # Expand KV heads to match Q heads (GQA support)
        key_states   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        kv_len = key_states.shape[2]
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Slice mask to match (possibly pruned) key length
            if attention_mask.shape[-1] > kv_len:
                attention_mask = attention_mask[..., :kv_len]
            attn_weights = attn_weights + attention_mask
        elif T > 1:
            # transformers 5.x passes attention_mask=None; build causal mask manually.
            dev   = query_states.device
            dtype = query_states.dtype
            zero  = torch.zeros((),    dtype=dtype, device=dev)
            neginf = torch.full((), float('-inf'), dtype=dtype, device=dev)

            q_idx = torch.arange(T, device=dev).unsqueeze(1)  # (T, 1)

            if retained_pos is not None:
                # After pruning, keys are at non-contiguous original positions.
                # Query i can attend to retained key j iff retained_pos[j] <= i.
                k_pos = retained_pos.unsqueeze(0)              # (1, n_keep)
                causal_mask = torch.where(k_pos <= q_idx, zero, neginf)
            else:
                # No pruning: standard lower-triangular mask.
                # offset > 0 when there are prefix tokens already in the cache.
                offset = kv_len - T
                k_idx  = torch.arange(kv_len, device=dev).unsqueeze(0)
                causal_mask = torch.where(k_idx <= q_idx + offset, zero, neginf)

            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)  # (B, H, T, D)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, T, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # transformers 5.x decoder layer unpacks 2 values (hidden, present_kv).
        # transformers 4.x decoder layer unpacks 3 values (hidden, attn_w, present_kv).
        if _TRANSFORMERS_V5:
            return (attn_output, past_kv_out)
        return (attn_output, attn_weights if output_attentions else None, past_kv_out)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_pruned_model(
    model_name: str,
    pruned_cfg: PrunedLlamaConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
    """
    Load a Llama 2 or Llama 3 model from HuggingFace and swap every
    LlamaAttention layer for a PrunedLlamaAttention.

    Returns (model, tokenizer).
    """
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )

    dev = torch.device(device)
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads
    head_dim   = model.config.hidden_size // num_heads
    kv_heads   = getattr(model.config, "num_key_value_heads", num_heads)

    filter_cfg = HeadFilterConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        total_budget=pruned_cfg.total_budget,
        q_buffer_size=pruned_cfg.q_buffer_size,
        budget_strategy=pruned_cfg.budget_strategy,
        decay_fn=pruned_cfg.decay_fn,
        decay_rate=pruned_cfg.decay_rate,
        always_keep_last=pruned_cfg.always_keep_last,
    )

    # One shared filter instance across all layers (each layer uses its layer_idx)
    shared_filter = PerHeadFilter(filter_cfg, device=device)

    print(f"Patching {num_layers} attention layers ...")
    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        pruned_attn = PrunedLlamaAttention(
            original=original_attn,
            pruned_cfg=pruned_cfg,
            layer_idx=layer_idx,
            device=dev,
        )
        pruned_attn.head_filter = shared_filter
        layer.self_attn = pruned_attn

    model.eval()
    print("Model ready.")
    return model, tokenizer

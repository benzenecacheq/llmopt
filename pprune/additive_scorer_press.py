"""
additive_scorer_press.py
------------------------
KVPress implementation of the additive KV cache scorer from:
  "Additive KV Cache Pruning: Combining Semantic Alignment and Value Norms
   for Long-Context Inference"

Scoring function:
    score_i = α · kq_i + (1 - α) · vn_i · decay_i

where:
  kq_i    = max over Q-buffer of pre-RoPE dot product (query · key_i / √d),
             normalized per head to [0, 1]
  vn_i    = L2 norm of value_i, normalized per head to [0, 1]
  decay_i = min_decay + (1 - min_decay) · (i / (T-1))
             linear ramp from min_decay at position 0 to 1.0 at position T-1

Pre-RoPE scoring removes the positional bias that makes early tokens appear
semantically irrelevant under standard post-RoPE scoring.

Usage
-----
    from additive_scorer_press import AdditiveScorerPress

    press = AdditiveScorerPress(compression_ratio=0.35)   # keep 65%
    with press(model):
        outputs = model.generate(input_ids, ...)
"""

from dataclasses import dataclass

import torch
from torch import nn

from kvpress import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class AdditiveScorerPress(ScorerPress):
    """
    Additive KV cache compression combining pre-RoPE KQ alignment and V-norm
    with linear distance decay.

    Parameters
    ----------
    compression_ratio : float
        Fraction of KV pairs to DROP. compression_ratio=0.35 keeps 65%
        of tokens, matching budget_fraction=0.65 in the paper.
    score_alpha : float
        Weight on the KQ alignment term. (1 - score_alpha) goes to V-norm × decay.
        Paper default: 0.65.
    min_decay : float
        Decay weight at position 0 (oldest token). Scales linearly to 1.0 at the
        most recent token. Context-length independent: the rate is auto-derived
        from min_decay and T, so behaviour is consistent at any sequence length.
        Paper default: 0.7.
    q_buffer_size : int
        Number of recent query vectors used for KQ scoring. Larger values give
        a better estimate of generation-phase attention but cost more memory.
        Paper default: 128.
    always_keep_last : int
        Always retain this many tail tokens regardless of score (protects the
        query context). Paper default: 16.
    always_keep_first : int
        Always retain this many head tokens regardless of score (prevents NaN
        from early queries having no visible keys). Paper default: 16.
    """

    compression_ratio: float = 0.35   # keep 65%
    score_alpha: float = 0.65
    min_decay: float = 0.7
    q_buffer_size: int = 128
    always_keep_last: int = 16
    always_keep_first: int = 16

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute per-token importance scores.

        Keys received here are pre-RoPE (KVPress guarantee), which is exactly
        what we need for position-independent KQ alignment.

        Parameters
        ----------
        module : attention module with q_proj, config.num_attention_heads, head_dim
        hidden_states : (bsz, seq_len, hidden_dim)
        keys : (bsz, num_kv_heads, seq_len, head_dim)  — pre-RoPE
        values : (bsz, num_kv_heads, seq_len, head_dim)
        attentions : may be None; not used by this scorer
        kwargs : forward-pass kwargs; not used by this scorer

        Returns
        -------
        scores : (bsz, num_kv_heads, seq_len)
            Higher score = more important = keep.
        """
        bsz, num_kv_heads, T, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        num_kv_groups = num_heads // num_kv_heads
        device = keys.device

        # ------------------------------------------------------------------
        # 1. Pre-RoPE query buffer
        #    get_prerope_query_states handles Llama / Mistral / Phi3 / Qwen3
        # ------------------------------------------------------------------
        buf_len = min(self.q_buffer_size, T)
        # (bsz, num_heads, buf_len, head_dim)
        q_raw = get_prerope_query_states(module, hidden_states[:, -buf_len:])

        # ------------------------------------------------------------------
        # 2. KQ alignment — pre-RoPE dot products
        # ------------------------------------------------------------------
        # Expand KV keys to match Q heads: (bsz, num_heads, T, head_dim)
        k_expanded = keys.repeat_interleave(num_kv_groups, dim=1)

        # (bsz, num_heads, buf_len, T)
        kq = torch.matmul(q_raw, k_expanded.transpose(-1, -2)) / (head_dim ** 0.5)

        # Max over Q buffer → (bsz, num_heads, T)
        kq = kq.amax(dim=2)

        # Normalize per head to [0, 1]
        kq_min = kq.amin(dim=-1, keepdim=True)
        kq_max = kq.amax(dim=-1, keepdim=True)
        kq = (kq - kq_min) / (kq_max - kq_min + 1e-8)

        # Aggregate Q heads → KV heads via max: (bsz, num_kv_heads, T)
        kq = kq.view(bsz, num_kv_heads, num_kv_groups, T).amax(dim=2)

        # ------------------------------------------------------------------
        # 3. V-norm
        # ------------------------------------------------------------------
        vn = values.norm(dim=-1)                          # (bsz, num_kv_heads, T)
        vn = vn / (vn.amax(dim=-1, keepdim=True) + 1e-8)

        # ------------------------------------------------------------------
        # 4. Linear distance decay
        #    decay[0] = min_decay (oldest), decay[T-1] = 1.0 (newest)
        # ------------------------------------------------------------------
        positions = torch.arange(T, device=device, dtype=keys.dtype)
        decay = self.min_decay + (1.0 - self.min_decay) * positions / max(T - 1, 1)
        # broadcasts over (bsz, num_kv_heads, T)

        # ------------------------------------------------------------------
        # 5. Additive combination
        # ------------------------------------------------------------------
        scores = self.score_alpha * kq + (1.0 - self.score_alpha) * vn * decay

        # ------------------------------------------------------------------
        # 6. Always-keep guards
        #    Assign max+1 to protected tokens so topk always selects them.
        # ------------------------------------------------------------------
        sentinel = scores.amax() + 1.0
        if self.always_keep_first > 0:
            n_first = min(self.always_keep_first, T)
            scores[:, :, :n_first] = sentinel
        if self.always_keep_last > 0:
            n_last = min(self.always_keep_last, T)
            scores[:, :, T - n_last:] = sentinel

        return scores

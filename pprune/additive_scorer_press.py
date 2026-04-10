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


@dataclass
class KStdAdaptiveAdditiveScorerPress(AdditiveScorerPress):
    """
    Adaptive variant that adjusts the recency decay per head based on the
    standard deviation of key vectors across the sequence.

    Motivation
    ----------
    In code-completion contexts, key vectors tend to be homogeneous — syntax
    tokens, operators, and common identifiers cluster closely in representation
    space, so KQ alignment loses discriminative power. In that regime, recency
    is a better guide than semantic similarity. In QA/narrative contexts, key
    vectors are diverse and semantic scoring works well.

    We measure homogeneity as the mean per-dimension std of key vectors across
    the sequence (k_std). Low k_std → homogeneous → favour recency. High k_std
    → diverse → favour semantic scoring. k_std is normalised per batch to
    [0, 1] and used to interpolate min_decay between min_decay_recency and
    min_decay_semantic, per head.

    Parameters
    ----------
    min_decay_semantic : float
        min_decay used for semantically diverse heads (standard setting, 0.7).
    min_decay_recency : float
        min_decay used for homogeneous heads (more recency bias, e.g. 0.2).
        A lower value makes the decay steeper, so recent tokens score higher
        relative to distant ones.
    """

    min_decay_semantic: float = 0.7    # standard min_decay — high k_std heads
    min_decay_recency:  float = 0.2    # recency-heavy — low k_std heads

    def score(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_kv_heads, T, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        num_kv_groups = num_heads // num_kv_heads
        device = keys.device

        # ------------------------------------------------------------------
        # 1. Pre-RoPE query buffer
        # ------------------------------------------------------------------
        buf_len = min(self.q_buffer_size, T)
        q_raw = get_prerope_query_states(module, hidden_states[:, -buf_len:])

        # ------------------------------------------------------------------
        # 2. KQ alignment (identical to AdditiveScorerPress)
        # ------------------------------------------------------------------
        k_expanded = keys.repeat_interleave(num_kv_groups, dim=1)
        kq = torch.matmul(q_raw, k_expanded.transpose(-1, -2)) / (head_dim ** 0.5)
        kq = kq.amax(dim=2)
        kq_min = kq.amin(dim=-1, keepdim=True)
        kq_max = kq.amax(dim=-1, keepdim=True)
        kq = (kq - kq_min) / (kq_max - kq_min + 1e-8)
        kq = kq.view(bsz, num_kv_heads, num_kv_groups, T).amax(dim=2)

        # ------------------------------------------------------------------
        # 3. V-norm
        # ------------------------------------------------------------------
        vn = values.norm(dim=-1)
        vn = vn / (vn.amax(dim=-1, keepdim=True) + 1e-8)

        # ------------------------------------------------------------------
        # 4. Adaptive decay: interpolate min_decay from k_std per head
        #
        #   k_std: mean across head_dim of per-position std across sequence
        #          shape (bsz, num_kv_heads)
        #   Normalise to [0,1] across heads so thresholding is scale-free.
        #   low  k_std_norm → homogeneous → min_decay_recency (steep decay)
        #   high k_std_norm → diverse     → min_decay_semantic (shallow decay)
        # ------------------------------------------------------------------
        # std across sequence positions, mean across head_dim
        k_std = keys.std(dim=2).mean(dim=-1)              # (bsz, num_kv_heads)
        k_std_min = k_std.amin(dim=-1, keepdim=True)
        k_std_max = k_std.amax(dim=-1, keepdim=True)
        k_std_norm = (k_std - k_std_min) / (k_std_max - k_std_min + 1e-8)
        # (bsz, num_kv_heads) in [0, 1]

        adaptive_min_decay = (
            self.min_decay_recency
            + k_std_norm * (self.min_decay_semantic - self.min_decay_recency)
        )  # (bsz, num_kv_heads)

        # Build per-head decay tensor: (bsz, num_kv_heads, T)
        positions = torch.arange(T, device=device, dtype=keys.dtype)
        # adaptive_min_decay[:, :, None] broadcasts over T
        decay = (
            adaptive_min_decay[:, :, None]
            + (1.0 - adaptive_min_decay[:, :, None]) * positions / max(T - 1, 1)
        )

        # ------------------------------------------------------------------
        # 5. Additive combination
        # ------------------------------------------------------------------
        scores = self.score_alpha * kq + (1.0 - self.score_alpha) * vn * decay

        # ------------------------------------------------------------------
        # 6. Always-keep guards
        # ------------------------------------------------------------------
        sentinel = scores.amax() + 1.0
        if self.always_keep_first > 0:
            n_first = min(self.always_keep_first, T)
            scores[:, :, :n_first] = sentinel
        if self.always_keep_last > 0:
            n_last = min(self.always_keep_last, T)
            scores[:, :, T - n_last:] = sentinel

        return scores


@dataclass
class EntropyAdaptiveAdditiveScorerPress(AdditiveScorerPress):
    """
    Adaptive variant that adjusts per-head recency bias based on the entropy
    of content-only importance scores (no decay term).

    Motivation
    ----------
    When the scorer is uncertain about which tokens matter (high entropy of
    content scores), recency is a more reliable fallback than semantic scoring.
    This happens in code contexts where many tokens have moderate, roughly equal
    relevance to the query. In targeted retrieval, one passage dominates and
    entropy is low — semantic scoring is trusted.

    Entropy is computed from content-only scores (α·kq + (1−α)·vn, no decay)
    to avoid circularity: the signal is independent of the recency weighting
    we are trying to set.

    min_decay per head:
        min_decay_i = min_decay_semantic × (1 − entropy_coefficient × entropy_norm_i)

    where entropy_norm_i ∈ [0,1] is the per-head entropy normalised across
    heads within the layer.

    At entropy_coefficient=0:   all heads use min_decay_semantic (reduces to
                                AdditiveScorerPress).
    At entropy_coefficient=0.5: max-entropy head gets min_decay = 0.35 (default),
                                zero-entropy head stays at 0.7.
    At entropy_coefficient=1.0: max-entropy head gets min_decay = 0 (pure recency).

    Parameters
    ----------
    min_decay_semantic : float
        Baseline min_decay for confident (low-entropy) heads. Default: 0.7.
    entropy_coefficient : float
        Controls how aggressively uncertainty drives recency. Start at 0.5.
    always_keep_last : int
        Unconditionally retained tail tokens (inherited from AdditiveScorerPress,
        default 16). Provides a recency floor independent of decay.
    """

    min_decay_semantic:   float = 0.7
    entropy_coefficient:  float = 0.5

    def score(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_kv_heads, T, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        num_kv_groups = num_heads // num_kv_heads
        device = keys.device

        # ------------------------------------------------------------------
        # 1. Pre-RoPE query buffer
        # ------------------------------------------------------------------
        buf_len = min(self.q_buffer_size, T)
        q_raw = get_prerope_query_states(module, hidden_states[:, -buf_len:])

        # ------------------------------------------------------------------
        # 2. KQ alignment (pre-RoPE)
        # ------------------------------------------------------------------
        k_expanded = keys.repeat_interleave(num_kv_groups, dim=1)
        kq = torch.matmul(q_raw, k_expanded.transpose(-1, -2)) / (head_dim ** 0.5)
        kq = kq.amax(dim=2)
        kq_min = kq.amin(dim=-1, keepdim=True)
        kq_max = kq.amax(dim=-1, keepdim=True)
        kq = (kq - kq_min) / (kq_max - kq_min + 1e-8)
        kq = kq.view(bsz, num_kv_heads, num_kv_groups, T).amax(dim=2)

        # ------------------------------------------------------------------
        # 3. V-norm
        # ------------------------------------------------------------------
        vn = values.norm(dim=-1)
        vn = vn / (vn.amax(dim=-1, keepdim=True) + 1e-8)

        # ------------------------------------------------------------------
        # 4. Content-only entropy (no decay) — used to set adaptive min_decay
        #
        #    Compute scores without decay first so the confidence signal is
        #    independent of the recency weighting we are about to set.
        # ------------------------------------------------------------------
        content_scores = self.score_alpha * kq + (1.0 - self.score_alpha) * vn
        probs = torch.softmax(content_scores, dim=-1)           # (bsz, num_kv_heads, T)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (bsz, num_kv_heads)

        # Normalise entropy to [0, 1] across heads within this layer
        ent_min = entropy.amin(dim=-1, keepdim=True)
        ent_max = entropy.amax(dim=-1, keepdim=True)
        entropy_norm = (entropy - ent_min) / (ent_max - ent_min + 1e-8)

        # Adaptive min_decay: high entropy → lower min_decay → more recency
        adaptive_min_decay = self.min_decay_semantic * (
            1.0 - self.entropy_coefficient * entropy_norm
        )  # (bsz, num_kv_heads), in [min_decay_semantic*(1-coeff), min_decay_semantic]

        # ------------------------------------------------------------------
        # 5. Per-head decay using adaptive min_decay
        # ------------------------------------------------------------------
        positions = torch.arange(T, device=device, dtype=keys.dtype)
        decay = (
            adaptive_min_decay[:, :, None]
            + (1.0 - adaptive_min_decay[:, :, None]) * positions / max(T - 1, 1)
        )  # (bsz, num_kv_heads, T)

        # ------------------------------------------------------------------
        # 6. Final additive combination
        # ------------------------------------------------------------------
        scores = self.score_alpha * kq + (1.0 - self.score_alpha) * vn * decay

        # ------------------------------------------------------------------
        # 7. Always-keep guards
        # ------------------------------------------------------------------
        sentinel = scores.amax() + 1.0
        if self.always_keep_first > 0:
            n_first = min(self.always_keep_first, T)
            scores[:, :, :n_first] = sentinel
        if self.always_keep_last > 0:
            n_last = min(self.always_keep_last, T)
            scores[:, :, T - n_last:] = sentinel

        return scores


@dataclass
class HeadAwareAdditiveScorerPress(AdditiveScorerPress):
    """
    Head-aware extension of AdditiveScorerPress.

    Different attention heads have different entropy profiles: some distribute
    attention broadly across the sequence (high entropy), some focus sharply on
    a few tokens (low entropy). A global compression ratio is too crude — it
    over-prunes high-entropy heads that genuinely need broad context and
    under-prunes low-entropy heads that concentrate on a small set.

    This press allocates the total token budget proportionally to per-head
    score entropy: heads with higher entropy receive more tokens. The total
    number of retained tokens is the same as AdditiveScorerPress at the same
    compression_ratio; only the distribution across heads differs.

    Parameters
    ----------
    min_head_ratio : float
        Minimum fraction of the global budget any single head may receive.
        Prevents degenerate allocation where a very low-entropy head is
        pruned to near-zero. Default: 0.5 (no head gets less than half the
        average allocation).
    """

    min_head_ratio: float = 0.5

    def compress(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        bsz, num_kv_heads, T, head_dim = keys.shape

        # Compute per-head importance scores (bsz, num_kv_heads, T)
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # ------------------------------------------------------------------
        # Per-head entropy from score distribution.
        # Use softmax to convert scores to a probability distribution before
        # computing entropy, so the scale of raw scores doesn't dominate.
        # ------------------------------------------------------------------
        probs = torch.softmax(scores, dim=-1)                          # (bsz, num_kv_heads, T)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)         # (bsz, num_kv_heads)

        # ------------------------------------------------------------------
        # Budget allocation proportional to entropy.
        # Total tokens to keep across all heads combined.
        # ------------------------------------------------------------------
        total_keep = int(T * (1.0 - self.compression_ratio))          # per-head budget (global)
        total_budget = total_keep * num_kv_heads                       # total tokens across all heads

        # Normalise entropy to get fractional allocation per head
        # Clamp minimum allocation to min_head_ratio × average
        avg_alloc = total_keep                                         # = total_budget / num_kv_heads
        min_alloc = max(1, int(avg_alloc * self.min_head_ratio))

        # (bsz, num_kv_heads) fractional weights
        ent_sum = entropy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        frac = entropy / ent_sum                                       # sums to 1.0 per batch

        # Convert to integer token counts, enforce minimum, re-normalise to total
        raw_alloc = (frac * total_budget).round().long()               # (bsz, num_kv_heads)
        raw_alloc = raw_alloc.clamp(min=min_alloc)

        # Re-scale to exactly hit total_budget (greedy: subtract/add from max-entropy heads)
        diff = total_budget - raw_alloc.sum(dim=-1, keepdim=True)
        # Distribute remainder by adding 1 to highest-entropy heads
        sorted_idx = entropy.argsort(dim=-1, descending=True)
        for b in range(bsz):
            d = diff[b, 0].item()
            if d == 0:
                continue
            step = 1 if d > 0 else -1
            for i in range(abs(int(d))):
                h = sorted_idx[b, i % num_kv_heads].item()
                raw_alloc[b, h] = (raw_alloc[b, h] + step).clamp(min=min_alloc, max=T)

        # ------------------------------------------------------------------
        # Gather per-head top-k using head-specific budgets.
        # Since each head may have a different n_kept we must loop over heads.
        # ------------------------------------------------------------------
        new_keys   = []
        new_values = []
        for h in range(num_kv_heads):
            n_kept = raw_alloc[0, h].item()          # bsz=1 assumed (KVPress standard)
            idx = scores[:, h, :].topk(n_kept, dim=-1).indices   # (bsz, n_kept)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, head_dim)
            new_keys.append(keys[:, h, :, :].gather(1, idx_expanded))    # (bsz, n_kept, head_dim)
            new_values.append(values[:, h, :, :].gather(1, idx_expanded))

        # Pad shorter heads to the same length so we can stack
        max_kept = max(t.shape[1] for t in new_keys)
        def pad(t, target_len):
            pad_len = target_len - t.shape[1]
            if pad_len == 0:
                return t
            return torch.cat([t, t[:, -1:, :].expand(-1, pad_len, -1)], dim=1)

        keys_out   = torch.stack([pad(k, max_kept) for k in new_keys],   dim=1)  # (bsz, H, max_kept, d)
        values_out = torch.stack([pad(v, max_kept) for v in new_values], dim=1)

        return keys_out.contiguous(), values_out.contiguous()

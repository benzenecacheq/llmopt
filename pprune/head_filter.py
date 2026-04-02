"""
head_filter.py
--------------
Per-head importance filter for KV token pruning.

For each attention head, scores every token by a combination of:
  1. KQ alignment   - dot product of token's K vector against a rolling Q buffer
                      sampled from the tail of the sequence
  2. V-norm         - magnitude of the token's V vector (max possible contribution)
  3. Distance decay - linear or exponential penalty for tokens far from sequence end

Maintains a per-head retention mask and supports two budget strategies:
  - equal:    each head keeps the same number of tokens (total_budget / num_heads)
  - entropy:  heads with higher historical attention entropy get proportionally
              more budget

Usage:
    from head_filter import HeadFilterConfig, PerHeadFilter

    cfg = HeadFilterConfig(
        num_heads=32,
        head_dim=128,
        total_budget=512,          # total KV slots across all heads
        q_buffer_size=64,          # how many recent Q vectors to track per head
        budget_strategy="entropy", # or "equal"
        decay_fn="exponential",    # or "linear"
        decay_rate=0.002,          # steepness of decay
    )
    filt = PerHeadFilter(cfg, device="cuda")

    # Call once per token as you stream tokens through
    filt.push_token(layer_idx, head_idx, k_vec, v_vec, q_vec)

    # After processing all tokens, get per-head retained indices
    masks = filt.get_masks()       # dict: (layer, head) -> LongTensor of indices
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class HeadFilterConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    total_budget: int                           # total retained tokens across ALL heads per layer
    q_buffer_size: int = 64                     # tail Q vectors to track per head
    budget_strategy: Literal["equal", "entropy"] = "both"
    decay_fn: Literal["linear", "exponential"] = "linear"
    decay_rate: float = 0.002                   # lambda for exponential; slope for linear
    score_eps: float = 1e-6                     # numerical stability floor
    always_keep_last: int = 16                  # unconditionally keep this many tail tokens


class _HeadState:
    """Accumulates per-token scores and Q buffer for a single (layer, head)."""

    def __init__(self, cfg: HeadFilterConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Rolling Q buffer — circular, fixed size.
        # After all tokens have been pushed this holds the last q_buffer_size
        # Q vectors — i.e. the tail of the sequence, which is the best proxy
        # for what the model will query at generation time.
        self.q_buffer: List[torch.Tensor] = []   # list of (head_dim,) tensors

        # K vectors stored for batch KQ scoring after all tokens are pushed.
        # KQ scores are NOT computed online because doing so would use Q vectors
        # from earlier in the sequence rather than the tail Q vectors that best
        # represent the generation-phase queries.
        self.k_vecs: List[torch.Tensor] = []     # list of (head_dim,) tensors

        # Per-token accumulators (grown dynamically as tokens arrive)
        self.v_norms: List[float] = []
        self.positions: List[int] = []           # absolute position index

        # Entropy tracking for budget allocation
        # We approximate attention entropy from the softmax over the Q buffer
        self.entropy_acc: float = 0.0
        self.entropy_count: int = 0

    def push(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, pos: int):
        """
        Record one token.
        k, v, q : (head_dim,) float tensors on device
        pos     : absolute position index of this token in the original sequence
        """
        cfg = self.cfg

        # Store K vector for batch KQ scoring in compute_scores()
        self.k_vecs.append(k.detach())

        # --- V-norm ---
        self.v_norms.append(v.norm().item())
        self.positions.append(pos)

        # Update rolling Q buffer (keep last q_buffer_size entries)
        self.q_buffer.append(q.detach())
        if len(self.q_buffer) > cfg.q_buffer_size:
            self.q_buffer.pop(0)

        # Update entropy estimate from softmax over Q buffer × current K.
        # This remains online because entropy is used for budget allocation,
        # not token selection, so approximation quality is less critical.
        if len(self.q_buffer) > 1:
            q_stack = torch.stack(self.q_buffer, dim=0)
            dots = (q_stack @ k) / math.sqrt(cfg.head_dim)
            probs = F.softmax(dots, dim=0)
            entropy = -(probs * (probs + cfg.score_eps).log()).sum().item()
            self.entropy_acc += entropy
            self.entropy_count += 1

    @property
    def mean_entropy(self) -> float:
        if self.entropy_count == 0:
            return 1.0
        return self.entropy_acc / self.entropy_count

    def compute_scores(self, seq_len: int) -> torch.Tensor:
        """
        Returns a (num_tokens,) importance score tensor combining
        KQ alignment, V-norm, and distance decay.

        KQ alignment is computed in batch here using the final Q buffer,
        so all tokens are scored against the tail Q vectors — the best
        available proxy for what the model will query at generation time.
        """
        cfg = self.cfg
        n = len(self.positions)
        if n == 0:
            return torch.empty(0, device=self.device)

        # Batch KQ alignment: (q_buf, head_dim) x (head_dim, n) -> (q_buf, n)
        # max over Q-buffer dimension gives per-token alignment score
        if len(self.q_buffer) > 0:
            q_stack = torch.stack(self.q_buffer, dim=0)          # (B, head_dim)
            k_stack = torch.stack(self.k_vecs,   dim=0)          # (n, head_dim)
            dots = (q_stack @ k_stack.T) / math.sqrt(cfg.head_dim)  # (B, n)
            kq = dots.max(dim=0).values                          # (n,)
        else:
            kq = torch.zeros(n, device=self.device, dtype=torch.float32)

        vn = torch.tensor(self.v_norms,   device=self.device, dtype=torch.float32)
        pos = torch.tensor(self.positions, device=self.device, dtype=torch.float32)

        # Normalize KQ and V-norm independently to [0,1]
        kq = (kq - kq.min()) / (kq.max() - kq.min() + cfg.score_eps)
        vn = vn / (vn.max() + cfg.score_eps)

        # Distance from end of sequence (0 = last token)
        dist = (seq_len - 1) - pos                              # (n,)

        if cfg.decay_fn == "linear":
            decay = 1.0 - cfg.decay_rate * dist
            decay = decay.clamp(min=0.0)
        else:  # exponential
            decay = torch.exp(-cfg.decay_rate * dist)

        return kq * vn * decay                                   # (n,)


class PerHeadFilter:
    """
    Top-level filter that manages one _HeadState per (layer, head) pair
    and produces retention masks after all tokens have been pushed.
    """

    def __init__(self, cfg: HeadFilterConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        # Keyed by (layer_idx, head_idx)
        self._states: Dict[Tuple[int, int], _HeadState] = {}
        self._seq_len: int = 0   # track total tokens pushed (same for all heads)

    def _get_state(self, layer: int, head: int) -> _HeadState:
        key = (layer, head)
        if key not in self._states:
            self._states[key] = _HeadState(self.cfg, self.device)
        return self._states[key]

    def push_token(
        self,
        layer_idx: int,
        head_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        pos: Optional[int] = None,
    ):
        """
        Register one token for (layer_idx, head_idx).
        k, v, q must be 1-D tensors of shape (head_dim,).
        pos defaults to the current push count if not supplied.
        """
        if pos is None:
            pos = self._seq_len
        self._get_state(layer_idx, head_idx).push(
            k.to(self.device).float(),
            v.to(self.device).float(),
            q.to(self.device).float(),
            pos,
        )
        # Advance global sequence counter only when head 0 layer 0 pushes
        # (caller is responsible for consistency)

    def advance_position(self):
        """Call once per token after pushing all layers/heads for that token."""
        self._seq_len += 1

    def _compute_budgets(self, strategy: str) -> Dict[Tuple[int, int], int]:
        """
        Allocate token budget per head.

        equal:   floor(total_budget / num_heads) for every head
        entropy: proportional to mean attention entropy of each head
        """
        cfg = self.cfg
        keys = list(self._states.keys())
        per_layer_heads = cfg.num_heads

        # Group keys by layer
        layers = sorted(set(k[0] for k in keys))
        budgets: Dict[Tuple[int, int], int] = {}

        for layer in layers:
            layer_keys = [(layer, h) for h in range(per_layer_heads)
                          if (layer, h) in self._states]

            if strategy == "equal":
                base = cfg.total_budget // per_layer_heads
                for key in layer_keys:
                    budgets[key] = base

            else:  # entropy
                entropies = torch.tensor(
                    [self._states[k].mean_entropy for k in layer_keys],
                    dtype=torch.float32,
                )
                weights = entropies / (entropies.sum() + self.cfg.score_eps)
                raw = (weights * cfg.total_budget).round().long()
                # Fix rounding so sum == total_budget
                diff = cfg.total_budget - raw.sum().item()
                raw[raw.argmax()] += int(diff)
                for key, b in zip(layer_keys, raw.tolist()):
                    budgets[key] = max(1, int(b))

        return budgets

    def get_masks(
        self,
        strategy: Optional[str] = None,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Returns per-(layer, head) LongTensors of retained token indices,
        sorted in ascending order (preserving sequence order).

        strategy: "equal" | "entropy" — overrides cfg if supplied.
        """
        cfg = self.cfg
        strat = strategy or cfg.budget_strategy
        if strat == "both":
            strat = "entropy"   # default when both are available

        budgets = self._compute_budgets(strat)
        seq_len = self._seq_len
        always_keep = cfg.always_keep_last
        masks: Dict[Tuple[int, int], torch.Tensor] = {}

        for key, state in self._states.items():
            n = len(state.positions)
            budget = budgets.get(key, cfg.total_budget // cfg.num_heads)

            # Clamp always_keep to at most half the per-head budget so there are
            # always some slots left for scored early tokens.
            effective_always_keep = min(always_keep, max(1, budget // 2))

            # Always-keep tail indices
            tail_start = max(0, n - effective_always_keep)
            tail_indices = set(range(tail_start, n))

            # Score all tokens
            scores = state.compute_scores(seq_len)             # (n,)

            # Zero out tail scores so they don't compete for early slots
            scores_early = scores.clone()
            scores_early[tail_start:] = -1.0

            # How many early slots do we have?
            early_budget = max(0, budget - len(tail_indices))
            early_n = tail_start

            if early_budget > 0 and early_n > 0:
                topk = min(early_budget, early_n)
                _, top_idx = scores_early[:early_n].topk(topk)
                keep = tail_indices | set(top_idx.tolist())
            else:
                keep = tail_indices

            kept = torch.tensor(sorted(keep), dtype=torch.long, device=self.device)
            masks[key] = kept

        return masks

    def reset(self):
        """Clear all state for reuse on a new sequence."""
        self._states.clear()
        self._seq_len = 0

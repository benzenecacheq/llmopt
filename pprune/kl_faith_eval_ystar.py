"""
kl_faith_eval_ystar.py
----------------------
KL faithfulness evaluation using y* (full-model tokens) as the shared prefix.

For each example:
  1. Generate y* from the uncompressed full model (output_scores=True captures
     P_full at each step for free — no second forward pass needed).
  2. For each compression method, teacher-force the compressed model on
     [comp_prompt + y*] and extract P_comp at the y* positions.
  3. Compute mean KL(P_full(· | full_ctx, y*_{<t}) || P_comp(· | comp_ctx, y*_{<t})).

All methods are scored on the same token sequence y*, eliminating the
path-dependence issue in kl_faith_eval.py where each method generated its own ŷ.

Loop order: task × example × method
  y* and P_full computed once per example; P_comp computed once per method.

Usage:
    python kl_faith_eval_ystar.py \\
        --model meta-llama/Llama-3.1-8B \\
        --data_dir lb_data_raw/data \\
        --output lb_results_base/kl_ystar.json \\
        --tasks 2wikimqa,narrativeqa \\
        --methods naive_65pct,kq_post_rope,snapkv,phrase_w64_t25 \\
        --n 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

try:
    from dataclasses import dataclass as _dataclass
    from kvpress import PyramidKVPress as _PyramidKVPress
    from kvpress import SnapKVPress as _SnapKVPress
    from kvpress import StreamingLLMPress as _StreamingLLMPress
    from kvpress import KeyRerotationPress as _KeyRerotationPress
    from kvpress.presses.scorer_press import ScorerPress as _ScorerPress
    from kvpress.utils import get_prerope_query_states as _get_prerope_query_states
    from transformers.models.llama.modeling_llama import rotate_half as _rotate_half
    import torch.nn as _nn

    @_dataclass
    class RadarPreRopePress(_ScorerPress):
        """Post-hoc KV eviction scored by pre-RoPE max-dot (position-invariant RADAR).

        Scores each key position by the maximum unnormalized dot product with the
        last `window_size` pre-RoPE queries.  Removing RoPE from both Q and K makes
        scoring position-independent: importance reflects semantic content, not where
        the token sits in the sequence.  Use with KeyRerotationPress for full fix.
        """
        compression_ratio: float = 0.0
        window_size: int = 128

        def score(self, module: _nn.Module, hidden_states: torch.Tensor,
                  keys: torch.Tensor, values: torch.Tensor,
                  attentions: torch.Tensor, kwargs) -> torch.Tensor:
            bsz, num_kv_heads, k_len, head_dim = keys.shape
            num_heads = module.config.num_attention_heads
            groups = num_heads // num_kv_heads

            # Pre-RoPE queries for last window positions only (avoids full q_proj)
            window = min(self.window_size, hidden_states.shape[1])
            q_win = _get_prerope_query_states(
                module, hidden_states[:, -window:, :]).float()  # (bsz, num_heads, window, head_dim)

            # Un-rotate keys to pre-RoPE: k_pre = k_post*cos - rotate_half(k_post)*sin
            cos, sin = kwargs["position_embeddings"]
            cos_k = cos.unsqueeze(1).float()  # (bsz, 1, k_len, head_dim)
            sin_k = sin.unsqueeze(1).float()
            k_pre = keys.float() * cos_k - _rotate_half(keys.float()) * sin_k
            k_pre = k_pre.repeat_interleave(groups, dim=1)  # (bsz, num_heads, k_len, head_dim)

            # Max dot product over query window → (bsz, num_heads, k_len)
            dots = torch.matmul(q_win, k_pre.transpose(2, 3)) / math.sqrt(head_dim)
            scores = dots.max(dim=2).values

            # Per-head [0,1] normalization
            s_min = scores.min(dim=-1, keepdim=True).values
            s_max = scores.max(dim=-1, keepdim=True).values
            scores = (scores - s_min) / (s_max - s_min + 1e-9)

            # Always keep observation window (match SnapKV convention)
            if window < k_len:
                scores[..., -window:] = scores.max()

            # Average over groups → (bsz, num_kv_heads, k_len)
            return scores.view(bsz, num_kv_heads, groups, k_len).mean(dim=2).to(keys.dtype)

    @_dataclass
    class RadarPostRopePress(_ScorerPress):
        """Post-hoc KV eviction scored by post-RoPE max-dot (position-aware RADAR).

        Same as RadarPreRopePress but uses post-RoPE Q and K, so scores are
        position-sensitive (tokens far from the query window are penalised by
        the rotational mismatch — the same bias that drove the original RADAR
        pre-rope design).  Useful ablation: pre vs. post when combined with
        rerotation, which corrects positions after eviction regardless of how
        selection was done.
        """
        compression_ratio: float = 0.0
        window_size: int = 128

        def score(self, module: _nn.Module, hidden_states: torch.Tensor,
                  keys: torch.Tensor, values: torch.Tensor,
                  attentions: torch.Tensor, kwargs) -> torch.Tensor:
            bsz, num_kv_heads, k_len, head_dim = keys.shape
            num_heads = module.config.num_attention_heads
            groups = num_heads // num_kv_heads

            # Post-RoPE queries for last window positions
            window = min(self.window_size, hidden_states.shape[1])
            q_pre = _get_prerope_query_states(
                module, hidden_states[:, -window:, :]).float()  # (bsz, num_heads, window, head_dim)
            cos, sin = kwargs["position_embeddings"]
            cos_q = cos[:, -window:].unsqueeze(1).float()  # (bsz, 1, window, head_dim)
            sin_q = sin[:, -window:].unsqueeze(1).float()
            q_win = q_pre * cos_q + _rotate_half(q_pre) * sin_q

            # Post-RoPE keys (from argument) expanded over groups
            k_post = keys.float().repeat_interleave(groups, dim=1)  # (bsz, num_heads, k_len, head_dim)

            # Max dot product over query window → (bsz, num_heads, k_len)
            dots = torch.matmul(q_win, k_post.transpose(2, 3)) / math.sqrt(head_dim)
            scores = dots.max(dim=2).values

            # Per-head [0,1] normalization
            s_min = scores.min(dim=-1, keepdim=True).values
            s_max = scores.max(dim=-1, keepdim=True).values
            scores = (scores - s_min) / (s_max - s_min + 1e-9)

            # Always keep observation window (match SnapKV convention)
            if window < k_len:
                scores[..., -window:] = scores.max()

            # Average over groups → (bsz, num_kv_heads, k_len)
            return scores.view(bsz, num_kv_heads, groups, k_len).mean(dim=2).to(keys.dtype)

    _RadarPreRopePress = RadarPreRopePress
    _RadarPostRopePress = RadarPostRopePress

    @_dataclass
    class PyramidKVRerotationPress(_PyramidKVPress):
        """PyramidKV per-layer budget + key re-rotation.

        KeyRerotationPress(PyramidKVPress) silently drops pyramid's layer-adaptive
        budget because KeyRerotationPress.compress() only calls .score() and applies
        a uniform n_kept.  This subclass preserves the pyramid budget by calling
        get_layer_budget() and then re-rotates retained keys to compact positions.
        """

        def compress(
            self,
            module: _nn.Module,
            hidden_states: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attentions: torch.Tensor,
            kwargs: dict,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.compression_ratio == 0:
                return keys, values

            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            q_len = keys.shape[2]
            n_kept = self.get_layer_budget(module, q_len)
            indices = scores.topk(n_kept, dim=-1).indices
            indices = torch.sort(indices, dim=2).values
            keys = _KeyRerotationPress.rerotate_keys(module, indices, keys)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            values = values.gather(2, indices).contiguous()
            return keys, values

    _PyramidKVRerotationPress = PyramidKVRerotationPress
    _KVPRESS_AVAILABLE = True
except ImportError:
    _PyramidKVPress = None
    _SnapKVPress = None
    _StreamingLLMPress = None
    _KeyRerotationPress = None
    _RadarPreRopePress = None
    _RadarPostRopePress = None
    _PyramidKVRerotationPress = None
    _KVPRESS_AVAILABLE = False

from llama_pruned import (
    HeadFilterConfig,
    PerHeadFilter,
    PrunedLlamaAttention,
    PrunedLlamaConfig,
)

# ---------------------------------------------------------------------------
# Import shared config from kl_faith_eval (tasks, prompts, method configs)
# ---------------------------------------------------------------------------

from kl_faith_eval import (
    ENGLISH_TASKS,
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_COMP,
    DEFAULT_MAX_SEQ_FULL,
    TASK_MAX_SEQ_COMP,
    TASK_MAX_SEQ_FULL,
    METHOD_CONFIGS,
    CHUNK_CONFIGS,
    patch_model,
    unpatch_model,
    naive_truncate,
    streaming_truncate,
    chunk_truncate,
    sent_truncate,
    load_ckpt,
    save_ckpt,
    fmt_duration,
    fmt_eta,
    YSTAR_STOP_TOKENS,
)

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class _FirstTokenTimer(LogitsProcessor):
    """Captures wall-clock TTFT via the first LogitsProcessor callback in generate()."""
    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self.ttft_s: Optional[float] = None

    def arm(self) -> None:
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.ttft_s is None and self._t0 is not None:
            torch.cuda.synchronize()
            self.ttft_s = time.perf_counter() - self._t0
        return scores


@torch.inference_mode()
def time_comp_prefill_decode(
    model,
    comp_ids: torch.Tensor,
    ystar: torch.Tensor,
    pcfg: Optional[PrunedLlamaConfig],
    device: str,
    n_decode: int = 5,
    press=None,
) -> dict:
    """Measure TTFT and TPT for a compression method.

    TTFT: wall time to process the compressed context in a single prefill pass.
    TPT:  mean wall time per single-token decode step (KV cache grown from prefill).

    For KV-pruning methods (pcfg set), pruning fires during the prefill.
    For select methods (pcfg=None), comp_ids is already the scored subset.
    For kvpress methods (press set), the press context manager wraps the prefill.
    sel_ms (scoring-pass overhead) must be timed separately in the main loop.
    """
    n_decode = min(n_decode, max(ystar.shape[0], 1))
    comp_input = comp_ids.to(device)
    originals  = None
    if pcfg is not None:
        originals = patch_model(model, pcfg, device)

    ctx = press(model) if press is not None else nullcontext()
    ttft_ms = tpt_ms = None
    try:
        torch.cuda.synchronize()
        t0  = time.perf_counter()
        with ctx:
            out = model(
                input_ids=comp_input,
                attention_mask=torch.ones_like(comp_input),
                use_cache=True,
            )
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        kv = out.past_key_values
        del out
        step_ms: list = []
        for i in range(n_decode):
            tok = ystar[i : i + 1].unsqueeze(0).to(device)
            torch.cuda.synchronize()
            ts       = time.perf_counter()
            step_out = model(input_ids=tok, past_key_values=kv, use_cache=True)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - ts) * 1000)
            kv = step_out.past_key_values
        del kv
        tpt_ms = sum(step_ms) / len(step_ms) if step_ms else None

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    finally:
        if originals is not None:
            unpatch_model(model, originals)

    torch.cuda.empty_cache()
    return {"ttft_ms": ttft_ms, "tpt_ms": tpt_ms}


# ---------------------------------------------------------------------------
# SnapKV-scored input selection (snapkv_select family)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def scored_select_ids(
    model,
    full_ids: torch.Tensor,
    device: str,
    score_mode: str,
    fraction: float = 0.65,
    min_decay: float = 0.7,
    always_keep_first: int = 16,
    always_keep_last: int = 16,
    n_score_layers: Optional[int] = None,
    pyramid_weighted: bool = False,
) -> Optional[torch.Tensor]:
    """
    Score tokens via a forward pass using the given score_mode, then return the
    top-k as a clean input tensor for honest prefill.  No KV cache corruption.

    score_mode: "snapkv" or "kq_post_rope" (RADAR)
    n_score_layers: use only the first N layers for scoring (None = all layers).
                    Cheaper proxy experiment: 1, 4, 8 vs. full 32.
    """
    T = full_ids.shape[1]
    budget = max(1, int(T * fraction))

    score_capture: list = []
    pcfg = PrunedLlamaConfig(
        score_mode=score_mode,
        snapkv_window=32,
        min_decay=min_decay,
        budget_fraction=1.0,       # score_capture triggers scoring even though budget >= T
        always_keep_first=0,
        always_keep_last=0,
        q_buffer_size=128,
        score_capture=score_capture,
    )
    originals = patch_model(model, pcfg, device)

    # For partial-layer scoring: disable head_filter on layers beyond n_score_layers.
    # Those layers still run as PrunedLlamaAttention but with head_filter=None,
    # so they skip _run_filter_prefill entirely and behave like standard attention.
    if n_score_layers is not None:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= n_score_layers:
                layer.self_attn.head_filter = None

    try:
        _ = model(
            input_ids=full_ids.to(device),
            attention_mask=torch.ones(1, T, device=device, dtype=torch.long),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        unpatch_model(model, originals)

    if not score_capture:
        return None

    # Aggregate per-layer scores.
    stacked = torch.stack(score_capture, dim=0)   # (n_layers, T)
    if pyramid_weighted:
        # Weight layer i by PyramidKV's budget for that layer:
        # lower layers get more weight, matching the pyramid allocation.
        n_layers = stacked.shape[0]
        comp_ratio = 1 - fraction
        window_size = 128; beta = 20
        max_cap = window_size + T * (1 - comp_ratio)
        min_num = (max_cap - window_size) / beta
        max_num = (max_cap - window_size) * 2 - min_num
        max_num = min(max_num, T - window_size)
        min_num = (max_cap - window_size) * 2 - max_num
        steps = (max_num - min_num) / max(n_layers - 1, 1)
        budgets = torch.tensor([max(1.0, max_num - i * steps) for i in range(n_layers)],
                               dtype=torch.float32)
        weights = budgets / budgets.sum()
        avg_scores = (stacked * weights.unsqueeze(1)).sum(dim=0)
    else:
        avg_scores = stacked.mean(dim=0)   # (T,)

    # Select tokens using the same always-keep + top-k logic as _run_filter_prefill
    head_end   = min(always_keep_first, T)
    tail_start = max(head_end, T - always_keep_last)
    always_set = set(range(0, head_end)) | set(range(tail_start, T))

    mid_start     = head_end
    mid_end       = tail_start
    mid_n         = max(0, mid_end - mid_start)
    middle_budget = max(0, budget - len(always_set))

    if middle_budget > 0 and mid_n > 0:
        mid_scores = avg_scores[mid_start:mid_end]
        topk       = min(middle_budget, mid_n)
        _, top_idx = mid_scores.topk(topk)
        keep_set   = always_set | {mid_start + i for i in top_idx.tolist()}
    else:
        keep_set = always_set

    retained = torch.tensor(sorted(keep_set), dtype=torch.long)
    return full_ids[:, retained]


# ---------------------------------------------------------------------------
# Compressed prompt construction (mirrors kl_faith_eval.kl_for_example)
# ---------------------------------------------------------------------------

# Maps select method name → (score_mode, fraction, n_score_layers, pyramid_weighted)
_SELECT_METHODS = {}
for _name, _mode in [("snapkv_select", "snapkv"), ("radar_select", "kq_post_rope")]:
    _SELECT_METHODS[_name] = (_mode, 0.65, None, False)
    for _frac in (0.50, 0.40, 0.35):
        _fpct = int(_frac * 100)
        _SELECT_METHODS[f"{_name}_f{_fpct}"] = (_mode, _frac, None, False)
    for _nl in (1, 4, 8):
        _SELECT_METHODS[f"{_name}_l{_nl}"] = (_mode, 0.65, _nl, False)

# pyramidkv_select: SnapKV attention scoring weighted by PyramidKV's layer budgets,
# then reconstructed as a gapless prompt. Separates token selection from KV patching.
_SELECT_METHODS["pyramidkv_select"] = ("snapkv", 0.65, None, True)
for _frac in (0.50, 0.40, 0.35):
    _SELECT_METHODS[f"pyramidkv_select_f{int(_frac*100)}"] = ("snapkv", _frac, None, True)


# ---------------------------------------------------------------------------
# Vocabulary-salience selection (static lookup table, zero inference overhead)
# ---------------------------------------------------------------------------

# Maps salience method name → (stat, fraction, unseen_policy)
# stat:          which aggregation to use from the calibration table
# unseen_policy: "keep"  → unseen tokens always selected (score = +inf)
#                "global"→ unseen tokens get the saved global mean score
_SALIENCE_METHODS: dict = {}
for _stat in ("mean", "max", "median"):
    _SALIENCE_METHODS[f"salience_{_stat}"]       = (_stat, 0.65, "keep")
    _SALIENCE_METHODS[f"salience_{_stat}_drop"]  = (_stat, 0.65, "global")
    for _frac in (0.50, 0.40, 0.35):
        _SALIENCE_METHODS[f"salience_{_stat}_f{int(_frac*100)}"] = (_stat, _frac, "keep")

_salience_cache: Optional[dict] = None   # loaded once per process

# ---------------------------------------------------------------------------
# KVPress-based methods (uses kvpress context manager; no llama_pruned patching)
# ---------------------------------------------------------------------------

_KVPRESS_METHODS: dict = {}
if _KVPRESS_AVAILABLE:
    _KVPRESS_METHODS["pyramidkv"] = _PyramidKVPress(
        compression_ratio=0.35,   # keep 65% — matches r=0.65
        window_size=128,
        beta=20,
    )
    _KVPRESS_METHODS["pyramidkv_f50"] = _PyramidKVPress(
        compression_ratio=0.50,   # keep 50%
        window_size=128,
        beta=20,
    )
    _KVPRESS_METHODS["pyramidkv_f40"] = _PyramidKVPress(
        compression_ratio=0.60,   # keep 40%
        window_size=128,
        beta=20,
    )
    _KVPRESS_METHODS["pyramidkv_f35"] = _PyramidKVPress(
        compression_ratio=0.65,   # keep 35%
        window_size=128,
        beta=20,
    )
    # "short" variants: identical compression config, but gt_eval_compression.py
    # caps their input at max_seq_comp (~4096) instead of max_seq_full (~7168).
    # This isolates the effect of long-context prefill from compression quality.
    _KVPRESS_METHODS["pyramidkv_short"] = _PyramidKVPress(
        compression_ratio=0.35,   # keep 65% — same as pyramidkv
        window_size=128,
        beta=20,
    )
    # No-window variant: identical compression but window_size=0 removes the
    # unconditional tail-retention guarantee.  Tests whether always_keep_last
    # drives first-token accuracy on tail-anchored tasks (TREC, LCC).
    _KVPRESS_METHODS["pyramidkv_no_window"] = _PyramidKVPress(
        compression_ratio=0.35,   # keep 65%
        window_size=1,            # minimum valid; effectively removes tail retention
        beta=20,
    )
    # "clean_first" variant: same compression config, but gt_eval_compression.py
    # prefills only full_ids[:, :-1] and then decodes last_tok against the
    # compressed KV cache.  T1 therefore attends to compressed context (same
    # as T2+), eliminating the full-prefill first-token advantage.
    _KVPRESS_METHODS["pyramidkv_clean_first"] = _PyramidKVPress(
        compression_ratio=0.35,   # keep 65% — same as pyramidkv
        window_size=128,
        beta=20,
    )
    _KVPRESS_METHODS["pyramidkv_f50_clean_first"] = _PyramidKVPress(
        compression_ratio=0.50,   # keep 50% — same as pyramidkv_f50
        window_size=128,
        beta=20,
    )
    _KVPRESS_METHODS["pyramidkv_f40_clean_first"] = _PyramidKVPress(
        compression_ratio=0.60,   # keep 40% — same as pyramidkv_f40
        window_size=128,
        beta=20,
    )
    _KVPRESS_METHODS["pyramidkv_f35_clean_first"] = _PyramidKVPress(
        compression_ratio=0.65,   # keep 35% — same as pyramidkv_f35
        window_size=128,
        beta=20,
    )

    # SnapKV via kvpress's standard post-hoc forward_hook: prunes the *stored*
    # cache after each layer's full, unrestricted prefill attention has already
    # run. This is the mechanism PyramidKV already uses above, and the one
    # SnapKV is conventionally implemented and benchmarked with — distinct from
    # this paper's own PrunedLlamaAttention path (snapkv in METHOD_CONFIGS),
    # which retroactively restricts attention *during* prefill at every layer.
    # Comparing snapkv_press to its pcfg counterpart isolates mechanism
    # (prefill-cascade vs. decode-only post-hoc eviction) from method (scoring
    # algorithm).
    _KVPRESS_METHODS["snapkv_press"] = _SnapKVPress(compression_ratio=0.35, window_size=128)
    _KVPRESS_METHODS["snapkv_press_f50"] = _SnapKVPress(compression_ratio=0.50, window_size=128)
    _KVPRESS_METHODS["snapkv_press_f40"] = _SnapKVPress(compression_ratio=0.60, window_size=128)
    _KVPRESS_METHODS["snapkv_press_f35"] = _SnapKVPress(compression_ratio=0.65, window_size=128)

    # SnapKV + re-rotation (diagnostic): same attention-score token selection as
    # snapkv_press, but remaining keys are re-rotated to compact positions after
    # eviction (identical to streaming_rerotated's position-correction step).
    # Isolates re-rotation effect from gap geometry: if this is close to snapkv_press,
    # geometry drives the KL gap; if much lower, re-rotation is the dominant factor.
    _KVPRESS_METHODS["snapkv_rerotated"] = _KeyRerotationPress(
        press=_SnapKVPress(compression_ratio=0.35, window_size=128))
    _KVPRESS_METHODS["snapkv_rerotated_f50"] = _KeyRerotationPress(
        press=_SnapKVPress(compression_ratio=0.50, window_size=128))
    _KVPRESS_METHODS["snapkv_rerotated_f40"] = _KeyRerotationPress(
        press=_SnapKVPress(compression_ratio=0.60, window_size=128))
    _KVPRESS_METHODS["snapkv_rerotated_f35"] = _KeyRerotationPress(
        press=_SnapKVPress(compression_ratio=0.65, window_size=128))

    # Streaming: only the KeyRerotationPress-wrapped version is used, since that
    # is the one that matches the published StreamingLLM algorithm. Per kvpress's
    # own StreamingLLMPress docstring, this wrapper is required "to fully match
    # the implementation described in the paper" -- without it, retained keys
    # keep their original absolute positions, the same positional-misalignment
    # failure mode as this paper's custom "Streaming" baseline (§3.1/§6.1). We
    # deliberately do not also register the unwrapped variant: running Streaming
    # any way other than as the original authors specified would just add a
    # third, uncited configuration to compare against.
    _KVPRESS_METHODS["streaming_rerotated"] = _KeyRerotationPress(
        press=_StreamingLLMPress(compression_ratio=0.35, n_sink=4))
    _KVPRESS_METHODS["streaming_rerotated_f50"] = _KeyRerotationPress(
        press=_StreamingLLMPress(compression_ratio=0.50, n_sink=4))
    _KVPRESS_METHODS["streaming_rerotated_f40"] = _KeyRerotationPress(
        press=_StreamingLLMPress(compression_ratio=0.60, n_sink=4))
    _KVPRESS_METHODS["streaming_rerotated_f35"] = _KeyRerotationPress(
        press=_StreamingLLMPress(compression_ratio=0.65, n_sink=4))

    # PyramidKV + re-rotation: per-layer pyramid budget (not uniform) combined with
    # key re-rotation.  Uses PyramidKVRerotationPress, which calls get_layer_budget()
    # before re-rotating — unlike KeyRerotationPress(PyramidKVPress), which ignores
    # the pyramid budget and silently degrades to SnapKV+rot.
    _KVPRESS_METHODS["pyramidkv_rerotated"] = _PyramidKVRerotationPress(
        compression_ratio=0.35, window_size=128, beta=20)
    _KVPRESS_METHODS["pyramidkv_rerotated_f50"] = _PyramidKVRerotationPress(
        compression_ratio=0.50, window_size=128, beta=20)
    _KVPRESS_METHODS["pyramidkv_rerotated_f35"] = _PyramidKVRerotationPress(
        compression_ratio=0.65, window_size=128, beta=20)

    # RADAR pre-RoPE and post-RoPE: post-hoc eviction with max-dot scoring.
    # Pre-RoPE (position-invariant): scores reflect semantic content similarity only.
    # Post-RoPE (position-aware): same bias as the attention kernel, ablates the
    # pre-RoPE choice.  Both wrapped with KeyRerotationPress for the rerotated
    # variant; bare presses registered for the eviction-only baseline.
    for _suffix, _cr in [("", 0.35), ("_f50", 0.50), ("_f35", 0.65)]:
        _KVPRESS_METHODS[f"radar_pre_press{_suffix}"]      = _RadarPreRopePress(compression_ratio=_cr, window_size=128)
        _KVPRESS_METHODS[f"radar_post_press{_suffix}"]     = _RadarPostRopePress(compression_ratio=_cr, window_size=128)
        _KVPRESS_METHODS[f"radar_pre_rerotated{_suffix}"]  = _KeyRerotationPress(
            press=_RadarPreRopePress(compression_ratio=_cr, window_size=128))
        _KVPRESS_METHODS[f"radar_post_rerotated{_suffix}"] = _KeyRerotationPress(
            press=_RadarPostRopePress(compression_ratio=_cr, window_size=128))

    for _cm in _KVPRESS_METHODS:
        METHOD_CONFIGS[_cm] = None


def load_salience_table(path: str) -> dict:
    """Load calibration table from disk; cache in module-level variable."""
    global _salience_cache
    if _salience_cache is None:
        _salience_cache = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Salience table loaded: {path}  "
              f"(vocab={_salience_cache['vocab_size']}, "
              f"seqs={_salience_cache.get('total_seqs','?')})", flush=True)
    return _salience_cache


def salience_select_ids(
    full_ids: torch.Tensor,         # (1, T)
    table: dict,
    stat: str          = "mean",    # "mean" | "max" | "median"
    fraction: float    = 0.65,
    unseen_policy: str = "keep",    # "keep" | "global"
    always_keep_first: int = 16,
    always_keep_last:  int = 16,
) -> torch.Tensor:
    """Select the top-fraction tokens by pre-computed salience score.

    No model forward pass.  Runs in microseconds — pure tensor indexing.

    unseen_policy="keep"  : tokens absent from calibration corpus get score=+inf
                            (always retained — they're rare and may be critical).
    unseen_policy="global": unseen tokens get the global mean score of seen tokens.
    """
    token_ids = full_ids[0]                       # (T,)
    T         = token_ids.shape[0]
    budget    = max(1, int(T * fraction))

    scores_table: torch.Tensor = table[stat]      # (vocab_size,)
    raw_scores = scores_table[token_ids]           # (T,) — lookup per position

    unseen_mask = raw_scores < 0                   # sentinel = -1.0
    if unseen_mask.any():
        if unseen_policy == "keep":
            raw_scores = raw_scores.clone()
            raw_scores[unseen_mask] = float("inf")
        else:
            raw_scores = raw_scores.clone()
            raw_scores[unseen_mask] = table[f"global_{stat}"]

    # always_keep_first / always_keep_last logic (mirrors scored_select_ids)
    head_end   = min(always_keep_first, T)
    tail_start = max(head_end, T - always_keep_last)
    always_set = set(range(head_end)) | set(range(tail_start, T))

    mid_start     = head_end
    mid_end       = tail_start
    mid_n         = max(0, mid_end - mid_start)
    middle_budget = max(0, budget - len(always_set))

    if middle_budget > 0 and mid_n > 0:
        mid_scores = raw_scores[mid_start:mid_end]
        topk       = min(middle_budget, mid_n)
        _, top_idx = mid_scores.topk(topk)
        keep_set   = always_set | {mid_start + i for i in top_idx.tolist()}
    else:
        keep_set = always_set

    retained = torch.tensor(sorted(keep_set), dtype=torch.long)
    return full_ids[:, retained]


def make_comp_ids(
    full_ids: torch.Tensor,
    method: str,
    tokenizer,
    question_text: str,
    max_seq_comp: int,
    model=None,
    device: str = "cuda",
    salience_table: Optional[dict] = None,
) -> torch.Tensor:
    """Return the compressed prompt token ids for a given method."""
    if method == "naive_65pct":
        comp_ids = naive_truncate(full_ids, head_frac=0.10)
    elif method.startswith("naive_") and method.endswith("pct") and method != "naive_tail":
        frac = int(method[6:-3]) / 100.0
        comp_ids = naive_truncate(full_ids, fraction=frac, head_frac=0.10)
    elif method == "naive_tail":
        comp_ids = naive_truncate(full_ids, head_frac=0.0)
    elif method == "streaming_select":
        comp_ids = streaming_truncate(full_ids, fraction=0.65, sink=4)
    elif method.startswith("streaming_select_f"):
        frac = int(method.rsplit("_f", 1)[1]) / 100.0
        comp_ids = streaming_truncate(full_ids, fraction=frac, sink=4)
    elif method == "chunk_sent":
        q_ids = (tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt")
                 if question_text else None)
        comp_ids = sent_truncate(full_ids, q_ids, tokenizer=tokenizer)
    elif method in CHUNK_CONFIGS:
        q_ids = (tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt")
                 if question_text else None)
        comp_ids = chunk_truncate(full_ids, q_ids, tokenizer=tokenizer, **CHUNK_CONFIGS[method])
    elif method in _SELECT_METHODS and model is not None:
        score_mode, frac, n_layers, pyr_weighted = _SELECT_METHODS[method]
        ids = scored_select_ids(model, full_ids, device, score_mode=score_mode,
                                fraction=frac, n_score_layers=n_layers,
                                pyramid_weighted=pyr_weighted)
        comp_ids = ids if ids is not None else naive_truncate(full_ids, fraction=frac, head_frac=0.10)
    elif method in _SALIENCE_METHODS and salience_table is not None:
        stat, frac, unseen_policy = _SALIENCE_METHODS[method]
        comp_ids = salience_select_ids(full_ids, salience_table,
                                       stat=stat, fraction=frac,
                                       unseen_policy=unseen_policy)
    elif method in _KVPRESS_METHODS:
        # KV compression is handled by the kvpress hook during the forward pass.
        # Return the full context (caller passes max_seq_full as max_seq_comp for these methods).
        comp_ids = full_ids.clone()
    else:
        comp_ids = full_ids.clone()

    if comp_ids.shape[1] > max_seq_comp:
        comp_ids = comp_ids[:, -max_seq_comp:]
    return comp_ids

# ---------------------------------------------------------------------------
# Per-example core functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_ystar(
    model,
    full_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_full: int,
    device: str,
    stop_token_ids: Optional[list] = None,
):
    """Generate y* from the uncompressed model.

    Returns (gen_tokens, log_p_full, timing) where:
      gen_tokens  : (n_gen,)        int64 — greedy tokens
      log_p_full  : (n_gen, vocab)  float32 — log-softmax at each step
      timing      : dict with ttft_ms and tpt_ms, or None on cache hit / OOM

    Returns (None, None, None) on OOM or empty generation.

    stop_token_ids: if given, generation halts when any of these token IDs is
    produced (in addition to EOS).  Pass YSTAR_STOP_TOKENS[task] for short-answer
    tasks so that y* contains only the answer and not the model regurgitating the
    next few-shot context block.
    """
    prompt_ids = full_ids
    if prompt_ids.shape[1] > max_seq_full:
        prompt_ids = prompt_ids[:, -max_seq_full:]
    prompt_ids = prompt_ids.to(device)
    n_prompt = prompt_ids.shape[1]

    # Build the full set of token IDs that should end generation
    eos_id = model.config.eos_token_id
    if isinstance(eos_id, int):
        eos_id = [eos_id]
    elif eos_id is None:
        eos_id = []
    extra = stop_token_ids or []
    all_stop = list(dict.fromkeys(eos_id + extra))  # deduplicated, order-preserving

    timer = _FirstTokenTimer()
    try:
        timer.arm()
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=all_stop,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=LogitsProcessorList([timer]),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None

    torch.cuda.synchronize()
    t_total_s = time.perf_counter() - timer._t0 if timer._t0 is not None else None

    if not out.scores:
        return None, None, None

    gen_tokens = out.sequences[0, n_prompt:].cpu()
    raw_logits = torch.stack([s[0] for s in out.scores], dim=0)
    log_p_full = F.log_softmax(raw_logits.float(), dim=-1).cpu()

    del out
    torch.cuda.empty_cache()

    if gen_tokens.shape[0] == 0:
        return None, None, None

    n_gen  = gen_tokens.shape[0]
    timing = None
    if timer.ttft_s is not None and t_total_s is not None:
        tpt_s  = (t_total_s - timer.ttft_s) / max(n_gen - 1, 1)
        timing = {"ttft_ms": timer.ttft_s * 1000, "tpt_ms": tpt_s * 1000}

    return gen_tokens, log_p_full, timing


@torch.inference_mode()
def get_comp_log_probs(
    model,
    comp_ids: torch.Tensor,
    ystar: torch.Tensor,
    pcfg: Optional[PrunedLlamaConfig],
    device: str,
    press=None,
) -> Optional[torch.Tensor]:
    """Teacher-force compressed model on [comp_ids + ystar].

    Returns log_p_comp (n_gen, vocab) float32, or None on OOM/shape mismatch.

    pcfg and press are mutually exclusive:
      pcfg  — our PrunedLlamaAttention monkey-patching
      press — a kvpress context manager (e.g. PyramidKVPress)

    For press methods: kvpress compresses the KV *cache* (stored K/V pairs), not
    the attention computation itself.  A single-pass teacher-force over
    [comp_ids + ystar] gives KL≈0 because the compression hooks only prune what
    gets stored in past_key_values — the causal attention scores are unchanged.
    The correct evaluation path is:
      1. Prefill comp_ids under press → pruned KV cache
      2. Step-decode each ystar token using the pruned cache
    This matches how PyramidKVPress is used in practice (generate() style).
    """
    n_gen = ystar.shape[0]

    if press is not None:
        comp_input = comp_ids.to(device)
        try:
            with press(model):
                out = model(
                    input_ids=comp_input,
                    attention_mask=torch.ones_like(comp_input),
                    use_cache=True,
                )
            kv = out.past_key_values
            # out.logits[0, -1, :] is P_comp predicting y*[0]; capture before
            # del out or position 0 is lost (off-by-one bug in step-decode loop).
            logits_list = [out.logits[0, -1, :].float()]
            del out

            for t in range(n_gen - 1):
                tok = ystar[t : t + 1].unsqueeze(0).to(device)
                step_out = model(input_ids=tok, past_key_values=kv, use_cache=True)
                logits_list.append(step_out.logits[0, -1, :].float())
                kv = step_out.past_key_values
            del kv

            if not logits_list:
                return None
            comp_logits = torch.stack(logits_list, dim=0)   # (n_gen, vocab)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

        return F.log_softmax(comp_logits, dim=-1).cpu()

    # --- Standard path: single forward pass over [comp_ids + ystar] ---
    full_input = torch.cat([comp_ids.to(device), ystar.unsqueeze(0).to(device)], dim=1)
    n_prompt_comp = comp_ids.shape[1]

    originals = None
    if pcfg is not None:
        originals = patch_model(model, pcfg, device)

    try:
        out = model(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input),
        )
        # Positions [n_prompt_comp-1 .. -1] predict ystar[0..n_gen-1]
        comp_logits = out.logits[0, n_prompt_comp - 1 : -1, :].float()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        comp_logits = None
    finally:
        if originals is not None:
            unpatch_model(model, originals)

    if comp_logits is None or comp_logits.shape[0] != n_gen:
        return None

    return F.log_softmax(comp_logits, dim=-1).cpu()


@torch.inference_mode()
def get_full_hidden_state(
    model,
    full_ids: torch.Tensor,
    ystar: torch.Tensor,
    device: str,
    max_seq_full: int,
) -> Optional[torch.Tensor]:
    """Mean-pooled last-layer hidden state of the uncompressed model while
    teacher-forced through [full_ids + ystar], at the ystar positions.

    Index-aligned with get_comp_log_probs' standard path: position
    n_prompt-1 is the state used to predict ystar[0] (context = prompt only);
    position n_prompt+t-1 is the state after feeding ystar[t-1].

    Returns (hidden_dim,) float32 on CPU, or None on OOM.
    """
    prompt_ids = full_ids
    if prompt_ids.shape[1] > max_seq_full:
        prompt_ids = prompt_ids[:, -max_seq_full:]
    full_input = torch.cat([prompt_ids.to(device), ystar.unsqueeze(0).to(device)], dim=1)
    n_prompt = prompt_ids.shape[1]
    n_gen = ystar.shape[0]

    try:
        out = model(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input),
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1][0, n_prompt - 1 : -1, :].float()  # (n_gen, dim)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    if hidden.shape[0] != n_gen:
        return None
    return hidden.mean(dim=0).cpu()


@torch.inference_mode()
def get_comp_hidden_state(
    model,
    comp_ids: torch.Tensor,
    ystar: torch.Tensor,
    pcfg: Optional[PrunedLlamaConfig],
    device: str,
    press=None,
) -> Optional[torch.Tensor]:
    """Mean-pooled last-layer hidden state of the compressed model while
    teacher-forced through [comp_ids + ystar], at the ystar positions.

    Mirrors get_comp_log_probs exactly (same step-decode loop for press
    methods, same single-pass slicing otherwise) so the two are index-aligned.

    Returns (hidden_dim,) float32 on CPU, or None on OOM/shape mismatch.
    """
    n_gen = ystar.shape[0]

    if press is not None:
        comp_input = comp_ids.to(device)
        try:
            with press(model):
                out = model(
                    input_ids=comp_input,
                    attention_mask=torch.ones_like(comp_input),
                    use_cache=True,
                    output_hidden_states=True,
                )
            kv = out.past_key_values
            hidden_list = [out.hidden_states[-1][0, -1, :].float()]
            del out

            for t in range(n_gen - 1):
                tok = ystar[t : t + 1].unsqueeze(0).to(device)
                step_out = model(
                    input_ids=tok, past_key_values=kv, use_cache=True,
                    output_hidden_states=True,
                )
                hidden_list.append(step_out.hidden_states[-1][0, -1, :].float())
                kv = step_out.past_key_values
            del kv

            if not hidden_list:
                return None
            hidden = torch.stack(hidden_list, dim=0)  # (n_gen, dim)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

        return hidden.mean(dim=0).cpu()

    # --- Standard path: single forward pass over [comp_ids + ystar] ---
    full_input = torch.cat([comp_ids.to(device), ystar.unsqueeze(0).to(device)], dim=1)
    n_prompt_comp = comp_ids.shape[1]

    originals = None
    if pcfg is not None:
        originals = patch_model(model, pcfg, device)

    try:
        out = model(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input),
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1][0, n_prompt_comp - 1 : -1, :].float()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        hidden = None
    finally:
        if originals is not None:
            unpatch_model(model, originals)

    if hidden is None or hidden.shape[0] != n_gen:
        return None

    return hidden.mean(dim=0).cpu()


@torch.inference_mode()
def embed_text(model, token_ids: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    """Mean-pooled last-layer hidden state of `model` reading `token_ids` in
    isolation (no compression, no preceding context) — used as a neutral
    embedding of free-generated text for semantic similarity comparison.

    token_ids: 1-D int64 tensor. Returns (hidden_dim,) float32 on CPU, or None.
    """
    if token_ids.numel() == 0:
        return None
    ids = token_ids.unsqueeze(0).to(device)
    try:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     output_hidden_states=True)
        hidden = out.hidden_states[-1][0].float()  # (seq_len, dim)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return hidden.mean(dim=0).cpu()


def cosine_sim(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None:
        return None
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def kl_divergence(log_p_full: torch.Tensor, log_p_comp: torch.Tensor) -> float:
    """Mean KL(P_full || P_comp) over generation steps, in nats."""
    p_full = log_p_full.exp()
    kl_steps = (p_full * (log_p_full - log_p_comp)).sum(dim=-1)  # (n_gen,)
    return kl_steps.mean().item()

# ---------------------------------------------------------------------------
# y* cache helpers
# ---------------------------------------------------------------------------

def load_ystar_cache(path: Path) -> dict:
    """Load y* cache from a .pt file. Returns {} if not found."""
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return {}


def save_ystar_cache(cache: dict, path: Path):
    """Atomically save y* cache to a .pt file."""
    tmp = path.with_suffix(".tmp.pt")
    torch.save(cache, tmp)
    tmp.replace(path)


def cache_key(task: str, idx: int) -> str:
    return f"{task}|{idx}"


def get_from_cache(cache: dict, task: str, idx: int):
    """Return (gen_tokens, log_p_full) from cache, or (None, None) if missing."""
    entry = cache.get(cache_key(task, idx))
    if entry is None:
        return None, None
    tokens     = entry["tokens"]                          # int64 (n_gen,)
    log_p_full = entry["log_p_full"].float()              # float16 → float32
    return tokens, log_p_full


def put_in_cache(cache: dict, task: str, idx: int,
                 gen_tokens: torch.Tensor, log_p_full: torch.Tensor):
    """Store y* in cache, compressing log_p_full to float16."""
    cache[cache_key(task, idx)] = {
        "tokens":     gen_tokens.cpu(),
        "log_p_full": log_p_full.cpu().half(),   # float32 → float16 to save space
    }

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    tasks: List[str],
    methods: List[str],
    data_dir: Path,
    output_path: Path,
    max_examples: int,
    device: str,
    max_new_tokens_override: Optional[int] = None,
    ystar_cache_path: Optional[Path] = None,
    do_timing: bool = False,
    salience_table: Optional[dict] = None,
):
    ckpt  = load_ckpt(output_path)
    cache = load_ystar_cache(ystar_cache_path) if ystar_cache_path else {}
    if ystar_cache_path:
        print(f"y* cache: {ystar_cache_path}  ({len(cache)} entries loaded)")

    n_tasks   = len(tasks)
    n_methods = len(methods)
    total_work = n_tasks * max_examples * n_methods
    completed_at_start = sum(
        1 for k, v in ckpt.items()
        if isinstance(v, float) and k not in ("_summary",)
    )

    print(f"\n{'='*72}")
    print(f"KL faithfulness (y* shared prefix)")
    print(f"  Tasks  : {n_tasks}  ({', '.join(tasks[:4])}{'...' if n_tasks > 4 else ''})")
    print(f"  Methods: {n_methods}  ({', '.join(methods)})")
    print(f"  N/task : {max_examples}")
    print(f"  Total  : {total_work} scores  ({completed_at_start} already done)")
    print(f"{'='*72}\n")

    grand_done  = completed_at_start
    grand_start = time.time()

    for t_idx, task in enumerate(tasks):
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file")
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template  = DATASET2PROMPT.get(task, "{context}{input}")
        max_new   = max_new_tokens_override or DATASET2MAXLEN.get(task, 64)
        max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex      = min(max_examples, len(dataset))

        task_start = time.time()
        task_done  = 0

        print(f"[{task}  {t_idx+1}/{n_tasks}]")

        for idx in range(n_ex):
            # Skip if all methods already done for this example
            if all(f"{task}|{m}|{idx}" in ckpt for m in methods):
                grand_done += n_methods
                task_done  += n_methods
                continue

            ex = dataset[idx]
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=question_text,
            )

            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

            # --- Step 1: generate y* and P_full (once per example, cache if possible) ---
            ystar, log_p_full = get_from_cache(cache, task, idx)
            if ystar is None:
                ystar, log_p_full, full_timing = generate_ystar(
                    model, full_ids, max_new, max_seq_full, device,
                    stop_token_ids=YSTAR_STOP_TOKENS.get(task),
                )
                if ystar is not None and ystar_cache_path:
                    put_in_cache(cache, task, idx, ystar, log_p_full)
                    save_ystar_cache(cache, ystar_cache_path)
                    ttft_str = f"  ttft={full_timing['ttft_ms']:.0f}ms  tpt={full_timing['tpt_ms']:.1f}ms" if full_timing else ""
                    print(f"  [cached y* for {task}|{idx}{ttft_str}]", flush=True)
            else:
                print(f"  [y* cache hit: {task}|{idx}  n_gen={ystar.shape[0]}]", flush=True)

            if ystar is None:
                for m in methods:
                    ckpt.setdefault(f"{task}|{m}|{idx}", None)
                save_ckpt(ckpt, output_path)
                grand_done += n_methods
                task_done  += n_methods
                continue

            n_gen = ystar.shape[0]

            # --- Step 2: for each method, teacher-force compressed model on y* ---
            for method in methods:
                ck_key = f"{task}|{method}|{idx}"
                if ck_key in ckpt:
                    grand_done += 1
                    task_done  += 1
                    continue

                pcfg  = METHOD_CONFIGS.get(method)
                press = _KVPRESS_METHODS.get(method)

                # kvpress methods process the full context internally; use max_seq_full
                effective_max = max_seq_full if press is not None else max_seq_comp

                if do_timing:
                    torch.cuda.synchronize()
                t_sel    = time.perf_counter()
                comp_ids = make_comp_ids(full_ids, method, tokenizer, question_text, effective_max,
                                         model=model, device=device,
                                         salience_table=salience_table)
                if do_timing:
                    torch.cuda.synchronize()
                sel_ms = (time.perf_counter() - t_sel) * 1000

                log_p_comp = get_comp_log_probs(model, comp_ids, ystar, pcfg, device, press=press)

                if log_p_comp is not None:
                    kl = kl_divergence(log_p_full, log_p_comp)
                    ckpt[ck_key] = kl
                else:
                    ckpt[ck_key] = None
                    kl = None

                if do_timing:
                    t_key = f"T|{ck_key}"
                    if t_key not in ckpt:
                        timed           = time_comp_prefill_decode(model, comp_ids, ystar, pcfg, device,
                                                                    press=press)
                        timed["sel_ms"] = sel_ms
                        ckpt[t_key]     = timed

                save_ckpt(ckpt, output_path)
                grand_done += 1
                task_done  += 1

                elapsed = time.time() - grand_start
                grand_eta = fmt_eta(grand_done - completed_at_start,
                                    total_work - completed_at_start, elapsed)
                kl_str = f"{kl:.4f}" if kl is not None else " OOM"
                print(
                    f"  ex {idx+1:>4}/{n_ex}  {method:<22}"
                    f"  n_gen={n_gen}  KL={kl_str}"
                    f"  [{grand_eta} total]",
                    flush=True,
                )

        task_elapsed = time.time() - task_start
        print(f"\n  --- {task} summary ---")
        per_method = {}
        for m in methods:
            scores = [ckpt[f"{task}|{m}|{i}"] for i in range(n_ex)
                      if isinstance(ckpt.get(f"{task}|{m}|{i}"), float)]
            mean = np.mean(scores) if scores else float("nan")
            per_method[m] = mean
            print(f"  {m:<24}  n={len(scores)}  mean_KL={mean:.4f}")

        if do_timing:
            print(f"\n  --- {task} timing (ms) ---")
            print(f"  {'method':<24}  {'sel_ms':>8}  {'ttft_ms':>9}  {'total_ttft':>11}  {'tpt_ms':>8}")
            for m in methods:
                t_vals = [ckpt[f"T|{task}|{m}|{i}"]
                          for i in range(n_ex)
                          if isinstance(ckpt.get(f"T|{task}|{m}|{i}"), dict)]
                if not t_vals:
                    continue
                def _avg(key):
                    vals = [v[key] for v in t_vals if v.get(key) is not None]
                    return sum(vals) / len(vals) if vals else float("nan")
                sel  = _avg("sel_ms");  ttft = _avg("ttft_ms");  tpt = _avg("tpt_ms")
                tot  = sel + ttft
                print(f"  {m:<24}  {sel:>8.1f}  {ttft:>9.1f}  {tot:>11.1f}  {tpt:>8.2f}")

        print(f"  [{fmt_duration(task_elapsed)}]\n")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"SUMMARY — KL(P_full || P_comp) with y* shared prefix (nats)")
    print(f"{'='*72}")

    col_w = 12
    header = f"{'Task':<26}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    per_method_all: Dict[str, List[float]] = {m: [] for m in methods}
    for task in tasks:
        row_vals = {}
        for m in methods:
            scores = [ckpt[f"{task}|{m}|{i}"] for i in range(max_examples)
                      if isinstance(ckpt.get(f"{task}|{m}|{i}"), float)]
            mean = np.mean(scores) if scores else float("nan")
            row_vals[m] = mean
            if not math.isnan(mean):
                per_method_all[m].append(mean)
        row = f"{task:<26}  " + "  ".join(
            f"{row_vals[m]:>{col_w}.4f}" if not math.isnan(row_vals[m]) else f"{'—':>{col_w}}"
            for m in methods
        )
        print(row)

    print("-" * len(header))
    avg = f"{'AVERAGE':<26}  " + "  ".join(
        f"{np.mean(per_method_all[m]):>{col_w}.4f}" if per_method_all[m] else f"{'—':>{col_w}}"
        for m in methods
    )
    print(avg)

    if do_timing:
        print(f"\n{'='*72}")
        print(f"TIMING SUMMARY (ms) — sel_ms=scoring pass, ttft_ms=compressed prefill, tpt_ms=per-token decode")
        print(f"{'='*72}")
        th = f"{'Method':<26}  {'sel_ms':>8}  {'ttft_ms':>9}  {'total_ttft':>11}  {'tpt_ms':>8}"
        print(th); print("-" * len(th))
        for m in methods:
            t_vals = [ckpt[f"T|{task}|{m}|{i}"]
                      for task in tasks
                      for i in range(max_examples)
                      if isinstance(ckpt.get(f"T|{task}|{m}|{i}"), dict)]
            if not t_vals:
                continue
            def _avg(key):
                vals = [v[key] for v in t_vals if v.get(key) is not None]
                return sum(vals) / len(vals) if vals else float("nan")
            sel  = _avg("sel_ms");  ttft = _avg("ttft_ms");  tpt = _avg("tpt_ms")
            tot  = sel + ttft
            print(f"{m:<26}  {sel:>8.1f}  {ttft:>9.1f}  {tot:>11.1f}  {tpt:>8.2f}")

    total_elapsed = time.time() - grand_start
    print(f"\nTotal wall time: {fmt_duration(total_elapsed)}")
    print(f"Results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, file_path: Path):
        self._file   = open(file_path, "a")
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg); self._file.write(msg); self._file.flush()
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def fileno(self):
        return self._stdout.fileno()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir",       default="lb_data_raw/data")
    p.add_argument("--output",         default="lb_results_base/kl_ystar.json")
    p.add_argument("--log",            default=None)
    p.add_argument("--tasks",          default=",".join(ENGLISH_TASKS))
    p.add_argument("--methods",        default="naive_65pct,kq_post_rope,snapkv,phrase_w64_t25")
    p.add_argument("--n",              type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--ystar_cache",    default="lb_results_base/ystar_cache.pt",
                   help="Path to cache y* tokens and full-model logits across runs")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--timing",         action="store_true",
                   help="Measure TTFT and TPT for each method (adds ~20-30%% overhead)")
    p.add_argument("--salience_table", default=None,
                   help="Path to token salience .pt file (from calibrate_token_salience.py)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    tasks   = [t.strip() for t in args.tasks.split(",")   if t.strip() in ENGLISH_TASKS]
    _all_known = set(METHOD_CONFIGS) | set(_SELECT_METHODS) | set(CHUNK_CONFIGS) | set(_KVPRESS_METHODS) | set(_SALIENCE_METHODS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in _all_known]

    if not tasks:
        print("No valid tasks."); return
    if not methods:
        print("No valid methods."); return

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    sal_table = None
    if args.salience_table:
        sal_table = load_salience_table(args.salience_table)

    run_eval(
        model, tokenizer,
        tasks=tasks,
        methods=methods,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        max_examples=args.n,
        device=args.device,
        max_new_tokens_override=args.max_new_tokens,
        ystar_cache_path=Path(args.ystar_cache) if args.ystar_cache else None,
        do_timing=args.timing,
        salience_table=sal_table,
    )


if __name__ == "__main__":
    main()

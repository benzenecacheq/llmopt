"""
kl_faith_eval.py
----------------
Full KL divergence faithfulness evaluation across all LongBench tasks.

For each method × task × example:
  1. Generate from compressed model (method config), capturing per-step logit
     distributions via output_scores=True.
  2. Run full model teacher-forced on [full_prompt + compressed_output_tokens].
  3. Compute mean KL(P_full || P_compressed) over generation steps.

Methods: kq_post_rope (RADAR), kq_only, naive_65pct, snapkv, streaming, vn_decay, pruned

Checkpointing: saves after every example; resume is automatic.
Progress: per-example line with running mean and ETA.

Usage:
    python kl_faith_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --data_dir lb_data_raw/data \\
        --output lb_results_base/kl_faithfulness.json

    # Resume (just re-run the same command):
    python kl_faith_eval.py --output lb_results_base/kl_faithfulness.json

    # Subset of tasks/methods for testing:
    python kl_faith_eval.py --tasks samsum,hotpotqa --methods kq_post_rope,streaming --n 10
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_pruned import (
    HeadFilterConfig,
    PerHeadFilter,
    PrunedLlamaAttention,
    PrunedLlamaConfig,
)

# ---------------------------------------------------------------------------
# Task / data definitions (mirrors faithfulness_deep.py)
# ---------------------------------------------------------------------------

ENGLISH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and "
        "do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary: "
        "The following is a summary of the above news articles:"
    ),
    "trec": (
        "Please determine the type of the question below. "
        "Here are some examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. "
        "The following are some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc":         ("Please complete the code given below.\n{context}Next line of code:\n"),
    "repobench-p": ("Please complete the code given below.\n{context}{input}Next line of code:\n"),
}

DATASET2MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
    "2wikimqa": 32, "musique": 32, "gov_report": 512, "qmsum": 512,
    "multi_news": 512, "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64,
}

# The compressed prefill uses a manual matmul (T × budget × n_heads × fp32) that
# can OOM at long contexts.  Cap compressed prompts conservatively.
# The full-model teacher-forced forward uses SDPA and can handle longer inputs.
DEFAULT_MAX_SEQ_COMP = 4096
DEFAULT_MAX_SEQ_FULL = 7168

TASK_MAX_SEQ_COMP: Dict[str, int] = {}   # override per task if needed
TASK_MAX_SEQ_FULL: Dict[str, int] = {
    "narrativeqa": 5120,
    "gov_report":  6144,
    "qmsum":       6144,
    "multi_news":  6144,
}

LOW_SIGNAL_THRESHOLD = 10.0

# ---------------------------------------------------------------------------
# Method configs
# ---------------------------------------------------------------------------

BASE_CFG = dict(
    budget_fraction=0.65,
    min_decay=0.7,
    always_keep_first=16,
    always_keep_last=16,
    q_buffer_size=128,
)

METHOD_CONFIGS: Dict[str, Optional[PrunedLlamaConfig]] = {
    "kq_post_rope": PrunedLlamaConfig(score_mode="kq_post_rope", **BASE_CFG),
    "kq_only":      PrunedLlamaConfig(score_mode="kq_only",      **BASE_CFG),
    "snapkv":       PrunedLlamaConfig(score_mode="snapkv",       **BASE_CFG),
    "streaming":    PrunedLlamaConfig(score_mode="streaming",    **BASE_CFG),
    "naive_65pct":  None,   # prompt truncation — no KV patching
    "naive_tail":   None,   # prompt truncation, pure recency tail (no head)
    "vn_decay":     PrunedLlamaConfig(score_mode="vn_decay",     **BASE_CFG),
    "pruned":       PrunedLlamaConfig(score_mode="additive", score_alpha=0.65, **BASE_CFG),
}

DEFAULT_METHODS = list(METHOD_CONFIGS)

# Exponential-decay screening configs — not included in DEFAULT_METHODS
_SCREEN_BASE = dict(
    budget_fraction=0.65,
    always_keep_first=16,
    always_keep_last=16,
    q_buffer_size=128,
    decay_fn="exponential",
)
for _md_str, _md_val in [("d070", 0.70), ("d080", 0.80), ("d090", 0.90), ("d095", 0.95)]:
    METHOD_CONFIGS[f"kq_pre_{_md_str}"]   = PrunedLlamaConfig(score_mode="kq_pre_rope",  min_decay=_md_val, **_SCREEN_BASE)
    METHOD_CONFIGS[f"kq_post_{_md_str}"]  = PrunedLlamaConfig(score_mode="kq_post_rope", min_decay=_md_val, **_SCREEN_BASE)
    METHOD_CONFIGS[f"snapkv_{_md_str}"]   = PrunedLlamaConfig(score_mode="snapkv_decay", min_decay=_md_val, **_SCREEN_BASE)

# Naive-tail + content-scored head experiments.
# tail_budget_frac=0.90 forces the most-recent 90% of middle-budget tokens to always be kept
# (matching naive_65pct's tail/head split of ~0.90/0.10); content scoring then picks the best
# remaining old tokens to fill the head portion.
_NAIVE_SCORED_BASE = dict(
    budget_fraction=0.65,
    always_keep_first=16,   # sink guard — prevents empty causal attention for early queries
    always_keep_last=16,
    q_buffer_size=128,
    tail_budget_frac=0.90,
)
METHOD_CONFIGS["naive_best_pre"]  = PrunedLlamaConfig(score_mode="kq_only",      **_NAIVE_SCORED_BASE)
METHOD_CONFIGS["naive_best_post"] = PrunedLlamaConfig(score_mode="kq_post_rope", min_decay=1.0, **_NAIVE_SCORED_BASE)

# Chunk-level prompt construction variants — all use None (no KV patching).
# Dispatch is via CHUNK_CONFIGS keyed by method name.
for _cm in ("chunk_best", "chunk_word", "chunk_bm25", "chunk_32", "chunk_128", "chunk_word128", "chunk_sent",
            "phrase_w32_t0", "phrase_w64_t0", "phrase_w32_t50", "phrase_w64_t50",
            "phrase_w64_t25", "phrase_w64_t75"):
    METHOD_CONFIGS[_cm] = None


# Structural control: same token selection as naive_tail (last 65%) but via KV-cache pruning.
# Tests whether the prompt-truncation structural advantage is real — if this is worse than
# naive_tail despite identical token selection, the corruption is in the pruning mechanism itself.
METHOD_CONFIGS["naive_stream"] = PrunedLlamaConfig(
    score_mode="streaming", sink_size=1,
    budget_fraction=0.65, always_keep_first=0, always_keep_last=0,
    q_buffer_size=128,
)

# ---------------------------------------------------------------------------
# Model patching helpers
# ---------------------------------------------------------------------------

def patch_model(model, pcfg: PrunedLlamaConfig, device: str):
    """Swap LlamaAttention → PrunedLlamaAttention; return originals for restore."""
    cfg = model.config
    filter_cfg = HeadFilterConfig(
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        total_budget=pcfg.total_budget,
        q_buffer_size=pcfg.q_buffer_size,
        budget_strategy="equal",
        decay_fn=pcfg.decay_fn,
        decay_rate=pcfg.decay_rate,
        always_keep_last=pcfg.always_keep_last,
    )
    shared_filter = PerHeadFilter(filter_cfg, device=device)
    originals = []
    for layer_idx, layer in enumerate(model.model.layers):
        originals.append(layer.self_attn)
        pa = PrunedLlamaAttention(layer.self_attn, pcfg, layer_idx, torch.device(device))
        pa.head_filter = shared_filter
        layer.self_attn = pa
    return originals


def unpatch_model(model, originals):
    for layer, orig in zip(model.model.layers, originals):
        layer.self_attn = orig


# ---------------------------------------------------------------------------
# Naive truncation
# ---------------------------------------------------------------------------

def naive_truncate(input_ids: torch.Tensor, fraction: float = 0.65,
                   head_frac: float = 0.10) -> torch.Tensor:
    """Truncate to `fraction` of length with a head/tail split."""
    T = input_ids.shape[1]
    budget = max(1, int(T * fraction))
    head_n = max(0, int(budget * head_frac))
    tail_n = budget - head_n
    head = input_ids[:, :head_n]
    tail = input_ids[:, -tail_n:] if tail_n > 0 else input_ids[:, :0]
    return torch.cat([head, tail], dim=1)


def _chunk_scores_token(chunks, q_tokens):
    """Raw subword-token overlap with question token set."""
    return [len(set(c[0].tolist()) & q_tokens) for c in chunks]


def _chunk_scores_word(chunks, question_ids, tokenizer):
    """Word-level overlap: detokenize both sides, split on whitespace, lowercase."""
    import re
    q_text  = tokenizer.decode(question_ids[0].tolist(), skip_special_tokens=True)
    q_words = set(re.findall(r"\b\w+\b", q_text.lower()))
    scores  = []
    for c in chunks:
        c_text  = tokenizer.decode(c[0].tolist(), skip_special_tokens=True)
        c_words = set(re.findall(r"\b\w+\b", c_text.lower()))
        scores.append(len(c_words & q_words))
    return scores


def _chunk_scores_bm25(chunks, question_ids, k1: float = 1.5, b: float = 0.75):
    """BM25 at the subword-token level: IDF-weighted TF with length normalisation."""
    from collections import Counter
    N          = len(chunks)
    chunk_toks = [c[0].tolist() for c in chunks]
    chunk_tfs  = [Counter(t) for t in chunk_toks]
    chunk_lens = [len(t) for t in chunk_toks]
    avgdl      = sum(chunk_lens) / N if N else 1.0

    df: Counter = Counter()
    for tf in chunk_tfs:
        for tok in tf:
            df[tok] += 1

    q_tokens = list(dict.fromkeys(question_ids[0].tolist()))  # unique, order-preserving
    scores   = []
    for tf, dl in zip(chunk_tfs, chunk_lens):
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf    = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            tf_val = tf[t]
            s     += idf * tf_val * (k1 + 1) / (tf_val + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def chunk_truncate(
    input_ids: torch.Tensor,
    question_ids: torch.Tensor,
    tokenizer=None,
    fraction: float = 0.65,
    chunk_size: int = 64,
    tail_frac: float = 0.0,
    scoring: str = "token",
) -> torch.Tensor:
    """
    Prompt-construction truncation via scored chunk selection.

    tail_frac controls what fraction of the budget is reserved as a fixed recency
    tail (identical to naive truncation).  The remaining budget is filled by the
    top-scoring chunks from the rest of the document.  tail_frac=0.0 (default)
    scores the entire document; ties broken by recency so zero-score chunks
    default to the most recent positions, matching naive behaviour.

    scoring options:
      "token" — raw subword-token overlap (original)
      "word"  — word-level overlap after detokenization (avoids BPE fragmentation)
      "bm25"  — BM25 at subword level (IDF + TF normalisation)

    Selected chunks are re-inserted in original order. No KV patching.
    """
    T = input_ids.shape[1]
    budget     = max(1, int(T * fraction))
    tail_n     = min(T, max(0, int(budget * tail_frac)))
    head_budget = budget - tail_n

    tail = input_ids[:, T - tail_n :] if tail_n > 0 else input_ids[:, :0]

    if head_budget == 0 or T - tail_n <= 0:
        return tail

    middle_end = T - tail_n
    starts  = list(range(0, middle_end, chunk_size))
    chunks  = [input_ids[:, s : min(s + chunk_size, middle_end)] for s in starts]
    has_q   = question_ids is not None and question_ids.shape[1] > 0

    if not has_q:
        scores = [0] * len(chunks)
    elif scoring == "word" and tokenizer is not None:
        scores = _chunk_scores_word(chunks, question_ids, tokenizer)
    elif scoring == "bm25":
        scores = _chunk_scores_bm25(chunks, question_ids)
    else:  # "token" (default)
        q_tokens = set(question_ids[0].tolist())
        scores   = _chunk_scores_token(chunks, q_tokens)

    # Ties broken by descending index (recency) so zero-score chunks default to tail.
    order    = sorted(range(len(chunks)), key=lambda i: (-scores[i], -i))
    selected: list = []
    remaining = head_budget
    for ci in order:
        if remaining <= 0:
            break
        chunk = chunks[ci]
        take  = min(chunk.shape[1], remaining)
        selected.append((ci, chunk[:, :take]))
        remaining -= take

    selected.sort(key=lambda x: x[0])
    parts = [c for _, c in selected]
    if tail_n > 0:
        parts.append(tail)
    return torch.cat(parts, dim=1)


# Maps chunk method names → chunk_truncate kwargs (all are None in METHOD_CONFIGS)
CHUNK_CONFIGS: Dict[str, dict] = {
    "chunk_best":    dict(chunk_size=64,  scoring="token"),
    "chunk_word":    dict(chunk_size=64,  scoring="word"),
    "chunk_bm25":    dict(chunk_size=64,  scoring="bm25"),
    "chunk_32":      dict(chunk_size=32,  scoring="token"),
    "chunk_128":     dict(chunk_size=128, scoring="token"),
    "chunk_word128": dict(chunk_size=128, scoring="word"),
    # Phrase selection sweep: word scoring × {32,64} chunk size × {0.0,0.5} tail_frac
    "phrase_w32_t0":  dict(chunk_size=32, scoring="word", tail_frac=0.0),
    "phrase_w64_t0":  dict(chunk_size=64, scoring="word", tail_frac=0.0),
    "phrase_w32_t50": dict(chunk_size=32, scoring="word", tail_frac=0.5),
    "phrase_w64_t50": dict(chunk_size=64, scoring="word", tail_frac=0.5),
    # Tail-fraction sweep: w64 × {0.25, 0.75}
    "phrase_w64_t25": dict(chunk_size=64, scoring="word", tail_frac=0.25),
    "phrase_w64_t75": dict(chunk_size=64, scoring="word", tail_frac=0.75),
}

# Multi-compression sweep: naive, RADAR (kq_post_rope), SnapKV, phrase_w64_t25 at 50/40/35%
for _frac in (0.50, 0.40, 0.35):
    _fpct = int(_frac * 100)
    METHOD_CONFIGS[f"naive_{_fpct}pct"]        = None
    METHOD_CONFIGS[f"radar_{_fpct}pct"]        = PrunedLlamaConfig(score_mode="kq_post_rope",
                                                    budget_fraction=_frac, min_decay=0.7,
                                                    always_keep_first=16, always_keep_last=16,
                                                    q_buffer_size=128)
    METHOD_CONFIGS[f"snapkv_{_fpct}pct"]       = PrunedLlamaConfig(score_mode="snapkv",
                                                    budget_fraction=_frac, min_decay=0.7,
                                                    always_keep_first=16, always_keep_last=16,
                                                    q_buffer_size=128)
    CHUNK_CONFIGS[f"phrase_w64_t25_f{_fpct}"]  = dict(chunk_size=64, scoring="word",
                                                       tail_frac=0.25, fraction=_frac)
    METHOD_CONFIGS[f"phrase_w64_t25_f{_fpct}"] = None

# *_select variants: use KV-pruning scoring criteria for token selection, but do a
# clean input-truncation prefill instead of pruning the KV cache.  Separates scoring
# quality from the structural corruption caused by the pruning mechanism itself.
for _sel in ("snapkv_select", "radar_select"):
    METHOD_CONFIGS[_sel] = None
    for _frac in (0.50, 0.40, 0.35):
        METHOD_CONFIGS[f"{_sel}_f{int(_frac*100)}"] = None


def sent_truncate(
    input_ids: torch.Tensor,
    question_ids: torch.Tensor,
    tokenizer,
    fraction: float = 0.65,
    tail_frac: float = 0.0,
) -> torch.Tensor:
    """
    Sentence/paragraph-boundary chunk pruning.

    Detokenizes the pre-tail region, splits on paragraph/sentence boundaries,
    re-tokenizes each segment, scores by word-level question overlap, and greedily
    fills the head budget with top-scoring segments in their original order.

    Avoids the fixed-size artifact of chunk_truncate — each unit is a complete thought.
    """
    import re

    T           = input_ids.shape[1]
    budget      = max(1, int(T * fraction))
    tail_n      = min(T, max(0, int(budget * tail_frac)))
    head_budget = budget - tail_n

    tail = input_ids[:, T - tail_n :] if tail_n > 0 else input_ids[:, :0]

    if head_budget == 0 or T - tail_n <= 0:
        return tail

    middle_end  = T - tail_n
    middle_text = tokenizer.decode(input_ids[0, :middle_end].tolist(), skip_special_tokens=True)

    # Split on paragraph breaks first; fall back to sentence endings within long paragraphs.
    raw_segs = re.split(r"\n+", middle_text)
    segs: list = []
    for seg in raw_segs:
        seg = seg.strip()
        if not seg:
            continue
        # If a paragraph is very long, split further on sentence boundaries.
        toks = tokenizer.encode(seg, add_special_tokens=False)
        if len(toks) > 200:
            sents = re.split(r"(?<=[.!?])\s+", seg)
            segs.extend(s.strip() for s in sents if s.strip())
        else:
            segs.append(seg)

    if not segs:
        return tail

    seg_ids = [
        tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")
        for s in segs
    ]
    seg_ids = [s for s in seg_ids if s.shape[1] > 0]
    if not seg_ids:
        return tail

    # Word-level overlap scoring against the question.
    has_q   = question_ids is not None and question_ids.shape[1] > 0
    q_words = set()
    if has_q:
        q_text  = tokenizer.decode(question_ids[0].tolist(), skip_special_tokens=True)
        q_words = set(re.findall(r"\b\w+\b", q_text.lower()))

    scores = []
    for s in seg_ids:
        if q_words:
            s_text  = tokenizer.decode(s[0].tolist(), skip_special_tokens=True)
            s_words = set(re.findall(r"\b\w+\b", s_text.lower()))
            scores.append(len(s_words & q_words))
        else:
            scores.append(0)

    # Ties broken by descending index (recency) so zero-score segments default to tail.
    order     = sorted(range(len(seg_ids)), key=lambda i: (-scores[i], -i))
    selected: list = []
    remaining = head_budget
    for ci in order:
        if remaining <= 0:
            break
        seg  = seg_ids[ci]
        take = min(seg.shape[1], remaining)
        selected.append((ci, seg[:, :take]))
        remaining -= take

    # Restore original document order.
    selected.sort(key=lambda x: x[0])
    parts = [c for _, c in selected]
    if tail_n > 0:
        parts.append(tail)
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Per-example KL computation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def kl_for_example(
    model,
    tokenizer,
    prompt: str,
    method: str,
    pcfg: Optional[PrunedLlamaConfig],
    max_new_tokens: int,
    max_seq_comp: int,
    max_seq_full: int,
    device: str,
    question_text: str = "",
) -> Optional[float]:
    """
    Returns mean KL(P_full || P_compressed) in nats, or None on OOM/empty gen.
    """
    full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

    # Prompt for the compressed model
    if method == "naive_65pct":
        comp_ids = naive_truncate(full_ids, head_frac=0.10)
    elif method.startswith("naive_") and method.endswith("pct") and method != "naive_tail":
        _frac = int(method[6:-3]) / 100.0
        comp_ids = naive_truncate(full_ids, fraction=_frac, head_frac=0.10)
    elif method == "naive_tail":
        comp_ids = naive_truncate(full_ids, head_frac=0.0)
    elif method == "chunk_sent":
        q_ids    = tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt") if question_text else None
        comp_ids = sent_truncate(full_ids, q_ids, tokenizer=tokenizer)
    elif method in CHUNK_CONFIGS:
        q_ids    = tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt") if question_text else None
        comp_ids = chunk_truncate(full_ids, q_ids, tokenizer=tokenizer, **CHUNK_CONFIGS[method])
    else:
        comp_ids = full_ids.clone()

    # Cap compressed model input to avoid OOM in manual attention matmul
    if comp_ids.shape[1] > max_seq_comp:
        comp_ids = comp_ids[:, -max_seq_comp:]

    comp_ids = comp_ids.to(device)
    n_prompt_comp = comp_ids.shape[1]

    # --- Step 1: compressed generation ---
    originals = None
    if pcfg is not None:
        originals = patch_model(model, pcfg, device)

    try:
        out = model.generate(
            input_ids=comp_ids,
            attention_mask=torch.ones_like(comp_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if originals is not None:
            unpatch_model(model, originals)
        return None
    finally:
        if originals is not None:
            unpatch_model(model, originals)

    # out.scores: tuple of (1, vocab) tensors
    if not out.scores:
        return None
    comp_logits = torch.stack([s[0] for s in out.scores], dim=0).cpu()  # (n_gen, vocab)
    gen_tokens  = out.sequences[0, n_prompt_comp:].cpu()                # (n_gen,)
    n_gen = gen_tokens.shape[0]
    if n_gen == 0:
        return None

    # Free KV cache and fragmented activations before the full-model forward
    del out
    torch.cuda.empty_cache()

    # --- Step 2: full model teacher-forced forward ---
    # Full model always sees the original (untruncated) prompt; SDPA handles longer seqs.
    full_prompt_ids = full_ids
    if full_prompt_ids.shape[1] > max_seq_full:
        full_prompt_ids = full_prompt_ids[:, -max_seq_full:]
    full_input = torch.cat([
        full_prompt_ids.to(device),
        gen_tokens.unsqueeze(0).to(device),
    ], dim=1)
    n_prompt_full = full_prompt_ids.shape[1]

    try:
        full_out = model(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    # Logits at [n_prompt_full-1 .. -1] predict tokens [n_prompt_full .. end]
    full_logits = full_out.logits[0, n_prompt_full - 1 : -1, :].float()  # (n_gen, vocab)
    if full_logits.shape[0] != n_gen:
        return None

    # --- Step 3: KL(P_full || P_compressed) ---
    log_p_full = F.log_softmax(full_logits,                        dim=-1)
    log_p_comp = F.log_softmax(comp_logits.to(device).float(),     dim=-1)
    p_full     = log_p_full.exp()
    kl_per_step = (p_full * (log_p_full - log_p_comp)).sum(dim=-1) # (n_gen,)
    return kl_per_step.mean().item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_ckpt(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_ckpt(data: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_eta(done: int, total: int, elapsed: float) -> str:
    if done == 0:
        return "eta=?"
    rate  = done / elapsed
    left  = (total - done) / rate
    return f"eta={fmt_duration(left)}"


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
    gt_results: Optional[dict],
    max_new_tokens_override: Optional[int] = None,
):
    ckpt = load_ckpt(output_path)

    n_tasks   = len(tasks)
    n_methods = len(methods)
    run_start = time.time()

    # Count total work for grand ETA
    total_work = n_tasks * n_methods * max_examples

    # Track completed work at start (for resume ETA)
    completed_at_start = sum(
        1 for k in ckpt
        if k not in ("_summary",) and isinstance(ckpt.get(k), (int, float))
    )

    print(f"\n{'='*72}")
    print(f"KL faithfulness evaluation")
    print(f"  Tasks  : {n_tasks}  ({', '.join(tasks[:4])}{'...' if n_tasks > 4 else ''})")
    print(f"  Methods: {n_methods}  ({', '.join(methods)})")
    print(f"  N/task : {max_examples}")
    print(f"  Total  : {total_work} examples  ({completed_at_start} already done)")
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

        template = DATASET2PROMPT.get(task, "{context}{input}")
        max_new     = max_new_tokens_override if max_new_tokens_override is not None else DATASET2MAXLEN.get(task, 64)
        max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex         = min(max_examples, len(dataset))
        gt_flag  = ""
        if gt_results:
            fg = gt_results.get(task, {}).get("full", 100.0)
            gt_flag = "†" if (isinstance(fg, float) and fg < LOW_SIGNAL_THRESHOLD) else " "

        for m_idx, method in enumerate(methods):
            pcfg = METHOD_CONFIGS.get(method)

            # Check if this task×method is already complete
            done_keys = [k for k in ckpt if k.startswith(f"{task}|{method}|")]
            if len(done_keys) >= n_ex:
                scores = [ckpt[k] for k in done_keys if isinstance(ckpt[k], float)]
                mean_kl = np.mean(scores) if scores else float("nan")
                print(
                    f"[{task:<22}{gt_flag} {t_idx+1:>2}/{n_tasks}] "
                    f"{method:<14} ({m_idx+1}/{n_methods})  "
                    f"SKIP — already done  n={len(scores)}  mean_KL={mean_kl:.4f}"
                )
                grand_done += n_ex
                continue

            scores_this: List[float] = [
                ckpt[f"{task}|{method}|{i}"]
                for i in range(n_ex)
                if f"{task}|{method}|{i}" in ckpt and isinstance(ckpt[f"{task}|{method}|{i}"], float)
            ]
            task_start = time.time()
            ex_done    = len(scores_this)

            header = (
                f"[{task:<22}{gt_flag} {t_idx+1:>2}/{n_tasks}] "
                f"{method:<14} ({m_idx+1}/{n_methods})"
            )

            for idx in range(n_ex):
                ck_key = f"{task}|{method}|{idx}"
                if ck_key in ckpt:
                    continue   # already computed in a previous run

                ex = dataset[idx] if idx < len(dataset) else None
                if ex is None:
                    continue

                question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
                prompt = template.format(
                    context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                    input=question_text,
                )

                kl = kl_for_example(
                    model, tokenizer, prompt, method, pcfg,
                    max_new, max_seq_comp, max_seq_full, device,
                    question_text=question_text,
                )

                if kl is not None:
                    scores_this.append(kl)
                    ckpt[ck_key] = kl
                else:
                    ckpt[ck_key] = None   # mark as attempted/skipped

                save_ckpt(ckpt, output_path)

                ex_done    += 1
                grand_done += 1
                task_elapsed = time.time() - task_start

                mean_so_far = np.mean(scores_this) if scores_this else float("nan")
                rate_ex     = ex_done / task_elapsed if task_elapsed > 0 else 0
                task_eta    = fmt_eta(ex_done, n_ex, task_elapsed)
                grand_eta   = fmt_eta(
                    grand_done - completed_at_start,
                    total_work - completed_at_start,
                    time.time() - grand_start,
                )

                kl_str = f"{kl:.4f}" if kl is not None else " OOM"
                print(
                    f"  {header}  ex {idx+1:>4}/{n_ex}"
                    f"  KL={kl_str}  mean={mean_so_far:.4f}"
                    f"  {rate_ex:.2f}ex/s  {task_eta}  [{grand_eta} total]",
                    flush=True,
                )

            # Task × method summary
            n_valid = len(scores_this)
            mean_kl = np.mean(scores_this) if scores_this else float("nan")
            elapsed = time.time() - task_start
            print(
                f"  >>> {task} / {method}:  n={n_valid}  mean_KL={mean_kl:.4f}"
                f"  [{fmt_duration(elapsed)}]\n",
                flush=True,
            )

    # ---------------------------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"SUMMARY — KL(P_full || P_compressed), mean over examples (nats)")
    print(f"  0 = identical distributions;  ln2≈0.693 = maximum JS divergence")
    print(f"{'='*72}")

    col_w = 12
    header_row = f"{'Task':<26}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(header_row)
    print("-" * len(header_row))

    per_method_all: Dict[str, List[float]] = {m: [] for m in methods}
    for task in tasks:
        gt_flag = ""
        if gt_results:
            fg = gt_results.get(task, {}).get("full", 100.0)
            gt_flag = "†" if (isinstance(fg, float) and fg < LOW_SIGNAL_THRESHOLD) else " "

        row_vals = {}
        for method in methods:
            scores = [
                ckpt[f"{task}|{method}|{i}"]
                for i in range(max_examples)
                if isinstance(ckpt.get(f"{task}|{method}|{i}"), float)
            ]
            mean_kl = np.mean(scores) if scores else float("nan")
            row_vals[method] = mean_kl
            if not math.isnan(mean_kl):
                per_method_all[method].append(mean_kl)

        row = f"{task:<24}{gt_flag}  " + "  ".join(
            f"{row_vals[m]:>{col_w}.4f}" if not math.isnan(row_vals[m]) else f"{'—':>{col_w}}"
            for m in methods
        )
        print(row)

    print("-" * len(header_row))
    avg_row = f"{'AVERAGE':<26}  " + "  ".join(
        f"{np.mean(per_method_all[m]):>{col_w}.4f}" if per_method_all[m] else f"{'—':>{col_w}}"
        for m in methods
    )
    print(avg_row)
    if gt_results:
        print("  † full-context ground-truth < 10%: model unreliable on this task")

    total_elapsed = time.time() - run_start
    print(f"\nTotal wall time: {fmt_duration(total_elapsed)}")
    print(f"Results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both stdout and a file, flushing both after every write."""
    def __init__(self, file_path: Path):
        self._file = open(file_path, "a")
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def fileno(self):
        return self._stdout.fileno()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir",    default="lb_data_raw/data")
    p.add_argument("--output",      default="lb_results_base/kl_faithfulness.json")
    p.add_argument("--gt_results",  default="lb_results_base/results.json")
    p.add_argument("--log",         default=None,
                   help="Also write all output to this file (appends)")
    p.add_argument("--tasks",       default=",".join(ENGLISH_TASKS))
    p.add_argument("--methods",     default=",".join(DEFAULT_METHODS))
    p.add_argument("--n",               type=int, default=100)
    p.add_argument("--max_new_tokens",  type=int, default=None,
                   help="Override per-task generation length (e.g. 1 for quick screening)")
    p.add_argument("--max_seq_comp",    type=int, default=None,
                   help="Override default max compressed sequence length (default 4096)")
    p.add_argument("--max_seq_full",    type=int, default=None,
                   help="Override default max full-model sequence length (default 7168)")
    p.add_argument("--device",          default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    tasks   = [t.strip() for t in args.tasks.split(",")   if t.strip() in ENGLISH_TASKS]
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHOD_CONFIGS]

    if not tasks:
        print("No valid tasks specified.")
        return
    if not methods:
        print("No valid methods specified.")
        return

    gt_results = None
    gt_path = Path(args.gt_results)
    if gt_path.exists():
        with open(gt_path) as f:
            gt_results = json.load(f)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    global DEFAULT_MAX_SEQ_COMP, DEFAULT_MAX_SEQ_FULL
    if args.max_seq_comp is not None:
        DEFAULT_MAX_SEQ_COMP = args.max_seq_comp
    if args.max_seq_full is not None:
        DEFAULT_MAX_SEQ_FULL = args.max_seq_full

    run_eval(
        model, tokenizer,
        tasks=tasks,
        methods=methods,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        max_examples=args.n,
        device=args.device,
        gt_results=gt_results,
        max_new_tokens_override=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phrase-level BPE extension.

Starting from Mistral's 32K subword tokenizer, performs additional BPE merges
at the token level using LongBench contexts as the training corpus.

Stopping criterion (MDL / Minimum Description Length):
  Merge pair (A, B) -> C only when frequency f > N / (V + 1),
  i.e., when the gain from fewer tokens outweighs the cost of a larger vocab.
  Derivation: merge is beneficial iff (N - f)(V + 1) < N * V  =>  f > N/(V+1).
"""

import json
import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DATA_DIR   = Path("lb_data_raw/data")
MAX_TOK_ID = 200_000   # headroom for new phrase tokens (pair code = a*MAX_TOK_ID + b)

TASK_FILES = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_and_tokenize(tokenizer):
    sequences = []
    for task in TASK_FILES:
        path = DATA_DIR / f"{task}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                ctx = ex.get("context", "")
                if ctx:
                    ids = tokenizer.encode(ctx, add_special_tokens=False)
                    if ids:
                        sequences.append(np.array(ids, dtype=np.int32))
    return sequences


# ---------------------------------------------------------------------------
# Pair counting (numpy, fast)
# ---------------------------------------------------------------------------

def count_pairs(sequences):
    """Return (unique_pair_codes, counts) sorted by code value."""
    chunks = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        codes = seq[:-1].astype(np.int64) * MAX_TOK_ID + seq[1:].astype(np.int64)
        chunks.append(codes)
    all_codes = np.concatenate(chunks)
    unique, counts = np.unique(all_codes, return_counts=True)
    return unique, counts


def best_pair(unique, counts):
    idx = np.argmax(counts)
    code = int(unique[idx])
    return code // MAX_TOK_ID, code % MAX_TOK_ID, int(counts[idx])


# ---------------------------------------------------------------------------
# Merge application
# ---------------------------------------------------------------------------

def apply_merge(sequences, a, b, new_id):
    """Replace all occurrences of adjacent (a, b) with new_id."""
    out = []
    for seq in sequences:
        if len(seq) < 2:
            out.append(seq)
            continue
        # Fast skip: numpy check before Python loop
        hit = (seq[:-1] == a) & (seq[1:] == b)
        if not hit.any():
            out.append(seq)
            continue
        # Python loop only for sequences that actually contain the pair
        arr = seq.tolist()
        new = []
        i = 0
        while i < len(arr):
            if i + 1 < len(arr) and arr[i] == a and arr[i + 1] == b:
                new.append(new_id)
                i += 2
            else:
                new.append(arr[i])
                i += 1
        out.append(np.array(new, dtype=np.int32))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LOG_PATH = Path("lb_results_base/phrase_bpe.log")


def log(msg, logfile):
    logfile.write(msg + "\n")
    logfile.flush()


def main():
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "w") as logfile:
        _run(logfile)


def _run(logfile):
    log("Loading Mistral tokenizer...", logfile)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    V = tokenizer.vocab_size   # 32 000

    log("Tokenizing LongBench contexts...", logfile)
    t0 = time.time()
    sequences = load_and_tokenize(tokenizer)
    N = sum(len(s) for s in sequences)
    N_init, V_init = N, V
    log(f"  {len(sequences):,} contexts  |  {N:,} tokens  |  {time.time()-t0:.1f}s", logfile)
    log(f"  Initial N×V = {N*V:,}  |  MDL threshold = {N/(V+1):.0f} occurrences\n", logfile)

    next_id = V
    merges  = []   # (a, b, new_id, freq, pair_str)

    for iteration in range(200_000):
        t0 = time.time()

        unique, counts = count_pairs(sequences)
        a, b, freq     = best_pair(unique, counts)
        threshold      = N / (V + 1)

        if freq <= threshold:
            log(f"\nMDL stop: best freq {freq:,} ≤ threshold {threshold:.0f}  "
                f"(iteration {iteration + 1})", logfile)
            break

        # Merge
        sequences = apply_merge(sequences, a, b, next_id)
        N -= freq
        V += 1

        try:
            pair_str = repr(tokenizer.decode([a]) + tokenizer.decode([b]))
        except Exception:
            pair_str = f"tok({a},{b})"

        merges.append((a, b, next_id, freq, pair_str))
        next_id += 1
        elapsed = time.time() - t0

        if (iteration < 20) or (iteration % 25 == 0):
            log(f"  [{iteration+1:5d}]  freq={freq:8,}  thresh={threshold:7.0f}"
                f"  {pair_str:<40}"
                f"  N={N:,}  ratio={N/N_init:.4f}"
                f"  ({elapsed:.1f}s)", logfile)

    # ------------------------------------------------------------------
    log("\n" + "=" * 70, logfile)
    log(f"Merges performed : {len(merges):,}", logfile)
    log(f"Vocab size       : {V_init:,} → {V:,}  (+{V - V_init:,} phrase tokens)", logfile)
    log(f"Token count      : {N_init:,} → {N:,}", logfile)
    log(f"Token compression: {N/N_init:.4f}  ({(1 - N/N_init)*100:.1f}% fewer tokens)", logfile)
    log(f"N×V (MDL proxy)  : {N_init*V_init:,} → {N*V:,}"
        f"  ({(1 - N*V/(N_init*V_init))*100:.2f}% reduction)", logfile)

    log("\nTop 40 new phrase tokens (by corpus frequency):", logfile)
    log(f"  {'freq':>8}  phrase", logfile)
    log("  " + "-" * 55, logfile)
    for a, b, c, freq, pair_str in sorted(merges, key=lambda x: -x[3])[:40]:
        log(f"  {freq:8,}  {pair_str}", logfile)

    # Save merge list
    out_path = Path("lb_results_base/phrase_bpe_merges.json")
    with open(out_path, "w") as f:
        json.dump([{"a": a, "b": b, "new_id": c, "freq": freq, "pair": pair_str}
                   for a, b, c, freq, pair_str in merges], f, indent=2)
    log(f"\nMerge list saved → {out_path}", logfile)


if __name__ == "__main__":
    main()

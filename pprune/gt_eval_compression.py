"""
gt_eval_compression.py
----------------------
Ground-truth LongBench evaluation across compression ratios.

Outer loop runs ratio batches in order: 65%, 50%, 40%, 35%.
Within each ratio, all methods are evaluated on all tasks before moving to the
next ratio.  This gives usable partial results at the most-generous compression
level early, rather than waiting for all ratios to complete.

Method dispatch
  prompt_construction (METHOD_CONFIGS[method] is None):
      make_comp_ids → model.generate(comp_ids)
      Covers: naive_*, chunk_word*, snapkv_select*, radar_select*
  kv_pruning (METHOD_CONFIGS[method] is a PrunedLlamaConfig):
      patch_model → model.generate(full_ids) → unpatch_model
      Covers: snapkv_NNpct, radar_NNpct

Resumable: checkpoint keyed by "task|idx|method"; any already-completed
(task, idx, method) triple is skipped automatically on restart.

Usage:
    python gt_eval_compression.py \\
        --model meta-llama/Llama-3.1-8B \\
        --data_dir lb_data_raw/data \\
        --output lb_results_base/gt_comp/ \\
        --tasks 2wikimqa,hotpotqa,qasper,narrativeqa,qmsum,triviaqa \\
        --n 100

    # Score only (no inference):
    python gt_eval_compression.py --output lb_results_base/gt_comp/ --score_only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from contextlib import nullcontext
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

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
)
from kl_faith_eval_ystar import make_comp_ids, _KVPRESS_METHODS


# ---------------------------------------------------------------------------
# Primary tasks (6 main LongBench tasks for paper results)
# ---------------------------------------------------------------------------

PRIMARY_TASKS = [
    "2wikimqa",
    "hotpotqa",
    "qasper",
    "narrativeqa",
    "qmsum",
    "triviaqa",
]

# ---------------------------------------------------------------------------
# Default ratio batches — outer loop order
# ---------------------------------------------------------------------------

RATIO_BATCHES = [
    {
        "label": "65pct",
        "methods": [
            "chunk_word160_t25",
            "chunk_word128_t20",
            "snapkv_select",
        ],
    },
    {
        "label": "50pct",
        "methods": [
            "naive_50pct",
            "chunk_word160_t25_f50",
            "chunk_word128_t20_f50",
            "snapkv_select_f50",
            "snapkv_50pct",
            "radar_50pct",
        ],
    },
    {
        "label": "40pct",
        "methods": [
            "naive_40pct",
            "chunk_word160_t25_f40",
            "chunk_word128_t20_f40",
            "snapkv_select_f40",
            "snapkv_40pct",
            "radar_40pct",
        ],
    },
    {
        "label": "35pct",
        "methods": [
            "naive_35pct",
            "chunk_word160_t25_f35",
            "chunk_word128_t20_f35",
            "snapkv_select_f35",
            "snapkv_35pct",
            "radar_35pct",
        ],
    },
]


# ---------------------------------------------------------------------------
# Scoring functions (from longbench_eval.py)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def qa_f1_score(prediction: str, ground_truths: List[str]) -> float:
    def _f1(pred: str, gt: str) -> float:
        pred_toks = _normalize_answer(pred).split()
        gt_toks   = _normalize_answer(gt).split()
        common    = Counter(pred_toks) & Counter(gt_toks)
        n_common  = sum(common.values())
        if n_common == 0:
            return 0.0
        p = n_common / len(pred_toks)
        r = n_common / len(gt_toks)
        return 2 * p * r / (p + r)
    return max(_f1(prediction, gt) for gt in ground_truths)


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def rouge_l_score(prediction: str, ground_truths: List[str]) -> float:
    return max(_rouge.score(gt, prediction)["rougeL"].fmeasure for gt in ground_truths)


def classification_score(prediction: str, ground_truths: List[str]) -> float:
    pred_norm = _normalize_answer(prediction).strip()
    return float(any(_normalize_answer(gt).strip() in pred_norm for gt in ground_truths))


def retrieval_score(prediction: str, ground_truths: List[str]) -> float:
    def extract_num(s: str):
        m = re.search(r"Paragraph\s+(\d+)", s, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"\b(\d+)\b", s)
        return m.group(1) if m else None
    pred_num = extract_num(prediction)
    if pred_num is None:
        return 0.0
    return float(any(extract_num(gt) == pred_num for gt in ground_truths))


def count_score(prediction: str, ground_truths: List[str]) -> float:
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    return float(any(nums[0] == re.sub(r"\D", "", gt) for gt in ground_truths))


def code_sim_score(prediction: str, ground_truths: List[str]) -> float:
    import difflib
    pred = prediction.strip().split("\n")[0].strip()
    return max(difflib.SequenceMatcher(None, pred, gt.strip()).ratio() for gt in ground_truths)


DATASET2SCORER = {
    "narrativeqa":          qa_f1_score,
    "qasper":               qa_f1_score,
    "multifieldqa_en":      qa_f1_score,
    "hotpotqa":             qa_f1_score,
    "2wikimqa":             qa_f1_score,
    "musique":              qa_f1_score,
    "gov_report":           rouge_l_score,
    "qmsum":                rouge_l_score,
    "multi_news":           rouge_l_score,
    "trec":                 classification_score,
    "triviaqa":             qa_f1_score,
    "samsum":               rouge_l_score,
    "passage_count":        count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc":                  code_sim_score,
    "repobench-p":          code_sim_score,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_ckpt(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_ckpt(ckpt: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    tmp.replace(path)


def ckpt_key(task: str, idx: int, method: str) -> str:
    return f"{task}|{idx}|{method}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_from_ids(
    model,
    tokenizer,
    input_ids: torch.Tensor,   # (1, T)
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
    press=None,
) -> str:
    """Generate and decode from a (1, T) input_ids tensor.

    press: optional kvpress context manager; wraps model.generate() if set.
    """
    cap = max_seq_len - max_new_tokens
    if input_ids.shape[1] > cap:
        input_ids = input_ids[:, -cap:]
    input_ids = input_ids.to(device)

    for attempt in range(4):
        try:
            ctx = press(model) if press is not None else nullcontext()
            with ctx:
                out = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0, input_ids.shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            input_ids = input_ids[:, input_ids.shape[1] // 2:]
            if attempt == 3 or input_ids.shape[1] < 64:
                raise


@torch.inference_mode()
def generate_clean_first(
    model,
    tokenizer,
    full_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
    press,
) -> str:
    """
    PyramidKV generation where T1 attends to the compressed KV cache,
    removing the full-prefill first-token advantage present in standard pyramidkv.

    Standard pyramidkv: T1 is determined by dense attention over all tokens (full
    prefill), while T2+ attend only to the compressed KV cache (~65% of tokens).

    Here: full_ids[:, :-1] is prefilled and compressed; then model.generate
    starts from last_tok using that compressed cache, so T1 (and all subsequent
    tokens) attend to the same compressed context.

    Implementation note: after kvpress compresses the KV tensors in place,
    DynamicCache._seen_tokens still holds the original (uncompressed) sequence
    length.  We rebuild a fresh DynamicCache with _seen_tokens equal to the
    compressed layer-0 length so that HuggingFace's position IDs and attention
    masks are consistent with the actual KV tensor sizes.
    """
    cap = max_seq_len - max_new_tokens
    if full_ids.shape[1] > cap:
        full_ids = full_ids[:, -cap:]
    full_ids = full_ids.to(device)

    for attempt in range(4):
        try:
            prefix_ids = full_ids[:, :-1]   # (1, T-1)
            last_tok   = full_ids[:, -1:]   # (1, 1)
            T = full_ids.shape[1]

            # Step 1: prefill prefix with PyramidKV → compressed KV cache.
            with press(model):
                out = model(
                    prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids),
                    use_cache=True,
                    return_dict=True,
                )
            compressed_kv = out.past_key_values
            del out
            torch.cuda.empty_cache()

            # Step 2: decode last_tok against the compressed KV via a direct
            # model() forward — avoids transformers 5.x generate()'s input
            # slicing which would truncate last_tok to an empty tensor.
            # cache_position tells the model where last_tok sits in the sequence
            # so RoPE encoding is correct.
            cache_pos = torch.tensor([T - 1], dtype=torch.long, device=device)
            out2 = model(
                last_tok,
                past_key_values=compressed_kv,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
            T1 = out2.logits[0, -1].argmax().reshape(1, 1)
            kv = out2.past_key_values
            del compressed_kv, out2
            torch.cuda.empty_cache()

            # Step 3: manual greedy decode loop for the remaining tokens.
            # Using model() directly keeps us clear of generate()'s assumptions
            # about input_ids vs past_key_values lengths.
            generated_ids = [T1.item()]
            cur_tok = T1
            for step in range(max_new_tokens - 1):
                cache_pos = torch.tensor([T + step], dtype=torch.long, device=device)
                out_s = model(
                    cur_tok,
                    past_key_values=kv,
                    cache_position=cache_pos,
                    use_cache=True,
                    return_dict=True,
                )
                next_tok = out_s.logits[0, -1].argmax().reshape(1, 1)
                kv = out_s.past_key_values
                del out_s
                tok_id = next_tok.item()
                generated_ids.append(tok_id)
                cur_tok = next_tok
                if tok_id == tokenizer.eos_token_id:
                    break

            del kv
            torch.cuda.empty_cache()
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            full_ids = full_ids[:, full_ids.shape[1] // 2:]
            if attempt == 3 or full_ids.shape[1] < 64:
                raise


def run_one(
    model,
    tokenizer,
    full_ids: torch.Tensor,   # (1, T) — full un-truncated prompt
    method: str,
    question_text: str,
    max_seq_comp: int,
    max_seq_full: int,
    max_new_tokens: int,
    device: str,
) -> str:
    """Run a single (example, method) and return the prediction string."""
    if method == "full":
        return generate_from_ids(model, tokenizer, full_ids, max_new_tokens, max_seq_full, device)
    press = _KVPRESS_METHODS.get(method)
    pcfg  = METHOD_CONFIGS.get(method)

    if press is not None:
        if method.endswith("_clean_first"):
            return generate_clean_first(model, tokenizer, full_ids, max_new_tokens,
                                        max_seq_full, device, press=press)
        # Methods ending in "_short" cap input at max_seq_comp to match naive's
        # effective input length, isolating compression quality from prefill length.
        use_max = max_seq_comp if method.endswith("_short") else max_seq_full
        return generate_from_ids(model, tokenizer, full_ids, max_new_tokens, use_max, device,
                                  press=press)
    elif pcfg is None:
        # Prompt construction — make_comp_ids handles naive, chunk, *_select
        comp_ids = make_comp_ids(
            full_ids, method, tokenizer, question_text,
            max_seq_comp, model=model, device=device,
        )
        return generate_from_ids(model, tokenizer, comp_ids, max_new_tokens, max_seq_full, device)
    else:
        # KV pruning — patch model, generate on the (possibly capped) full prompt
        originals = patch_model(model, pcfg, device)
        try:
            return generate_from_ids(model, tokenizer, full_ids, max_new_tokens, max_seq_full, device)
        finally:
            unpatch_model(model, originals)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def load_task(task: str, data_dir: Path) -> List[dict]:
    path = data_dir / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ratio_batch(
    batch_label: str,
    methods: List[str],
    tasks: List[str],
    model,
    tokenizer,
    ckpt: dict,
    ckpt_path: Path,
    n: Optional[int],
    data_dir: Path,
    device: str,
):
    """Run all methods × tasks × examples for one compression ratio batch."""
    print(f"\n{'='*60}")
    print(f"  Ratio batch: {batch_label}   methods: {methods}")
    print(f"{'='*60}", flush=True)

    for task in tasks:
        dataset      = load_task(task, data_dir)
        n_ex         = min(len(dataset), n) if n else len(dataset)
        max_new      = DATASET2MAXLEN.get(task, 64)
        max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        template     = DATASET2PROMPT[task]

        # Count already-done
        done = sum(
            1 for idx in range(n_ex)
            if all(ckpt_key(task, idx, m) in ckpt for m in methods)
        )
        print(f"\n[{batch_label}] {task}  ({done}/{n_ex} fully done)", flush=True)

        for idx in range(n_ex):
            needed = [m for m in methods if ckpt_key(task, idx, m) not in ckpt]
            if not needed:
                continue

            ex     = dataset[idx]
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
            )
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")

            full_ids = tokenizer.encode(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
            # Cap full_ids at max_seq_full (left-truncate to preserve question at end)
            if full_ids.shape[1] > max_seq_full:
                full_ids = full_ids[:, -max_seq_full:]

            for method in needed:
                t0 = time.perf_counter()
                try:
                    pred = run_one(
                        model, tokenizer, full_ids, method, question_text,
                        max_seq_comp, max_seq_full, max_new, device,
                    )
                except Exception as e:
                    print(f"  WARNING [{task}][{idx}][{method}]: {e}", flush=True)
                    traceback.print_exc()
                    pred = ""

                elapsed = time.perf_counter() - t0
                ckpt[ckpt_key(task, idx, method)] = {
                    "prediction": pred,
                    "answers":    ex["answers"],
                }
                save_ckpt(ckpt, ckpt_path)
                print(f"  [{task}] ex={idx}/{n_ex-1}  {method}  {elapsed:.1f}s  → {repr(pred[:60])}", flush=True)


# ---------------------------------------------------------------------------
# Scoring and display
# ---------------------------------------------------------------------------

def score_results(tasks: List[str], ckpt: dict, n: Optional[int]) -> dict:
    all_methods: set = set()
    for k in ckpt:
        parts = k.split("|")
        if len(parts) == 3:
            all_methods.add(parts[2])

    results: dict = {}
    for task in tasks:
        scorer_fn = DATASET2SCORER.get(task)
        if scorer_fn is None:
            continue

        indices = sorted({int(k.split("|")[1]) for k in ckpt if k.startswith(f"{task}|")})
        if n:
            indices = indices[:n]

        task_scores: Dict[str, List[float]] = {m: [] for m in all_methods}
        for idx in indices:
            for method in all_methods:
                key = ckpt_key(task, idx, method)
                if key not in ckpt:
                    continue
                entry = ckpt[key]
                s = scorer_fn(entry["prediction"], entry["answers"])
                task_scores[method].append(s)

        results[task] = {
            m: (100.0 * sum(v) / len(v) if v else float("nan"))
            for m, v in task_scores.items()
            if v  # only methods with any data
        }
        results[task]["n"] = len(indices)

    return results


def print_table(results: dict, method_order: Optional[List[str]] = None):
    if not results:
        return

    all_methods: set = set()
    for r in results.values():
        all_methods.update(k for k in r if k not in ("n",))
    if method_order:
        methods = [m for m in method_order if m in all_methods]
        methods += sorted(all_methods - set(method_order))
    else:
        methods = sorted(all_methods)

    col_w  = 12
    header = f"{'Task':<26} {'N':>5}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)

    totals: Dict[str, List[float]] = {m: [] for m in methods}
    for task, r in sorted(results.items()):
        row = f"{task:<26} {r.get('n', 0):>5}  " + "  ".join(
            f"{r.get(m, float('nan')):>{col_w}.2f}" for m in methods
        )
        print(row)
        for m in methods:
            v = r.get(m, float("nan"))
            if not math.isnan(v):
                totals[m].append(v)

    print(sep)
    avg_row = f"{'AVERAGE':<26} {'':>5}  " + "  ".join(
        f"{(sum(v)/len(v) if v else float('nan')):>{col_w}.2f}" for m in methods
        for v in [totals[m]]
    )
    print(avg_row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ground-truth LongBench evaluation across compression ratios."
    )
    p.add_argument("--model",      default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir",   default="lb_data_raw/data")
    p.add_argument("--output",     default="lb_results_base/gt_comp",
                   help="Output directory for checkpoint and results JSON")
    p.add_argument("--tasks",      default=",".join(PRIMARY_TASKS),
                   help="Comma-separated task names")
    p.add_argument("--methods",    default=None,
                   help="Comma-separated method names; if set, runs a single 65pct batch "
                        "with exactly these methods instead of the default RATIO_BATCHES")
    p.add_argument("--n",          type=int, default=100,
                   help="Max examples per task (default: 100)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--score_only", action="store_true",
                   help="Skip inference; just score whatever is in the checkpoint")
    return p.parse_args()


def main():
    args     = parse_args()
    tasks    = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    out_dir  = Path(args.output)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"

    print(f"Tasks     : {tasks}")
    print(f"N/task    : {args.n}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = load_ckpt(ckpt_path)

    if not args.score_only:
        print(f"\nLoading {args.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=args.device,
        )
        model.eval()

        if args.methods:
            override = [m.strip() for m in args.methods.split(",")
                        if m.strip() in METHOD_CONFIGS or m.strip() in _KVPRESS_METHODS]
            batches = [{"label": "65pct", "methods": override}]
        else:
            batches = RATIO_BATCHES

        all_methods = [m for batch in batches for m in batch["methods"]]
        print(f"All methods: {all_methods}\n")

        try:
            for batch in batches:
                run_ratio_batch(
                    batch_label=batch["label"],
                    methods=batch["methods"],
                    tasks=tasks,
                    model=model,
                    tokenizer=tokenizer,
                    ckpt=ckpt,
                    ckpt_path=ckpt_path,
                    n=args.n,
                    data_dir=data_dir,
                    device=args.device,
                )
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved. Re-run to resume.")
            sys.exit(0)

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Score and report
    # ------------------------------------------------------------------
    print("\n=== Ground-truth scores ===")
    results = score_results(tasks, ckpt, args.n)

    if args.methods:
        batches = [{"label": "65pct", "methods": [m.strip() for m in args.methods.split(",")
                    if m.strip() in METHOD_CONFIGS or m.strip() in _KVPRESS_METHODS]}]
    else:
        batches = RATIO_BATCHES
    method_order = [m for batch in batches for m in batch["methods"]]
    print_table(results, method_order=method_order)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

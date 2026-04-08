"""
kvpress_benchmark.py
--------------------
Benchmark AdditiveScorerPress against SnapKVPress and full-context baseline
on a representative LongBench subset, using the KVPress hook interface.

Loads the model once; each press is applied via context manager without
reloading weights.

Usage
-----
    python kvpress_benchmark.py \\
        --model meta-llama/Llama-3.1-8B \\
        --tasks passage_retrieval_en,hotpotqa,2wikimqa,gov_report,qmsum,lcc \\
        --max_examples 30 \\
        --compression_ratio 0.35 \\
        --output kvpress_benchmark_results.json

Notes
-----
- compression_ratio = fraction to DROP (0.35 → keep 65%, matching paper default)
- SnapKV window_size is set to match q_buffer_size for a fair comparison
- Full-context baseline uses no press (press=None)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import SnapKVPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress

from additive_scorer_press import AdditiveScorerPress, HeadAwareAdditiveScorerPress

# ---------------------------------------------------------------------------
# LongBench task definitions (subset used here)
# ---------------------------------------------------------------------------

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
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news passages.\n\n"
        "News Passages:\n{context}\n\n"
        "Now, write a one-page summary of all the news passages above.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read through each paragraph and determine how many unique paragraphs there "
        "are after removing duplicates. In your response, just output a single number. "
        "Do not explain your reasoning.\n\n{context}\n\nNumber of unique paragraphs:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is:"
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n{context}{input}Next line of code:\n"
    ),
}

DATASET2MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
    "2wikimqa": 32, "musique": 32, "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128, "passage_count": 32,
    "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64,
}

ROUGE_TASKS = {"gov_report", "qmsum", "multi_news", "samsum"}
F1_TASKS = {"narrativeqa", "qasper", "hotpotqa", "2wikimqa", "musique",
            "trec", "triviaqa", "multifieldqa_en"}
RETRIEVAL_TASKS = {"passage_retrieval_en"}
COUNT_TASKS = {"passage_count"}
CODE_TASKS = {"lcc", "repobench-p"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def f1(pred: str, answers: list[str]) -> float:
    pred_toks = normalize(pred).split()
    scores = []
    for ans in answers:
        ans_toks = normalize(ans).split()
        common = Counter(pred_toks) & Counter(ans_toks)
        n_common = sum(common.values())
        if n_common == 0:
            scores.append(0.0)
            continue
        p = n_common / len(pred_toks)
        r = n_common / len(ans_toks)
        scores.append(2 * p * r / (p + r))
    return max(scores)

def rouge_l(pred: str, answers: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(scorer.score(ans, pred)["rougeL"].fmeasure for ans in answers)

def retrieval_score(pred: str, answers: list[str]) -> float:
    """Extract paragraph number from prediction and answer; compare."""
    def extract_num(s):
        m = re.search(r"Paragraph\s+(\d+)", s, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"\b(\d+)\b", s)
        return m.group(1) if m else None
    pred_num = extract_num(pred)
    if pred_num is None:
        return 0.0
    return float(any(extract_num(a) == pred_num for a in answers))

def count_score(pred: str, answers: list[str]) -> float:
    def extract_num(s):
        m = re.search(r"\b(\d+)\b", s)
        return m.group(1) if m else None
    pred_num = extract_num(pred)
    if pred_num is None:
        return 0.0
    return float(any(extract_num(a) == pred_num for a in answers))

def edit_sim(pred: str, answers: list[str]) -> float:
    """Normalised edit similarity for code tasks."""
    import difflib
    return max(difflib.SequenceMatcher(None, pred, a).ratio() for a in answers)

def score_example(task: str, pred: str, answers: list[str]) -> float:
    if task in ROUGE_TASKS:
        return rouge_l(pred, answers)
    if task in F1_TASKS:
        return f1(pred, answers)
    if task in RETRIEVAL_TASKS:
        return retrieval_score(pred, answers)
    if task in COUNT_TASKS:
        return count_score(pred, answers)
    if task in CODE_TASKS:
        return edit_sim(pred, answers)
    return f1(pred, answers)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task(task: str, data_dir: Path, max_examples: int):
    path = data_dir / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= max_examples:
                break
    return examples


def build_prompt(task: str, example: dict, tokenizer, max_seq_len: int) -> str:
    template = DATASET2PROMPT[task]
    context = example.get("context", "")
    input_text = example.get("input", "")
    prompt = template.format(context=context, input=input_text)

    # Truncate context if prompt is too long
    toks = tokenizer.encode(prompt)
    if len(toks) > max_seq_len:
        # Re-build with truncated context
        excess = len(toks) - max_seq_len
        ctx_toks = tokenizer.encode(context)
        context = tokenizer.decode(ctx_toks[excess:], skip_special_tokens=True)
        prompt = template.format(context=context, input=input_text)
    return prompt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_task(task, examples, model, tokenizer, press, max_seq_len, device):
    max_new = DATASET2MAXLEN[task]
    scores = []
    for ex in examples:
        prompt = build_prompt(task, ex, tokenizer, max_seq_len)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        answers = ex.get("answers", [ex.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]

        try:
            if press is not None:
                with press(model):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            scores.append(0.0)
            continue
        except Exception as e:
            print(f"  WARNING: {task} example failed: {e}")
            torch.cuda.empty_cache()
            scores.append(0.0)
            continue
        finally:
            torch.cuda.empty_cache()

        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(score_example(task, pred, answers))

    return sum(scores) / len(scores) * 100 if scores else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",            default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tasks",            default="passage_retrieval_en,hotpotqa,2wikimqa,gov_report,qmsum,lcc")
    p.add_argument("--max_examples",     type=int, default=30)
    p.add_argument("--compression_ratio",type=float, default=0.35,
                   help="Fraction of tokens to DROP (0.35 = keep 65%)")
    p.add_argument("--score_alpha",      type=float, default=0.65)
    p.add_argument("--min_decay",        type=float, default=0.7)
    p.add_argument("--q_buffer_size",    type=int,   default=128)
    p.add_argument("--always_keep_first",type=int,   default=16)
    p.add_argument("--always_keep_last", type=int,   default=16)
    p.add_argument("--max_seq_len",      type=int,   default=6144)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--data_dir",         default="lb_data_raw/data")
    p.add_argument("--output",           default="kvpress_benchmark_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in DATASET2PROMPT]
    data_dir = Path(args.data_dir)

    print(f"Model  : {args.model}")
    print(f"Tasks  : {tasks}")
    print(f"Keep   : {100*(1-args.compression_ratio):.0f}% (compression_ratio={args.compression_ratio})")
    print(f"N      : {args.max_examples} examples/task")

    print(f"\nLoading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    device = args.device

    presses = {
        "full": None,
        "additive": AdditiveScorerPress(
            compression_ratio=args.compression_ratio,
            score_alpha=args.score_alpha,
            min_decay=args.min_decay,
            q_buffer_size=args.q_buffer_size,
            always_keep_first=args.always_keep_first,
            always_keep_last=args.always_keep_last,
        ),
        "head_aware": HeadAwareAdditiveScorerPress(
            compression_ratio=args.compression_ratio,
            score_alpha=args.score_alpha,
            min_decay=args.min_decay,
            q_buffer_size=args.q_buffer_size,
            always_keep_first=args.always_keep_first,
            always_keep_last=args.always_keep_last,
        ),
        "snapkv": SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.q_buffer_size,
        ),
        "streaming": StreamingLLMPress(
            compression_ratio=args.compression_ratio,
        ),
    }

    results = {}
    timings = {}

    col_w = 20
    task_w = 22
    header = f"{'Task':<{task_w}}" + "".join(f"  {k:>{col_w}}" for k in presses)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for task in tasks:
        try:
            examples = load_task(task, data_dir, args.max_examples)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        row_scores = {}
        row_times = {}
        for name, press in presses.items():
            t0 = time.time()
            score = run_task(task, examples, model, tokenizer, press, args.max_seq_len, device)
            elapsed = time.time() - t0
            row_scores[name] = score
            row_times[name] = elapsed
            torch.cuda.empty_cache()

        results[task] = row_scores
        timings[task] = row_times

        row = f"{task:<{task_w}}" + "".join(f"  {row_scores[n]:>{col_w}.1f}" for n in presses)
        print(row)

    print("=" * len(header))

    # Summary averages
    if results:
        print(f"\n{'AVERAGE':<{task_w}}", end="")
        for name in presses:
            avg = sum(results[t][name] for t in results if name in results[t]) / len(results)
            print(f"  {avg:>{col_w}.1f}", end="")
        print()

        # Timing summary
        print(f"\n{'TOTAL TIME (s)':<{task_w}}", end="")
        for name in presses:
            total = sum(timings[t][name] for t in timings if name in timings[t])
            print(f"  {total:>{col_w}.0f}", end="")
        print()

    # Save
    out = {"results": results, "timings": timings, "args": vars(args)}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

"""
longbench_eval.py
-----------------
Evaluates baseline, naive-truncation, and KV-pruned Llama models on LongBench.

Resumable: predictions are checkpointed to a JSON file after every example.
On restart, already-completed examples are skipped automatically.

Strategy (memory-safe for 32 GB V100):
  Pass 1 — load base Llama (fp16, ~16 GB), run "full" and "naive" for all tasks.
  Pass 2 — load pruned model (~16 GB), run "pruned" for all tasks.
  Score  — compute metrics and print table.

Usage:
    python longbench_eval.py \\
        --model meta-llama/Llama-3-8B-Instruct \\
        --budget 4096 \\
        --tasks narrativeqa,hotpotqa,gov_report \\
        --max_examples 200 \\
        --output lb_results/

    # Interrupt with Ctrl-C at any time; re-run the same command to resume.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from llama_pruned import PrunedLlamaConfig, build_pruned_model


# ---------------------------------------------------------------------------
# LongBench task definitions (English only)
# ---------------------------------------------------------------------------

ENGLISH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

DATASET2PROMPT: Dict[str, str] = {
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
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
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
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n{context}{input}Next line of code:\n"
    ),
}

DATASET2MAXLEN: Dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lower, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def qa_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Token-level F1, maximised over multiple ground-truth answers."""
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
    """ROUGE-L F-measure, maximised over multiple references."""
    scores = [_rouge.score(gt, prediction)["rougeL"].fmeasure for gt in ground_truths]
    return max(scores)


def classification_score(prediction: str, ground_truths: List[str]) -> float:
    pred_norm = _normalize_answer(prediction).strip()
    return float(any(_normalize_answer(gt).strip() in pred_norm for gt in ground_truths))


def retrieval_score(prediction: str, ground_truths: List[str]) -> float:
    """Check if the predicted paragraph number matches any ground truth.
    Accepts 'Paragraph N', 'paragraph N', or a bare number."""
    def extract_num(s: str) -> str | None:
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
    pred_num = nums[0]
    return float(any(pred_num == re.sub(r"\D", "", gt) for gt in ground_truths))


def code_sim_score(prediction: str, ground_truths: List[str]) -> float:
    import difflib
    pred = prediction.strip().split("\n")[0].strip()
    scores = [
        difflib.SequenceMatcher(None, pred, gt.strip()).ratio()
        for gt in ground_truths
    ]
    return max(scores)


DATASET2SCORER = {
    "narrativeqa":        qa_f1_score,
    "qasper":             qa_f1_score,
    "multifieldqa_en":    qa_f1_score,
    "hotpotqa":           qa_f1_score,
    "2wikimqa":           qa_f1_score,
    "musique":            qa_f1_score,
    "gov_report":         rouge_l_score,
    "qmsum":              rouge_l_score,
    "multi_news":         rouge_l_score,
    "trec":               classification_score,
    "triviaqa":           qa_f1_score,
    "samsum":             rouge_l_score,
    "passage_count":      count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc":                code_sim_score,
    "repobench-p":        code_sim_score,
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_prompt(task: str, example: dict) -> str:
    template = DATASET2PROMPT[task]
    return template.format(
        context=example.get("context", ""),
        input=example.get("input", ""),
    )


def truncate_to_budget(
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    max_new_tokens: int,
) -> str:
    """
    Truncate prompt so that (prompt tokens + max_new_tokens) <= max_tokens.
    Removes tokens from the middle of the context to preserve the question
    (which is at the end) and a bit of the beginning.
    """
    ids = tokenizer.encode(prompt)
    budget = max_tokens - max_new_tokens
    if len(ids) <= budget:
        return prompt
    # Keep first 10% and last 90% of the budget (question is at the end)
    keep_start = max(budget // 10, 64)
    keep_end   = budget - keep_start
    ids = ids[:keep_start] + ids[-keep_end:]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.inference_mode()
def generate(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]

    # Hard-truncate from left if prompt exceeds the model context window
    cap = max_seq_len - max_new_tokens
    if input_ids.shape[1] > cap:
        input_ids = input_ids[:, -cap:]

    input_ids = input_ids.to(device)

    # Retry with progressively shorter input on OOM
    for attempt in range(4):
        try:
            out = model.generate(
                input_ids,
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    tmp.replace(path)   # atomic on POSIX


def ckpt_key(task: str, idx: int, method: str) -> str:
    return f"{task}|{idx}|{method}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def load_task(task: str, data_dir: Path) -> List[dict]:
    """Load a LongBench task from local JSONL files."""
    path = data_dir / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run:  python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset', local_dir='lb_data_raw')\"\n"
            f"Then: python -c \"import zipfile; zipfile.ZipFile('lb_data_raw/data.zip').extractall('lb_data_raw/')\""
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pass(
    pass_name: str,
    methods: List[str],
    tasks: List[str],
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    ckpt: dict,
    ckpt_path: Path,
    budget: int,
    max_examples: Optional[int],
    max_seq_len: int,
    device: str,
    data_dir: Path,
):
    """Run inference for the given methods/tasks using the already-loaded model."""
    for task in tasks:
        print(f"\n[{pass_name}] Task: {task}")
        dataset = load_task(task, data_dir)
        max_new_tokens = DATASET2MAXLEN[task]

        n = min(len(dataset), max_examples) if max_examples else len(dataset)
        done = 0

        for idx in range(n):
            keys_needed = [ckpt_key(task, idx, m) for m in methods]
            if all(k in ckpt for k in keys_needed):
                done += 1
                continue

            example = dataset[idx]
            prompt  = build_prompt(task, example)

            for method in methods:
                key = ckpt_key(task, idx, method)
                if key in ckpt:
                    continue

                if method == "naive":
                    p = truncate_to_budget(tokenizer, prompt, budget, max_new_tokens)
                else:
                    p = prompt

                try:
                    pred = generate(model, tokenizer, p, max_new_tokens, max_seq_len, device)
                except Exception as e:
                    print(f"  WARNING: {task}[{idx}] {method} failed: {e}")
                    pred = ""

                ckpt[key] = {"prediction": pred, "answers": example["answers"]}
                save_checkpoint(ckpt, ckpt_path)

            done += 1
            if done % 20 == 0 or done == n:
                print(f"  {done}/{n} examples done")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(tasks: List[str], ckpt: dict, max_examples: Optional[int]) -> dict:
    # Discover all methods present in the checkpoint (preserving "full" first, "naive" second)
    all_methods_seen: set = set()
    for k in ckpt:
        parts = k.split("|")
        if len(parts) == 3:
            all_methods_seen.add(parts[2])
    preferred_order = ["full", "naive"]
    methods = preferred_order + sorted(all_methods_seen - set(preferred_order))

    results: dict = {}
    for task in tasks:
        scorer_fn = DATASET2SCORER[task]
        task_scores: Dict[str, List[float]] = {m: [] for m in methods}

        indices = sorted({
            int(k.split("|")[1])
            for k in ckpt
            if k.startswith(f"{task}|")
        })
        if max_examples:
            indices = indices[:max_examples]

        for idx in indices:
            for method in methods:
                key = ckpt_key(task, idx, method)
                if key not in ckpt:
                    continue
                entry = ckpt[key]
                s = scorer_fn(entry["prediction"], entry["answers"])
                task_scores[method].append(s)

        results[task] = {
            m: (100.0 * sum(v) / len(v) if v else float("nan"))
            for m, v in task_scores.items()
        }
        results[task]["n"] = len(indices)
        results[task]["_methods"] = methods

    return results


def print_table(results: dict):
    if not results:
        return
    methods = next(iter(results.values()))["_methods"]
    col_w = 8
    header = f"{'Task':<24} {'N':>5}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    totals: Dict[str, List[float]] = {m: [] for m in methods}
    for task, r in results.items():
        row = f"{task:<24} {r['n']:>5}  " + "  ".join(f"{r.get(m, float('nan')):>{col_w}.2f}" for m in methods)
        print(row)
        for m in methods:
            v = r.get(m, float("nan"))
            if not math.isnan(v):
                totals[m].append(v)
    print(sep)
    avgs = {m: sum(v) / len(v) if v else float("nan") for m, v in totals.items()}
    avg_row = f"{'AVERAGE':<24} {'':>5}  " + "  ".join(f"{avgs[m]:>{col_w}.2f}" for m in methods)
    print(avg_row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LongBench evaluation with KV pruning")
    p.add_argument("--model",        default="meta-llama/Llama-3-8B-Instruct")
    p.add_argument("--budget",       type=int, default=4096,
                   help="KV token budget for naive truncation (and pruned model when budget_fraction=0)")
    p.add_argument("--budget_fraction", type=float, default=0.0,
                   help="If > 0, pruned model keeps this fraction of tokens per sequence (overrides --budget for pruning)")
    p.add_argument("--tasks",        default=",".join(ENGLISH_TASKS),
                   help="Comma-separated list of tasks (default: all English tasks)")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Cap examples per task (useful for quick smoke tests)")
    p.add_argument("--max_seq_len",  type=int, default=7168,
                   help="Hard context-window cap; leave headroom for activations (default 7168)")
    p.add_argument("--output",       default="lb_results",
                   help="Output directory for checkpoint and final results")
    p.add_argument("--device",       default="cuda")
    # PrunedLlamaConfig knobs
    p.add_argument("--score_mode",   default="additive",
                   choices=["kq_vn_decay", "kq_only", "kq_post_rope", "snapkv",
                            "vn_only", "vn_decay", "additive", "streaming"])
    p.add_argument("--score_alpha",  type=float, default=0.65)
    p.add_argument("--sink_size",    type=int, default=4,
                   help="Attention sink tokens kept in streaming mode")
    p.add_argument("--method_label", default="pruned",
                   help="Checkpoint key used for Pass 2 predictions (default: 'pruned'). "
                        "Set to e.g. 'streaming' to add a new column without overwriting existing results.")
    p.add_argument("--decay_fn",     default="linear",
                   choices=["linear", "exponential"])
    p.add_argument("--min_decay",    type=float, default=0.7)
    p.add_argument("--always_keep_first", type=int, default=16)
    p.add_argument("--always_keep_last",  type=int, default=16)
    p.add_argument("--q_buffer_size",     type=int, default=64)
    p.add_argument("--data_dir",     default="lb_data_raw/data",
                   help="Directory containing LongBench JSONL files")
    p.add_argument("--score_only",   action="store_true",
                   help="Skip inference, just re-score whatever is in the checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    tasks    = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    out_dir  = Path(args.output)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"

    print(f"Tasks     : {tasks}")
    print(f"Budget    : {args.budget}")
    print(f"Max seq   : {args.max_seq_len}")
    print(f"Data dir  : {data_dir}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)

    if not args.score_only:
        # ------------------------------------------------------------------
        # Pass 1: base model — "full" context and "naive" truncation
        # ------------------------------------------------------------------
        print("\n=== Pass 1: base model (full + naive) ===")
        print(f"Loading {args.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=args.device,
        )
        base_model.eval()

        try:
            run_pass(
                pass_name="Pass1",
                methods=["full", "naive"],
                tasks=tasks,
                model=base_model,
                tokenizer=tokenizer,
                ckpt=ckpt,
                ckpt_path=ckpt_path,
                budget=args.budget,
                max_examples=args.max_examples,
                max_seq_len=args.max_seq_len,
                device=args.device,
                data_dir=data_dir,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved. Re-run to continue.")
            sys.exit(0)

        del base_model
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Pass 2: pruned model
        # ------------------------------------------------------------------
        print("\n=== Pass 2: pruned model ===")
        pcfg = PrunedLlamaConfig(
            total_budget=args.budget,
            budget_fraction=args.budget_fraction,
            q_buffer_size=args.q_buffer_size,
            decay_fn=args.decay_fn,
            min_decay=args.min_decay,
            always_keep_last=args.always_keep_last,
            always_keep_first=args.always_keep_first,
            score_mode=args.score_mode,
            score_alpha=args.score_alpha,
            sink_size=args.sink_size,
        )
        pruned_model, tokenizer = build_pruned_model(
            args.model, pcfg, device=args.device, dtype=torch.float16,
        )

        try:
            run_pass(
                pass_name="Pass2",
                methods=[args.method_label],
                tasks=tasks,
                model=pruned_model,
                tokenizer=tokenizer,
                ckpt=ckpt,
                ckpt_path=ckpt_path,
                budget=args.budget,
                max_examples=args.max_examples,
                max_seq_len=args.max_seq_len,
                device=args.device,
                data_dir=data_dir,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved. Re-run to continue.")
            sys.exit(0)

        del pruned_model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Score and report
    # ------------------------------------------------------------------
    print("\n=== Scoring ===")
    results = score_results(tasks, ckpt, args.max_examples)
    print_table(results)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()

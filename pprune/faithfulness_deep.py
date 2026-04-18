"""
faithfulness_deep.py
--------------------
Two model-independent faithfulness metrics that avoid the cascading-divergence
problem of token-level lexical comparison:

  1. Perplexity-based (--perplexity):
     For each compressed method's output, compute the mean log-probability per
     token under the full-context model.  Reports a "faithfulness ratio":
       faith_ratio = exp(loss_full - loss_compressed)
     where loss_full is the full-context model's own output scored against
     itself (the best achievable score), and loss_compressed is the compressed
     output scored by the same model.  faith_ratio = 1.0 means the compressed
     output is exactly as likely as the full-context output; < 1.0 means it is
     less likely.  Reported as a percentage (× 100).

  2. Embedding-based (--embedding):
     Encode each output with a sentence-transformer and compute cosine
     similarity between the compressed output and the full-context output.
     Handles paraphrasing; does not require the LLM at all.

Both metrics sidestep cascading token divergence: perplexity conditions on
the reference tokens, not on the model's own previous outputs; embedding
similarity compares semantic content, not token identity.

Checkpointing
-------------
Results are saved after every task completes.  If the output file already
exists, completed tasks are skipped automatically — just re-run the same
command to resume after a crash or kill.

Usage
-----
    # Both metrics, Llama, lb_results_base checkpoint:
    python faithfulness_deep.py \\
        --model meta-llama/Llama-3.1-8B \\
        --checkpoint lb_results_base/checkpoint.json \\
        --output lb_results_base/faithfulness_deep.json \\
        --perplexity --embedding

    # Embedding only (no GPU needed for the LLM):
    python faithfulness_deep.py \\
        --checkpoint lb_results_base/checkpoint.json \\
        --output lb_results_base/faithfulness_deep.json \\
        --embedding --no_perplexity
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Task / data definitions (mirrors longbench_eval.py)
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
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary: The following is a summary of the above news articles:"
    ),
    "trec": ("Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}"),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": ("Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}"),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": ("Please complete the code given below.\n{context}Next line of code:\n"),
    "repobench-p": ("Please complete the code given below.\n{context}{input}Next line of code:\n"),
}

DATASET2MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
    "2wikimqa": 32, "musique": 32, "gov_report": 512, "qmsum": 512,
    "multi_news": 512, "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64,
}

LOW_SIGNAL_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_partial(output_path: Path) -> dict:
    """Load existing partial results from output file, if any."""
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def save_partial(all_results: dict, output_path: Path):
    """Atomically save current results."""
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2)
    tmp.replace(output_path)


# ---------------------------------------------------------------------------
# Perplexity scoring
# ---------------------------------------------------------------------------

@torch.inference_mode()
def score_output_loss(
    model,
    tokenizer,
    prompt: str,
    output_text: str,
    max_seq_len: int,
    device: str,
) -> Optional[float]:
    """
    Return the mean per-token cross-entropy loss when the model is asked to
    reproduce output_text given prompt (teacher-forced).
    Lower loss = output is more likely under this model.
    Returns None if output_text is empty or too short to score.
    """
    output_text = output_text.strip()
    if not output_text:
        return None

    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    if len(output_ids) == 0:
        return None

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Combine; trim from left if over limit, preserving all output tokens
    max_prompt = max_seq_len - len(output_ids) - 1
    if max_prompt < 32:
        max_prompt = 32
    prompt_ids = prompt_ids[-max_prompt:]

    input_ids = torch.tensor([prompt_ids + output_ids], device=device)
    prompt_len = len(prompt_ids)

    try:
        out = model(input_ids)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    # Logits for positions [prompt_len-1 .. end-1] predict tokens [prompt_len .. end]
    logits = out.logits[0, prompt_len - 1 : -1, :]       # (output_len, vocab)
    targets = input_ids[0, prompt_len:]                   # (output_len,)

    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss.item()


def compute_perplexity_faithfulness(
    tasks: List[str],
    ckpt: dict,
    data_dir: Path,
    model,
    tokenizer,
    max_examples: int,
    max_seq_len: int,
    device: str,
    all_results: dict,
    output_path: Path,
    reference_method: str = "full",
) -> dict:
    """
    For each method, compute the mean faithfulness ratio:
        faith_ratio_i = exp(loss_full_i - loss_method_i)
    where loss_full_i is the full-context model's OWN output scored against
    itself, and loss_method_i is the compressed output scored by the same
    full-context model.  Ratio = 1 means equally likely; < 1 means less likely.
    Reported × 100.

    Saves results incrementally to output_path after each task.
    Skips tasks already present in all_results["perplexity"].
    """
    all_methods = sorted({
        k.split("|")[2] for k in ckpt
        if len(k.split("|")) == 3 and k.split("|")[2] != reference_method
    })

    existing = all_results.get("perplexity", {})
    results: dict = dict(existing)
    n_tasks = len(tasks)

    for t_idx, task in enumerate(tasks):
        if task in existing:
            print(f"  [{task}] SKIP (already done)  ({t_idx+1}/{n_tasks})", flush=True)
            continue

        print(f"  [{task}]", end="", flush=True)
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f" SKIP (no data file)  ({t_idx+1}/{n_tasks})", flush=True)
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        indices = sorted({
            int(k.split("|")[1])
            for k in ckpt if k.startswith(f"{task}|")
        })[:max_examples]

        task_scores: Dict[str, List[float]] = {m: [] for m in all_methods}
        ref_losses: List[float] = []

        for idx in indices:
            ref_key = f"{task}|{idx}|{reference_method}"
            if ref_key not in ckpt:
                continue
            ref_output = ckpt[ref_key]["prediction"].strip()
            if not ref_output:
                continue

            ex = dataset[idx] if idx < len(dataset) else None
            if ex is None:
                continue
            template = DATASET2PROMPT.get(task, "{context}{input}")
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
            )

            # Score the full-context output against itself (reference baseline)
            ref_loss = score_output_loss(model, tokenizer, prompt, ref_output,
                                         max_seq_len, device)
            if ref_loss is None:
                continue
            ref_losses.append(ref_loss)

            for method in all_methods:
                mkey = f"{task}|{idx}|{method}"
                if mkey not in ckpt:
                    continue
                mout = ckpt[mkey]["prediction"].strip()
                if not mout:
                    task_scores[method].append(0.0)
                    continue
                m_loss = score_output_loss(model, tokenizer, prompt, mout,
                                           max_seq_len, device)
                if m_loss is None:
                    continue
                ratio = math.exp(ref_loss - m_loss)
                task_scores[method].append(min(ratio, 2.0) * 100)  # cap at 200%

        results[task] = {
            m: (float(np.mean(v)) if v else float("nan"))
            for m, v in task_scores.items()
        }
        results[task]["_methods"] = all_methods
        results[task]["n"] = len(ref_losses)
        print(f" n={len(ref_losses)}  ({t_idx+1}/{n_tasks})", flush=True)

        # Save after every task
        all_results["perplexity"] = results
        save_partial(all_results, output_path)

    return results


# ---------------------------------------------------------------------------
# Embedding-based scoring
# ---------------------------------------------------------------------------

def compute_embedding_faithfulness(
    tasks: List[str],
    ckpt: dict,
    max_examples: int,
    all_results: dict,
    output_path: Path,
    reference_method: str = "full",
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Compute cosine similarity between compressed-method outputs and the
    full-context output using a sentence-transformer embedding model.
    Scores are in [0, 100]; 100 = identical embedding, 50 = orthogonal.

    Saves results incrementally to output_path after each task.
    Skips tasks already present in all_results["embedding"].
    """
    from sentence_transformers import SentenceTransformer
    print(f"  Loading sentence-transformer: {model_name} ...")
    embedder = SentenceTransformer(model_name)

    all_methods = sorted({
        k.split("|")[2] for k in ckpt
        if len(k.split("|")) == 3 and k.split("|")[2] != reference_method
    })

    existing = all_results.get("embedding", {})
    results: dict = dict(existing)
    n_tasks = len(tasks)

    for t_idx, task in enumerate(tasks):
        if task in existing:
            print(f"  [{task}] SKIP (already done)  ({t_idx+1}/{n_tasks})", flush=True)
            continue

        print(f"  [{task}]", end="", flush=True)

        indices = sorted({
            int(k.split("|")[1])
            for k in ckpt if k.startswith(f"{task}|")
        })[:max_examples]

        task_scores: Dict[str, List[float]] = {m: [] for m in all_methods}

        ref_texts, method_texts_by_m = [], {m: [] for m in all_methods}

        for idx in indices:
            ref_key = f"{task}|{idx}|{reference_method}"
            if ref_key not in ckpt:
                continue
            ref_out = ckpt[ref_key]["prediction"].strip()
            if not ref_out:
                continue
            ref_texts.append(ref_out)
            for m in all_methods:
                mkey = f"{task}|{idx}|{m}"
                mout = ckpt[mkey]["prediction"].strip() if mkey in ckpt else ""
                method_texts_by_m[m].append(mout if mout else " ")

        if not ref_texts:
            results[task] = {m: float("nan") for m in all_methods}
            results[task]["_methods"] = all_methods
            results[task]["n"] = 0
            print(f" n=0  ({t_idx+1}/{n_tasks})", flush=True)
        else:
            ref_embs = embedder.encode(ref_texts, batch_size=64,
                                       show_progress_bar=False, normalize_embeddings=True)

            for m in all_methods:
                m_embs = embedder.encode(method_texts_by_m[m], batch_size=64,
                                         show_progress_bar=False, normalize_embeddings=True)
                sims = (ref_embs * m_embs).sum(axis=1)
                task_scores[m] = ((sims + 1) / 2 * 100).tolist()

            results[task] = {
                m: float(np.mean(task_scores[m])) if task_scores[m] else float("nan")
                for m in all_methods
            }
            results[task]["_methods"] = all_methods
            results[task]["n"] = len(ref_texts)
            print(f" n={len(ref_texts)}  ({t_idx+1}/{n_tasks})", flush=True)

        # Save after every task
        all_results["embedding"] = results
        save_partial(all_results, output_path)

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_faithfulness_table(results: dict, title: str, gt_results: Optional[dict] = None):
    if not results:
        return
    methods = next(iter(results.values()))["_methods"]
    col_w = 9
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    header = f"{'Task':<26} {'N':>4}  " + "  ".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))
    totals: Dict[str, List[float]] = {m: [] for m in methods}
    for task, r in results.items():
        if task.startswith("_"):
            continue
        flag = ""
        if gt_results:
            fg = gt_results.get(task, {}).get("full", 100.0)
            flag = "†" if (isinstance(fg, float) and fg < LOW_SIGNAL_THRESHOLD) else " "
        row = f"{task:<24}{flag} {r.get('n', 0):>4}  " + \
              "  ".join(f"{r.get(m, float('nan')):>{col_w}.1f}" for m in methods)
        print(row)
        for m in methods:
            v = r.get(m, float("nan"))
            if not math.isnan(v):
                totals[m].append(v)
    print("-" * len(header))
    avgs = {m: np.mean(v) if v else float("nan") for m, v in totals.items()}
    avg_row = f"{'AVERAGE':<26} {'':>4}  " + \
              "  ".join(f"{avgs[m]:>{col_w}.1f}" for m in methods)
    print(avg_row)
    if gt_results:
        print("  † full-context ground-truth accuracy < 10%: model unreliable on this task")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="meta-llama/Llama-3.1-8B")
    p.add_argument("--checkpoint",   default="lb_results_base/checkpoint.json")
    p.add_argument("--data_dir",     default="lb_data_raw/data")
    p.add_argument("--output",       default="lb_results_base/faithfulness_deep.json")
    p.add_argument("--gt_results",   default="lb_results_base/results.json",
                   help="Ground-truth results JSON for † flagging")
    p.add_argument("--tasks",        default=",".join(ENGLISH_TASKS))
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--max_seq_len",  type=int, default=7168)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--perplexity",   action="store_true", default=True)
    p.add_argument("--no_perplexity",action="store_true")
    p.add_argument("--embedding",    action="store_true", default=True)
    p.add_argument("--no_embedding", action="store_true")
    p.add_argument("--embed_model",  default="all-MiniLM-L6-v2",
                   help="Sentence-transformer model name")
    p.add_argument("--reference",    default="full")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in ENGLISH_TASKS]
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    run_perplexity = args.perplexity and not args.no_perplexity
    run_embedding  = args.embedding  and not args.no_embedding

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output     : {args.output}")
    print(f"Tasks      : {len(tasks)}")
    print(f"Perplexity : {run_perplexity}")
    print(f"Embedding  : {run_embedding}")

    with open(args.checkpoint) as f:
        ckpt = json.load(f)

    gt_results = None
    if Path(args.gt_results).exists():
        with open(args.gt_results) as f:
            gt_results = json.load(f)

    # Load any partial results from a previous run
    all_results = load_partial(output_path)
    if all_results:
        sections = [k for k in all_results if k in ("perplexity", "embedding")]
        if sections:
            counts = {s: len([t for t in all_results[s] if not t.startswith("_")]) for s in sections}
            print(f"Resuming from existing output: {counts}")

    # ------------------------------------------------------------------
    # Perplexity-based faithfulness
    # ------------------------------------------------------------------
    if run_perplexity:
        print(f"\nLoading {args.model} for perplexity scoring ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=args.device
        )
        model.eval()

        print("\nPerplexity faithfulness (forward passes only, no generation):")
        ppl_results = compute_perplexity_faithfulness(
            tasks, ckpt, data_dir, model, tokenizer,
            args.max_examples, args.max_seq_len, args.device,
            all_results, output_path, args.reference,
        )
        print_faithfulness_table(ppl_results,
            "Perplexity faithfulness — exp(loss_full − loss_method) × 100\n"
            "  100 = as likely as full-context output; >100 = MORE likely; <100 = less likely",
            gt_results)

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Embedding-based faithfulness
    # ------------------------------------------------------------------
    if run_embedding:
        print("\nEmbedding faithfulness (cosine similarity in sentence-transformer space):")
        emb_results = compute_embedding_faithfulness(
            tasks, ckpt, args.max_examples,
            all_results, output_path, args.reference, args.embed_model,
        )
        print_faithfulness_table(emb_results,
            "Embedding faithfulness — cosine similarity (sentence-transformer)\n"
            "  100 = identical embedding; 50 = orthogonal; rescaled from [-1,1] to [0,100]",
            gt_results)

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

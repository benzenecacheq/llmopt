"""
eval.py
-------
Three-way evaluation harness:
  1. Full context baseline      (standard Llama2, context_len = x)
  2. Naive truncation           (standard Llama2, truncated to x/2 from left)
  3. Per-head importance filter (pruned Llama2, x tokens in, x/2 retained)

Metrics computed per condition:
  - Token-level accuracy vs full-context output (next-token prediction match)
  - BLEU-4 (sacrebleu)
  - ROUGE-1/2/L (rouge-score)
  - Semantic similarity (cosine of sentence embeddings via sentence-transformers)
  - Perplexity on reference text (optional, slower)

Tasks supported:
  - "needle"      : needle-in-a-haystack (delegates to needle_test.py)
  - "qa"          : extractive QA from a JSONL file of {context, question, answer}
  - "summarize"   : summarization from a JSONL file of {document, summary}
  - "icl"         : in-context learning — few-shot examples followed by a query

Usage:
    python eval.py \
        --model  meta-llama/Llama-2-7b-hf \
        --task   qa \
        --data   my_qa.jsonl \
        --context_len 1024 \
        --budget 512 \
        --output eval_results.json

    # Or run the built-in needle test directly:
    python eval.py --task needle --context_len 1024 --budget 512
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Optional metric deps — we import lazily so the harness can be used without all of them
try:
    from sacrebleu.metrics import BLEU
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

from llama_pruned import PrunedLlamaConfig, build_pruned_model
from needle_test import run_needle_test


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def token_accuracy(pred_ids: List[int], ref_ids: List[int]) -> float:
    """Fraction of positions where pred matches ref (truncated to shorter)."""
    n = min(len(pred_ids), len(ref_ids))
    if n == 0:
        return 0.0
    matches = sum(p == r for p, r in zip(pred_ids[:n], ref_ids[:n]))
    return matches / n


def bleu_score(hypothesis: str, reference: str) -> float:
    if not HAS_BLEU:
        return float("nan")
    b = BLEU(effective_order=True)
    return b.sentence_score(hypothesis, [reference]).score / 100.0


def rouge_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    if not HAS_ROUGE:
        return {"rouge1": float("nan"), "rouge2": float("nan"), "rougeL": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: v.fmeasure for k, v in scores.items()}


def semantic_similarity(text_a: str, text_b: str, sbert_model) -> float:
    if not HAS_SBERT or sbert_model is None:
        return float("nan")
    embs = sbert_model.encode([text_a, text_b], convert_to_tensor=True)
    return F.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()


def compute_perplexity(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str,
    max_len: int = 512,
) -> float:
    """Cross-entropy perplexity of `text` under `model`."""
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    ids = ids[:, :max_len]
    with torch.inference_mode():
        out = model(ids, labels=ids)
    return math.exp(out.loss.item())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> tuple[str, List[int]]:
    """Returns (decoded_text, token_id_list)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, new_ids


def naive_truncate(tokenizer: AutoTokenizer, prompt: str, max_len: int) -> str:
    ids = tokenizer.encode(prompt)
    if len(ids) > max_len:
        ids = ids[-max_len:]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_qa(
    records: List[Dict],
    baseline_model, pruned_model, tokenizer,
    context_len: int, budget: int,
    device: str, sbert_model,
    max_new_tokens: int = 64,
) -> Dict:
    """
    Each record: {"context": ..., "question": ..., "answer": ...}
    """
    agg = {m: {"token_acc": [], "bleu": [], "rouge1": [], "rouge2": [],
               "rougeL": [], "sem_sim": []}
           for m in ["full", "naive", "pruned"]}

    for i, rec in enumerate(records):
        prompt = (
            f"Context: {rec['context']}\n\n"
            f"Question: {rec['question']}\n\n"
            f"Answer:"
        )
        reference = rec["answer"]

        # --- Full ---
        text_full, ids_full = generate(baseline_model, tokenizer, prompt,
                                       max_new_tokens, device)
        # --- Naive ---
        naive_prompt = naive_truncate(tokenizer, prompt, budget)
        text_naive, ids_naive = generate(baseline_model, tokenizer, naive_prompt,
                                         max_new_tokens, device)
        # --- Pruned ---
        text_pruned, ids_pruned = generate(pruned_model, tokenizer, prompt,
                                           max_new_tokens, device)

        for method, text, ids in [
            ("full",   text_full,   ids_full),
            ("naive",  text_naive,  ids_naive),
            ("pruned", text_pruned, ids_pruned),
        ]:
            agg[method]["token_acc"].append(token_accuracy(ids, ids_full))
            agg[method]["bleu"].append(bleu_score(text, reference))
            r = rouge_scores(text, reference)
            agg[method]["rouge1"].append(r["rouge1"])
            agg[method]["rouge2"].append(r["rouge2"])
            agg[method]["rougeL"].append(r["rougeL"])
            agg[method]["sem_sim"].append(semantic_similarity(text, reference, sbert_model))

        if (i + 1) % 10 == 0:
            print(f"  QA {i+1}/{len(records)}")

    return _aggregate(agg)


def run_summarize(
    records: List[Dict],
    baseline_model, pruned_model, tokenizer,
    context_len: int, budget: int,
    device: str, sbert_model,
    max_new_tokens: int = 128,
) -> Dict:
    """
    Each record: {"document": ..., "summary": ...}
    """
    agg = {m: {"token_acc": [], "bleu": [], "rouge1": [], "rouge2": [],
               "rougeL": [], "sem_sim": []}
           for m in ["full", "naive", "pruned"]}

    for i, rec in enumerate(records):
        prompt = f"Summarize the following:\n\n{rec['document']}\n\nSummary:"
        reference = rec["summary"]

        text_full,   ids_full   = generate(baseline_model, tokenizer, prompt,
                                           max_new_tokens, device)
        naive_p = naive_truncate(tokenizer, prompt, budget)
        text_naive,  ids_naive  = generate(baseline_model, tokenizer, naive_p,
                                           max_new_tokens, device)
        text_pruned, ids_pruned = generate(pruned_model,   tokenizer, prompt,
                                           max_new_tokens, device)

        for method, text, ids in [
            ("full",   text_full,   ids_full),
            ("naive",  text_naive,  ids_naive),
            ("pruned", text_pruned, ids_pruned),
        ]:
            agg[method]["token_acc"].append(token_accuracy(ids, ids_full))
            agg[method]["bleu"].append(bleu_score(text, reference))
            r = rouge_scores(text, reference)
            agg[method]["rouge1"].append(r["rouge1"])
            agg[method]["rouge2"].append(r["rouge2"])
            agg[method]["rougeL"].append(r["rougeL"])
            agg[method]["sem_sim"].append(semantic_similarity(text, reference, sbert_model))

        if (i + 1) % 10 == 0:
            print(f"  Summarize {i+1}/{len(records)}")

    return _aggregate(agg)


def _aggregate(agg: Dict) -> Dict:
    out = {}
    for method, metrics in agg.items():
        out[method] = {k: (sum(v) / len(v) if v else float("nan"))
                       for k, v in metrics.items()}
    return out


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_results(results: Dict, task: str):
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")
    metric_keys = ["token_acc", "bleu", "rouge1", "rouge2", "rougeL", "sem_sim"]
    header = f"{'Method':<10}" + "".join(f"{k:>12}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for method in ["full", "naive", "pruned"]:
        if method not in results:
            continue
        row = f"{method:<10}"
        for k in metric_keys:
            v = results[method].get(k, float("nan"))
            row += f"{v:>12.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Three-way eval harness")
    parser.add_argument("--model",       default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--task",        default="needle",
                        choices=["needle", "qa", "summarize"])
    parser.add_argument("--data",        default=None,
                        help="Path to JSONL data file (required for qa/summarize)")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--budget",      type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--budget_strategy", default="entropy",
                        choices=["equal", "entropy"])
    parser.add_argument("--decay_fn",    default="exponential",
                        choices=["linear", "exponential"])
    parser.add_argument("--decay_rate",  type=float, default=None,
                        help="Distance decay lambda. Defaults to ln(2)/(context_len/2) "
                             "so ~50%% decay at the sequence midpoint.")
    parser.add_argument("--q_buffer",    type=int, default=64)
    parser.add_argument("--always_keep", type=int, default=16)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default="eval_results.json")
    parser.add_argument("--use_sbert",   action="store_true",
                        help="Load sentence-transformers for semantic similarity")
    args = parser.parse_args()

    decay_rate = args.decay_rate if args.decay_rate is not None \
        else math.log(2) / (args.context_len / 2)

    pcfg = PrunedLlamaConfig(
        total_budget=args.budget,
        q_buffer_size=args.q_buffer,
        budget_strategy=args.budget_strategy,
        decay_fn=args.decay_fn,
        decay_rate=decay_rate,
        always_keep_last=args.always_keep,
    )

    # Needle delegates entirely to needle_test.py
    if args.task == "needle":
        results = run_needle_test(
            model_name=args.model,
            context_len=args.context_len,
            budget=args.budget,
            num_trials=args.num_samples,
            pruned_cfg=pcfg,
            device=args.device,
            seed=args.seed,
            output_path=args.output,
        )
        return

    # For QA / summarize we need both models
    print("Loading baseline model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    baseline_model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    baseline_model.eval()

    print("Loading pruned model ...")
    pruned_model, _ = build_pruned_model(args.model, pcfg, device=args.device)

    sbert = None
    if args.use_sbert and HAS_SBERT:
        print("Loading sentence-transformers ...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

    if args.data is None:
        raise ValueError(f"--data is required for task={args.task}")

    records = load_jsonl(args.data)
    records = records[: args.num_samples]
    print(f"Loaded {len(records)} records from {args.data}")

    if args.task == "qa":
        results = run_qa(records, baseline_model, pruned_model, tokenizer,
                         args.context_len, args.budget, args.device, sbert)
    elif args.task == "summarize":
        results = run_summarize(records, baseline_model, pruned_model, tokenizer,
                                args.context_len, args.budget, args.device, sbert)

    print_results(results, args.task)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

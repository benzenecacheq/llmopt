"""
needle_test.py
--------------
Needle-in-a-haystack evaluation.

Plants a single key fact at a controlled position early in a long context,
then asks the model to retrieve it. Measures retrieval accuracy across:

  - Full context (ground truth)
  - Naive head truncation (drop oldest tokens)
  - Per-head importance filter (your method)

Results are reported per needle-position bucket (early / middle / late)
and written to needle_results.json.

Usage:
    python needle_test.py \
        --model meta-llama/Llama-2-7b-hf \
        --context_len 1024 \
        --num_trials 50 \
        --budget 512 \
        --output needle_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from llama_pruned import PrunedLlamaConfig, build_pruned_model


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------

# Varied business-document sentences so the haystack is realistic and
# non-repetitive.  The needle is a plausible fact that could appear in such
# a document, making retrieval genuinely challenging.
HAYSTACK_SENTENCES = [
    "Please review the attached documentation before the next meeting. ",
    "All team members are required to complete the compliance training module. ",
    "The quarterly report has been submitted to the finance department for review. ",
    "Remember to update your contact information in the HR system by Friday. ",
    "The project deadline has been extended by two weeks due to scope changes. ",
    "Please ensure all expense reports are submitted before the end of the month. ",
    "The new policy takes effect on the first day of the upcoming quarter. ",
    "Attendance at the weekly team sync is mandatory for all staff members. ",
    "Please reply to this message to confirm receipt of the updated schedule. ",
    "All change requests must be approved by your direct manager before implementation. ",
    "The server maintenance window is scheduled for this weekend from midnight to 4am. ",
    "Employees working remotely must be available during core business hours. ",
    "The budget for the next fiscal year has been approved by the board. ",
    "Please submit your time sheets no later than noon on the last business day. ",
    "The client presentation has been moved to the large conference room on the third floor. ",
    "All software licenses must be renewed through the IT procurement process. ",
    "The onboarding materials for new hires have been updated on the internal portal. ",
    "Please coordinate with your counterpart on the partner team before the release. ",
    "The annual performance review cycle begins at the start of next month. ",
    "Security badges must be worn visibly at all times while on company premises. ",
]

NEEDLE_TEMPLATE = (
    "Note: the authorization code for {system} has been set to {code}. "
    "Please store this securely."
)

SYSTEMS = [
    "the backup server", "the VPN gateway", "the document vault",
    "the staging environment", "the release pipeline", "the audit log",
    "the data warehouse", "the configuration manager", "the deployment system",
    "the monitoring dashboard",
]

QUESTION_TEMPLATE = (
    "Question: According to the notes above, what is the authorization code for {system}?\n"
    "Answer: The authorization code is "
)


def random_code() -> str:
    """Random 4-digit numeric code — common and plausible, unlike gibberish."""
    return str(random.randint(1000, 9999))


def build_prompt(
    tokenizer: AutoTokenizer,
    context_len: int,
    needle_position_frac: float,   # 0.0 = very start, 1.0 = just before question
) -> tuple[str, str]:
    """
    Returns (full_prompt, target_code).
    Builds a prompt of approximately context_len tokens with the needle
    planted at needle_position_frac * context_len.
    The haystack is sampled from varied business-document sentences so it
    looks realistic and is non-repetitive.
    """
    system = random.choice(SYSTEMS)
    target = random_code()
    needle   = NEEDLE_TEMPLATE.format(system=system, code=target)
    question = QUESTION_TEMPLATE.format(system=system)

    # Estimate token counts
    needle_toks = len(tokenizer.encode(needle))
    q_toks      = len(tokenizer.encode(question))

    # Build haystack by cycling through varied sentences
    total_hay_budget = context_len - needle_toks - q_toks - 10
    hay_sentences: List[str] = []
    hay_len = 0
    idx = 0
    while True:
        sent = HAYSTACK_SENTENCES[idx % len(HAYSTACK_SENTENCES)]
        sent_toks = len(tokenizer.encode(sent))
        if hay_len + sent_toks > total_hay_budget:
            break
        hay_sentences.append(sent)
        hay_len += sent_toks
        idx += 1

    needle_pos = int(len(hay_sentences) * needle_position_frac)
    pre_hay  = "".join(hay_sentences[:needle_pos])
    post_hay = "".join(hay_sentences[needle_pos:])

    prompt = pre_hay + needle + " " + post_hay + "\n\n" + question
    return prompt, target


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_answer(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    device: str = "cuda",
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_code(answer: str) -> str:
    """Pull out the first 4-digit number from the model's answer."""
    m = re.search(r"\b\d{4}\b", answer)
    return m.group(0) if m else answer.strip()


def truncate_naive(
    tokenizer: AutoTokenizer,
    prompt: str,
    max_len: int,
) -> str:
    """Truncate from the left (drop oldest tokens) to max_len."""
    ids = tokenizer.encode(prompt)
    if len(ids) > max_len:
        ids = ids[-max_len:]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_needle_test(
    model_name: str,
    context_len: int,
    budget: int,
    num_trials: int,
    pruned_cfg: PrunedLlamaConfig,
    device: str = "cuda",
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)

    # Position buckets
    position_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    bucket_labels  = ["very_early", "early", "middle", "late", "very_late"]

    # Load models
    print("Loading baseline (full context) model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    baseline_model.eval()

    print("Loading pruned model ...")
    pruned_model, _ = build_pruned_model(model_name, pruned_cfg, device=device)

    results = {
        "config": {
            "model": model_name,
            "context_len": context_len,
            "budget": budget,
            "num_trials": num_trials,
            "budget_strategy": pruned_cfg.budget_strategy,
            "decay_fn": pruned_cfg.decay_fn,
            "decay_rate": pruned_cfg.decay_rate,
        },
        "by_position": {},
        "aggregate": {},
    }

    for frac, label in zip(position_fracs, bucket_labels):
        print(f"\n--- Needle position: {label} ({frac:.0%} into context) ---")
        counts = {"full": 0, "naive": 0, "pruned": 0, "total": 0}

        for trial in range(num_trials):
            prompt, target = build_prompt(tokenizer, context_len, frac)

            # --- Full context ---
            ans_full = generate_answer(baseline_model, tokenizer, prompt,
                                       device=device)
            correct_full = extract_code(ans_full) == target

            # --- Naive truncation ---
            naive_prompt = truncate_naive(tokenizer, prompt, budget)
            ans_naive = generate_answer(baseline_model, tokenizer, naive_prompt,
                                        device=device)
            correct_naive = extract_code(ans_naive) == target

            # --- Pruned model ---
            ans_pruned = generate_answer(pruned_model, tokenizer, prompt,
                                         device=device)
            correct_pruned = extract_code(ans_pruned) == target

            counts["full"]   += int(correct_full)
            counts["naive"]  += int(correct_naive)
            counts["pruned"] += int(correct_pruned)
            counts["total"]  += 1

            if (trial + 1) % 10 == 0:
                print(f"  Trial {trial+1}/{num_trials} | "
                      f"full={counts['full']} naive={counts['naive']} "
                      f"pruned={counts['pruned']}")

        n = counts["total"]
        results["by_position"][label] = {
            "needle_frac": frac,
            "full_accuracy":   counts["full"]   / n,
            "naive_accuracy":  counts["naive"]  / n,
            "pruned_accuracy": counts["pruned"] / n,
            "trials": n,
        }

        print(f"  Results: full={counts['full']/n:.2%} "
              f"naive={counts['naive']/n:.2%} "
              f"pruned={counts['pruned']/n:.2%}")

    # Aggregate across positions
    all_pos = results["by_position"].values()
    for method in ["full", "naive", "pruned"]:
        acc = sum(p[f"{method}_accuracy"] for p in all_pos) / len(position_fracs)
        results["aggregate"][f"{method}_accuracy"] = acc

    print("\n=== Aggregate Results ===")
    for method in ["full", "naive", "pruned"]:
        print(f"  {method:8s}: {results['aggregate'][f'{method}_accuracy']:.2%}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-haystack test")
    parser.add_argument("--model",       default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--budget",      type=int, default=512,
                        help="Target retained token count (= context_len/2)")
    parser.add_argument("--num_trials",  type=int, default=50)
    parser.add_argument("--budget_strategy", default="entropy",
                        choices=["equal", "entropy"])
    parser.add_argument("--decay_fn",    default="linear",
                        choices=["linear", "exponential"])
    parser.add_argument("--decay_rate",  type=float, default=0.0,
                        help="Explicit decay rate. Leave 0 to auto-compute from --min_decay.")
    parser.add_argument("--min_decay",   type=float, default=0.7,
                        help="Decay value at position 0 (oldest token). "
                             "Rate is derived as (1-min_decay)/(T-1) for linear "
                             "or -ln(min_decay)/(T-1) for exponential. "
                             "Ignored when --decay_rate > 0.")
    parser.add_argument("--q_buffer",      type=int, default=64)
    parser.add_argument("--always_keep",   type=int, default=16)
    parser.add_argument("--always_keep_first", type=int, default=16)
    parser.add_argument("--score_mode", default="kq_vn_decay",
                        choices=["kq_vn_decay", "kq_only", "vn_only", "vn_decay"])
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default="needle_results.json")
    args = parser.parse_args()

    pcfg = PrunedLlamaConfig(
        total_budget=args.budget,
        q_buffer_size=args.q_buffer,
        budget_strategy=args.budget_strategy,
        decay_fn=args.decay_fn,
        decay_rate=args.decay_rate,
        min_decay=args.min_decay,
        always_keep_last=args.always_keep,
        always_keep_first=args.always_keep_first,
        score_mode=args.score_mode,
    )

    run_needle_test(
        model_name=args.model,
        context_len=args.context_len,
        budget=args.budget,
        num_trials=args.num_trials,
        pruned_cfg=pcfg,
        device=args.device,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

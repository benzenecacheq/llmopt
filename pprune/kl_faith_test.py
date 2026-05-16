"""
kl_faith_test.py
----------------
Quick sanity check for KL divergence faithfulness.

For each example:
  1. Patch the model as compressed, generate with output_scores=True to capture
     P_compressed at each step.  Scores moved to CPU immediately.
  2. Restore the original attention layers (same weights, no second load).
  3. Teacher-forced forward on [full_context + compressed_tokens] to get P_full.
  4. Compute mean KL(P_full || P_compressed) over generation steps.

Uses 5 hotpotqa examples by default (short context, 32 max new tokens — fast).

Usage:
    python kl_faith_test.py
    python kl_faith_test.py --task samsum --n 10
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_pruned import PrunedLlamaAttention, PrunedLlamaConfig, HeadFilterConfig, PerHeadFilter

MODEL        = "meta-llama/Llama-3.1-8B"
DATA_DIR     = Path("lb_data_raw/data")
DEVICE       = "cuda"
MAX_PROMPT_TOKENS = 2048   # truncate prompt from left to stay within GPU memory

DATASET2PROMPT = {
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
}
DATASET2MAXNEW = {"hotpotqa": 32, "samsum": 128, "triviaqa": 32}


def patch_model(model, pcfg, device):
    """Swap all LlamaAttention layers for PrunedLlamaAttention.  Returns originals."""
    import transformers as _t
    from transformers.models.llama.modeling_llama import LlamaAttention
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


def run_kl_test(task: str, n: int, model_name: str):
    data_file = DATA_DIR / f"{task}.jsonl"
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    with open(data_file) as f:
        examples = [json.loads(l) for l in f if l.strip()][:n]

    template = DATASET2PROMPT[task]
    max_new = DATASET2MAXNEW[task]

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    pcfg = PrunedLlamaConfig(
        score_mode="kq_post_rope",
        budget_fraction=0.65,
        min_decay=0.7,
        always_keep_first=16,
        always_keep_last=16,
        q_buffer_size=128,
    )

    kl_scores = []

    for i, ex in enumerate(examples):
        prompt = template.format(
            context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
            input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > MAX_PROMPT_TOKENS:
            input_ids = input_ids[:, -MAX_PROMPT_TOKENS:]
        inputs = {"input_ids": input_ids.to(DEVICE)}
        n_prompt = input_ids.shape[1]

        # --- Step 1: compressed generation, capture per-step distributions ---
        originals = patch_model(model, pcfg, DEVICE)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        unpatch_model(model, originals)

        # out.scores: tuple of (1, vocab) tensors, one per generated token
        comp_logits = torch.stack([s[0] for s in out.scores], dim=0).cpu()  # (n_gen, vocab)
        gen_tokens  = out.sequences[0, n_prompt:].cpu()                     # (n_gen,)
        n_gen = gen_tokens.shape[0]
        if n_gen == 0:
            print(f"  [{i}] empty generation, skipping")
            continue

        # --- Step 2: full-model teacher-forced forward ---
        full_input = torch.cat([
            input_ids.to(DEVICE),
            gen_tokens.unsqueeze(0).to(DEVICE),
        ], dim=1)

        with torch.inference_mode():
            full_out = model(full_input)

        # Logits at positions [n_prompt-1 .. -1] predict tokens [n_prompt .. end]
        full_logits = full_out.logits[0, n_prompt - 1 : -1, :].float()  # (n_gen, vocab)

        # --- Step 3: KL(P_full || P_compressed) per generation step ---
        log_p_full = F.log_softmax(full_logits, dim=-1)
        log_p_comp = F.log_softmax(comp_logits.to(DEVICE).float(), dim=-1)
        p_full     = log_p_full.exp()

        kl_per_step = (p_full * (log_p_full - log_p_comp)).sum(dim=-1)  # (n_gen,)
        mean_kl     = kl_per_step.mean().item()
        kl_scores.append(mean_kl)

        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"  [{i}] KL={mean_kl:.4f}  gen={repr(gen_text[:60])}")

    if kl_scores:
        grand_mean = sum(kl_scores) / len(kl_scores)
        print(f"\nMean KL(P_full || P_compressed) over {len(kl_scores)} examples: {grand_mean:.4f}")
        print(f"  (nats; 0 = identical distributions, ln2≈0.693 = max JS)")
        # Sanity: KL of full model against itself should be ~0
        print("\nRunning self-KL sanity check (expect ~0) on first example ...")
        ex = examples[0]
        prompt = template.format(
            context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
            input=ex.get("input", "").replace("NEWLINE_CHAR", "\n"),
        )
        input_ids_s = tokenizer.encode(prompt, return_tensors="pt")
        if input_ids_s.shape[1] > MAX_PROMPT_TOKENS:
            input_ids_s = input_ids_s[:, -MAX_PROMPT_TOKENS:]
        n_prompt = input_ids_s.shape[1]
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids_s.to(DEVICE), max_new_tokens=max_new, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        full_logits_gen  = torch.stack([s[0] for s in out.scores], dim=0).float()  # (n, v) on GPU
        gen_tokens       = out.sequences[0, n_prompt:]
        full_input       = out.sequences
        with torch.inference_mode():
            full_out2 = model(full_input)
        full_logits_tf   = full_out2.logits[0, n_prompt - 1 : -1, :].float()
        log_p1 = F.log_softmax(full_logits_gen, dim=-1)
        log_p2 = F.log_softmax(full_logits_tf,  dim=-1)
        p1     = log_p1.exp()
        self_kl = (p1 * (log_p1 - log_p2)).sum(dim=-1).mean().item()
        print(f"  self-KL = {self_kl:.6f}  (should be ~0; non-zero only from fp16 rounding)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",  default="samsum", choices=list(DATASET2PROMPT))
    p.add_argument("--n",     type=int, default=5)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    run_kl_test(args.task, args.n, args.model)


if __name__ == "__main__":
    main()

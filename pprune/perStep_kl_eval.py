"""
perStep_kl_eval.py
------------------
Per-step KL faithfulness evaluation using y* shared prefix.

Records KL(P_full || P_comp) and token-match at each generation step t,
rather than averaging. Designed to test the T0-advantage hypothesis:
PyramidKV's T0 KL should be ~0 (full prefill) while T1+ should rise
rapidly (pruned KV cache).

Output checkpoint: {task}|{method}|{idx} ->
    {"kl_steps": [float, ...], "match_steps": [bool, ...], "n_gen": int,
     "fout": float, "fout_early": float}

Usage:
    # Filtered LongBench (long-answer tasks only, reuse y* cache):
    python perStep_kl_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --tasks gov_report,multi_news,qmsum \\
        --methods naive_65pct,chunk_word128_t20,snapkv,pyramidkv \\
        --output lb_results_base/perstep_kl_longbench.json \\
        --ystar_cache lb_results_base/ystar_cache_v3.pt \\
        --min_gen_tokens 50 --n 50

    # Synthetic long-answer prompts:
    python perStep_kl_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --data_dir synthetic_prompts \\
        --tasks synthetic_long \\
        --methods naive_65pct,chunk_word128_t20,snapkv,pyramidkv \\
        --output lb_results_base/perstep_kl_synthetic.json \\
        --max_new_tokens 300
"""

from __future__ import annotations

import argparse
import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from longbench_eval import qa_f1_score

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
    load_ckpt,
    save_ckpt,
    fmt_duration,
    YSTAR_STOP_TOKENS,
)
from llama_pruned import PrunedLlamaConfig
from kl_faith_eval_ystar import (
    _KVPRESS_METHODS,
    _SELECT_METHODS,
    make_comp_ids,
    generate_ystar,
    get_comp_log_probs,
    get_full_hidden_state,
    get_comp_hidden_state,
    embed_text,
    cosine_sim,
    load_ystar_cache,
    save_ystar_cache,
    get_from_cache,
    put_in_cache,
)

# Default template for custom (non-LongBench) tasks.
# The JSONL should have "context" (the passage) and "input" (the question).
CUSTOM_TASK_TEMPLATE = "{context}\n\n{input}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Per-step metrics
# ---------------------------------------------------------------------------

def kl_per_step(log_p_full: torch.Tensor, log_p_comp: torch.Tensor) -> List[float]:
    """KL(P_full || P_comp) at each generation step. Returns list of length n_gen."""
    p_full = log_p_full.exp()
    kl_steps = (p_full * (log_p_full - log_p_comp)).sum(dim=-1)  # (n_gen,)
    return kl_steps.tolist()


def match_per_step(ystar: torch.Tensor, log_p_comp: torch.Tensor) -> List[bool]:
    """Whether the compressed model's argmax matches y*[t] at each step t."""
    pred = log_p_comp.argmax(dim=-1)  # (n_gen,)
    return (pred == ystar).tolist()


def nll_ref_per_step(ystar: torch.Tensor, log_p: torch.Tensor) -> List[float]:
    """-log P(y*_t | y*_<t}) at each step, teacher-forced (no free generation).

    Same regardless of whether log_p comes from the full or compressed model;
    used for both so the two are directly comparable (excess perplexity = their ratio).
    """
    nll = -log_p.gather(-1, ystar.unsqueeze(-1)).squeeze(-1)  # (n_gen,)
    return nll.tolist()


# ---------------------------------------------------------------------------
# F_out: free generation + word F1 scoring
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_from_compressed(
    model,
    comp_ids: torch.Tensor,
    n_tokens: int,
    device: str,
    pcfg=None,
    press=None,
) -> Optional[torch.Tensor]:
    """Free-decode from the compressed model. Returns int64 (n_gen,) or None on OOM.

    comp_ids should be the same tensor produced by make_comp_ids for this method:
    - Prompt-construction (naive, phr128): the shortened prompt
    - KV-pruning (snapkv, pyramidkv): the full-length prompt (pruning fires inside)
    """
    input_ids = comp_ids.to(device)
    originals = None
    try:
        if pcfg is not None:
            originals = patch_model(model, pcfg, device)
        ctx = press(model) if press is not None else nullcontext()
        with ctx:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=n_tokens,
                do_sample=False,
            )
        return out[0, input_ids.shape[1]:].cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        if originals is not None:
            unpatch_model(model, originals)
        torch.cuda.empty_cache()


def compute_fout(
    gen_tokens: torch.Tensor,
    ystar: torch.Tensor,
    tokenizer,
    early_n: int = 20,
) -> tuple:
    """Return (overall_fout, early_fout) as word-F1 scores in [0, 1].

    overall_fout: word F1 between full generation and full y*
    early_fout:   word F1 between first early_n tokens of each
    """
    gen_text  = tokenizer.decode(gen_tokens,          skip_special_tokens=True)
    ref_text  = tokenizer.decode(ystar,               skip_special_tokens=True)
    overall   = qa_f1_score(gen_text, ref_text)

    n = min(early_n, len(gen_tokens), len(ystar))
    early_gen = tokenizer.decode(gen_tokens[:n], skip_special_tokens=True)
    early_ref = tokenizer.decode(ystar[:n],      skip_special_tokens=True)
    early     = qa_f1_score(early_gen, early_ref)

    return overall, early


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
    min_gen_tokens: int = 0,
    early_n: int = 20,
):
    ckpt = load_ckpt(output_path)
    cache = load_ystar_cache(ystar_cache_path) if ystar_cache_path else {}
    if ystar_cache_path:
        print(f"y* cache: {ystar_cache_path}  ({len(cache)} entries)")

    def _is_done(key):
        if key not in ckpt:
            return False
        v = ckpt[key]
        return v is None or (isinstance(v, dict) and "kl_steps" in v)

    n_tasks   = len(tasks)
    n_methods = len(methods)
    total_work = n_tasks * max_examples * n_methods
    completed_at_start = sum(1 for k, v in ckpt.items() if _is_done(k))

    print(f"\n{'='*72}")
    print(f"Per-step KL + F_out faithfulness (y* shared prefix)")
    print(f"  Tasks  : {n_tasks}  ({', '.join(tasks[:4])}{'...' if n_tasks > 4 else ''})")
    print(f"  Methods: {n_methods}  ({', '.join(methods)})")
    print(f"  N/task : {max_examples}  (min_gen_tokens={min_gen_tokens}  early_n={early_n})")
    print(f"  Total  : {total_work} entries  ({completed_at_start} already done)")
    print(f"{'='*72}\n")

    grand_done  = completed_at_start
    grand_start = time.time()

    for t_idx, task in enumerate(tasks):
        # Look for JSONL in data_dir; fall back to task.jsonl directly
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            data_file = data_dir / task / "test.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file at {data_file}")
            continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        # Use LongBench prompt template if known task; generic template otherwise
        if task in DATASET2PROMPT:
            template  = DATASET2PROMPT[task]
            max_new   = max_new_tokens_override or DATASET2MAXLEN.get(task, 64)
            max_seq_comp = TASK_MAX_SEQ_COMP.get(task, DEFAULT_MAX_SEQ_COMP)
            max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
            stop_toks = YSTAR_STOP_TOKENS.get(task)
        else:
            template     = CUSTOM_TASK_TEMPLATE
            max_new      = max_new_tokens_override or 300
            max_seq_comp = DEFAULT_MAX_SEQ_COMP
            max_seq_full = DEFAULT_MAX_SEQ_FULL
            stop_toks    = None

        n_ex = min(max_examples, len(dataset))
        print(f"[{task}  {t_idx+1}/{n_tasks}]  n={n_ex}  max_new={max_new}")

        skipped_short = 0

        for idx in range(n_ex):
            if all(_is_done(f"{task}|{m}|{idx}") for m in methods):
                grand_done += n_methods
                continue

            ex = dataset[idx]
            question_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            context_text  = ex.get("context", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(context=context_text, input=question_text)

            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

            # Generate y* (or load from cache)
            ystar, log_p_full = get_from_cache(cache, task, idx)
            if ystar is None:
                ystar, log_p_full, _ = generate_ystar(
                    model, full_ids, max_new, max_seq_full, device,
                    stop_token_ids=stop_toks,
                )
                if ystar is not None and ystar_cache_path:
                    put_in_cache(cache, task, idx, ystar, log_p_full)
                    save_ystar_cache(cache, ystar_cache_path)
                    print(f"  [cached y* {task}|{idx}  n_gen={ystar.shape[0]}]", flush=True)
            else:
                print(f"  [y* cache hit: {task}|{idx}  n_gen={ystar.shape[0]}]", flush=True)

            if ystar is None or ystar.shape[0] < max(1, min_gen_tokens):
                if ystar is not None:
                    skipped_short += 1
                for m in methods:
                    ckpt.setdefault(f"{task}|{m}|{idx}", None)
                save_ckpt(ckpt, output_path)
                grand_done += n_methods
                continue

            n_gen = ystar.shape[0]

            # Full-model reference embeddings, computed once per example:
            #   full_hidden   — teacher-forced hidden state through [full_ids + ystar],
            #                   for representational similarity (no free generation).
            #   emb_full_text — neutral embedding of y* read in isolation, for
            #                   semantic similarity against each method's free generation.
            full_hidden   = get_full_hidden_state(model, full_ids, ystar, device, max_seq_full)
            emb_full_text = embed_text(model, ystar, device)

            for method in methods:
                ck_key = f"{task}|{method}|{idx}"
                if _is_done(ck_key):
                    grand_done += 1
                    continue

                pcfg  = METHOD_CONFIGS.get(method)
                press = _KVPRESS_METHODS.get(method)
                effective_max = max_seq_full if press is not None else max_seq_comp

                comp_ids = make_comp_ids(
                    full_ids, method, tokenizer, question_text,
                    effective_max, model=model, device=device,
                )

                log_p_comp = get_comp_log_probs(
                    model, comp_ids, ystar, pcfg, device, press=press,
                )
                comp_hidden = get_comp_hidden_state(
                    model, comp_ids, ystar, pcfg, device, press=press,
                )
                emb_sim_tf = cosine_sim(full_hidden, comp_hidden)

                if log_p_comp is not None:
                    kl_steps     = kl_per_step(log_p_full, log_p_comp)
                    match_steps  = match_per_step(ystar, log_p_comp)
                    nll_comp_ref = nll_ref_per_step(ystar, log_p_comp)
                    nll_full_ref = nll_ref_per_step(ystar, log_p_full)
                    mean_kl      = float(np.mean(kl_steps))
                    match_t0     = match_steps[0] if match_steps else None
                else:
                    kl_steps = match_steps = nll_comp_ref = nll_full_ref = None
                    mean_kl = match_t0 = None

                # F_out: free generation then word F1 against y*
                gen_tokens = generate_from_compressed(
                    model, comp_ids, n_gen + 10, device, pcfg=pcfg, press=press,
                )
                if gen_tokens is not None and len(gen_tokens) > 0:
                    fout, fout_early = compute_fout(gen_tokens, ystar, tokenizer, early_n)
                    emb_comp_text = embed_text(model, gen_tokens, device)
                    emb_sim_free  = cosine_sim(emb_full_text, emb_comp_text)
                else:
                    fout = fout_early = None
                    emb_sim_free = None

                if kl_steps is not None:
                    ckpt[ck_key] = {
                        "kl_steps":     kl_steps,
                        "match_steps":  match_steps,
                        "nll_comp_ref": nll_comp_ref,
                        "nll_full_ref": nll_full_ref,
                        "n_gen":        n_gen,
                        "fout":         fout,
                        "fout_early":   fout_early,
                        "early_n":      early_n,
                        "emb_sim_tf":   emb_sim_tf,
                        "emb_sim_free": emb_sim_free,
                    }
                else:
                    ckpt[ck_key] = None

                save_ckpt(ckpt, output_path)
                grand_done += 1

                elapsed   = time.time() - grand_start
                rate      = (grand_done - completed_at_start) / max(elapsed, 1)
                remaining = (total_work - grand_done) / max(rate, 1e-9)
                kl_str    = f"{mean_kl:.4f}" if mean_kl is not None else " OOM"
                f_str     = f"fout={fout:.3f}" if fout is not None else "fout=OOM"
                fe_str    = f"fe={fout_early:.3f}" if fout_early is not None else ""
                t0_str    = f"t0={'Y' if match_t0 else 'N'}" if match_t0 is not None else ""
                etf_str   = f"sim_tf={emb_sim_tf:.3f}" if emb_sim_tf is not None else ""
                esf_str   = f"sim_free={emb_sim_free:.3f}" if emb_sim_free is not None else ""
                print(
                    f"  ex {idx+1:>4}/{n_ex}  {method:<26}"
                    f"  n_gen={n_gen:>4}  KL={kl_str}  {f_str}  {fe_str}  {t0_str}  {etf_str}  {esf_str}"
                    f"  [ETA {fmt_duration(remaining)}]",
                    flush=True,
                )

        if skipped_short:
            print(f"  Skipped {skipped_short} examples with < {min_gen_tokens} generated tokens")

    total_elapsed = time.time() - grand_start
    print(f"\nTotal wall time: {fmt_duration(total_elapsed)}")
    print(f"Results saved → {output_path}")


# ---------------------------------------------------------------------------
# Summary / quick analysis
# ---------------------------------------------------------------------------

def print_summary(output_path: Path, tasks: List[str], methods: List[str], max_examples: int):
    ckpt = load_ckpt(output_path)

    print(f"\n{'='*72}")
    print(f"PER-STEP SUMMARY — {output_path}")
    print(f"{'='*72}")
    print(f"{'Method':<26}  {'n_ex':>5}  {'KL_mean':>8}  {'KL_t0':>7}  {'KL_t5':>7}"
          f"  {'t0_match%':>10}  {'F_out':>7}  {'F_early':>8}")
    print("-" * 86)

    for m in methods:
        all_kl_steps    = []
        all_match_steps = []
        all_fout        = []
        all_fout_early  = []
        n_included = 0

        for task in tasks:
            for idx in range(max_examples):
                v = ckpt.get(f"{task}|{m}|{idx}")
                if not isinstance(v, dict):
                    continue
                all_kl_steps.append(v["kl_steps"])
                all_match_steps.append(v["match_steps"])
                if v.get("fout") is not None:
                    all_fout.append(v["fout"])
                if v.get("fout_early") is not None:
                    all_fout_early.append(v["fout_early"])
                n_included += 1

        if not all_kl_steps:
            print(f"{m:<26}  {'—':>5}")
            continue

        max_t = max(len(s) for s in all_kl_steps)
        mean_kl_t = [np.mean([s[t] for s in all_kl_steps if len(s) > t]) for t in range(max_t)]

        mean_kl = np.mean([np.mean(s) for s in all_kl_steps])
        t0_kl   = mean_kl_t[0] if mean_kl_t else float("nan")
        t5_kl   = mean_kl_t[5] if len(mean_kl_t) > 5 else float("nan")
        t0_match = np.mean([s[0] for s in all_match_steps if s]) * 100
        fout_str  = f"{np.mean(all_fout)*100:>6.1f}%" if all_fout else f"{'—':>7}"
        fearly_str = f"{np.mean(all_fout_early)*100:>7.1f}%" if all_fout_early else f"{'—':>8}"

        print(
            f"{m:<26}  {n_included:>5}  {mean_kl:>8.4f}"
            f"  {t0_kl:>7.4f}  {t5_kl:>7.4f}"
            f"  {t0_match:>9.1f}%  {fout_str}  {fearly_str}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path: Path):
        self._file   = open(path, "a")
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
    p.add_argument("--output",         required=True)
    p.add_argument("--log",            default=None)
    p.add_argument("--tasks",          required=True)
    p.add_argument("--methods",        default="naive_65pct,chunk_word128_t20,snapkv,pyramidkv")
    p.add_argument("--n",              type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--min_gen_tokens", type=int, default=0,
                   help="Skip examples where full model generates fewer than this many tokens")
    p.add_argument("--early_n",        type=int, default=20,
                   help="Token count for early F_out window (default: 20)")
    p.add_argument("--ystar_cache",    default=None,
                   help="Path to y* cache .pt file (shared with kl_faith_eval_ystar.py)")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--summary_only",   action="store_true",
                   help="Print summary of existing results without running eval")
    return p.parse_args()


def main():
    args = parse_args()

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    _all_known = (set(METHOD_CONFIGS) | set(_SELECT_METHODS) |
                  set(CHUNK_CONFIGS) | set(_KVPRESS_METHODS))

    tasks   = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in _all_known]

    if not tasks:
        print("No tasks specified."); return
    if not methods:
        print(f"No valid methods in: {args.methods}"); return

    output_path = Path(args.output)

    if args.summary_only:
        print_summary(output_path, tasks, methods, args.n)
        return

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    run_eval(
        model, tokenizer,
        tasks=tasks,
        methods=methods,
        data_dir=Path(args.data_dir),
        output_path=output_path,
        max_examples=args.n,
        device=args.device,
        max_new_tokens_override=args.max_new_tokens,
        ystar_cache_path=Path(args.ystar_cache) if args.ystar_cache else None,
        min_gen_tokens=args.min_gen_tokens,
        early_n=args.early_n,
    )

    print_summary(output_path, tasks, methods, args.n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combined Option A + B logit eval using the gt_eval-capped prompt.

For each (task, example):
  1. Run full model freely with output_scores=True.
     → tokens  = y* (generated under the CORRECT, gt_eval-capped prompt)
     → scores  = log_p_full at each step (along the full model's own path)

  2. For each compressed method, TWO passes:

     Option B — free generation:
       Run compressed model freely with output_scores=True.
       Compare step-by-step while contexts are still aligned (same prior tokens).
       Records: argmax_agree, top-5/10 overlap, KL, EM.

     Option A — teacher-forced through y*:
       Prefill compressed model with prompt → step through y* tokens one by one.
       Compare log_p_comp_TF vs log_p_full at each step.
       Records: KL_TF per step (= proper KL faithfulness at correct prompt length).

Both share the single full-model run per example.

Output: lb_results_base/free_gen_logits.pt
  Dict keyed "task|method|idx":
    free-gen (Option B):
      full_toks, comp_toks, full_pred, comp_pred, em,
      step_agree, step_diverged, step_overlap5, step_overlap10, step_kl_fg
    teacher-forced (Option A):
      step_kl_tf        -- KL per y* step under teacher forcing
      mean_kl_tf        -- scalar summary
"""

import json
import time
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from kl_faith_eval import (
    ENGLISH_TASKS, DATASET2PROMPT, DATASET2MAXLEN,
    DEFAULT_MAX_SEQ_FULL, TASK_MAX_SEQ_FULL,
    METHOD_CONFIGS,
)
from kl_faith_eval_ystar import (
    _KVPRESS_METHODS, make_comp_ids, get_comp_log_probs,
    PyramidKVRerotationPress,
)

DEFAULT_TASKS   = ["hotpotqa", "triviaqa", "trec", "2wikimqa"]
DEFAULT_METHODS = ["snapkv_press", "snapkv_rerotated", "pyramidkv", "pyramidkv_rerotated"]
DEFAULT_N       = 50


def _is_rerotated(press):
    """True for any press that needs explicit cache_position during decode.

    KeyRerotationPress and PyramidKVRerotationPress both re-rotate keys to
    contiguous positions, so decode must start at M (= get_seq_length() or T)
    rather than at T+1 as model.generate() would assume.
    """
    if isinstance(press, PyramidKVRerotationPress):
        return True
    try:
        from kvpress import KeyRerotationPress
        return isinstance(press, KeyRerotationPress)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_with_scores_rerotated(model, tokenizer, input_ids, max_new_tokens, device, press):
    """Greedy generation for KeyRerotationPress methods with explicit cache_position.

    Replicates generate_rerotated() from gt_eval_compression.py but captures
    per-step log-softmax distributions for KL analysis.
    """
    torch.cuda.empty_cache()
    input_ids = input_ids.to(device)
    with press(model):
        out = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )
    kv = out.past_key_values
    M = kv.get_seq_length() if not isinstance(press, PyramidKVRerotationPress) else input_ids.shape[1]
    # Step 0 score: from prefill logit (correct full-context attention)
    score0 = F.log_softmax(out.logits[0, -1].float().cpu(), dim=-1)
    log_probs = [score0]
    tok_ids   = [score0.argmax().item()]
    cur_tok   = torch.tensor([[tok_ids[0]]], device=device)
    del out
    torch.cuda.empty_cache()

    for step in range(max_new_tokens - 1):
        cache_pos = torch.tensor([M + step], dtype=torch.long, device=device)
        out_s = model(cur_tok, past_key_values=kv, cache_position=cache_pos,
                      use_cache=True, return_dict=True)
        lp = F.log_softmax(out_s.logits[0, -1].float().cpu(), dim=-1)
        log_probs.append(lp)
        next_id = lp.argmax().item()
        tok_ids.append(next_id)
        kv = out_s.past_key_values
        del out_s
        cur_tok = torch.tensor([[next_id]], device=device)
        if next_id == tokenizer.eos_token_id:
            break

    del kv
    torch.cuda.empty_cache()
    return tok_ids, log_probs


def generate_with_scores(model, tokenizer, input_ids, max_new_tokens, device, press=None):
    """Greedy generation; returns (token_ids list, list of log-softmax tensors)."""
    if press is not None and _is_rerotated(press):
        return generate_with_scores_rerotated(
            model, tokenizer, input_ids, max_new_tokens, device, press)

    input_ids = input_ids.to(device)
    ctx = press(model) if press is not None else nullcontext()
    with ctx:
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    new_toks = out.sequences[0, input_ids.shape[1]:].tolist()
    log_probs = [F.log_softmax(s[0].float().cpu(), dim=-1) for s in out.scores]
    del out
    torch.cuda.empty_cache()
    return new_toks, log_probs   # list[int], list[Tensor(vocab)]


def topk_overlap(lp_a, lp_b, k):
    _, ids_a = torch.topk(lp_a, k)
    _, ids_b = torch.topk(lp_b, k)
    return len(set(ids_a.tolist()) & set(ids_b.tolist())) / k


def kl_div(log_p, log_q):
    p = log_p.exp()
    return (p * (log_p - log_q)).sum().item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_results(path):
    if path.exists():
        d = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded existing: {path}  ({len(d)} entries)")
        return d
    return {}


def save_results(data, path):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(data, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, tasks, methods, data_dir, output_path,
             max_examples, device):
    results = load_results(output_path)
    t0      = time.time()
    total   = len(tasks) * max_examples * len(methods)
    done    = 0

    print(f"\n{'='*70}")
    print(f"Free-gen + teacher-forced logit eval")
    print(f"  Tasks:   {tasks}")
    print(f"  Methods: {methods}")
    print(f"  N/task:  {max_examples}")
    print(f"{'='*70}\n")

    for task in tasks:
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — no data file"); continue
        with open(data_file) as f:
            dataset = [json.loads(l) for l in f if l.strip()]

        template     = DATASET2PROMPT.get(task, "{context}{input}")
        max_new      = DATASET2MAXLEN.get(task, 64)
        max_seq_full = TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        gt_cap       = max_seq_full - max_new   # same truncation gt_eval uses
        n_ex         = min(max_examples, len(dataset))

        for idx in range(n_ex):
            all_done = all(f"{task}|{m}|{idx}" in results for m in methods)
            if all_done:
                done += len(methods); continue

            ex     = dataset[idx]
            q_text = ex.get("input", "").replace("NEWLINE_CHAR", "\n")
            prompt = template.format(
                context=ex.get("context", "").replace("NEWLINE_CHAR", "\n"),
                input=q_text,
            )
            full_ids = tokenizer.encode(prompt, add_special_tokens=True,
                                        return_tensors="pt")
            if full_ids.shape[1] > gt_cap:
                full_ids = full_ids[:, -gt_cap:]   # left-truncate, same as gt_eval

            # ── Full model: one run gives y* and log_p_full ──────────────────
            full_toks, full_log_probs = generate_with_scores(
                model, tokenizer, full_ids, max_new, device, press=None)
            full_pred = tokenizer.decode(full_toks, skip_special_tokens=True).strip()
            ystar     = torch.tensor(full_toks, dtype=torch.long)   # (n_gen,)

            # Stack log_p_full: (n_gen, vocab) — used for teacher-forced KL
            lp_full_stacked = torch.stack(full_log_probs, dim=0)    # (n_gen, vocab)
            torch.cuda.empty_cache()

            for method in methods:
                ck_key = f"{task}|{method}|{idx}"
                if ck_key in results:
                    done += 1; continue

                press = _KVPRESS_METHODS.get(method)
                pcfg  = METHOD_CONFIGS.get(method)

                comp_ids = make_comp_ids(
                    full_ids, method, tokenizer, q_text, gt_cap,
                    model=model, device=device,
                )

                entry = {}

                # ── Option B: free generation ────────────────────────────────
                try:
                    comp_toks, comp_log_probs = generate_with_scores(
                        model, tokenizer, comp_ids, max_new, device, press=press)
                    comp_pred = tokenizer.decode(comp_toks, skip_special_tokens=True).strip()

                    n_fg = min(len(full_log_probs), len(comp_log_probs))
                    step_agree = []; step_div = []; step_o5 = []; step_o10 = []; step_kl_fg = []
                    diverged = False
                    for t in range(n_fg):
                        if t > 0 and full_toks[t-1] != comp_toks[t-1]:
                            diverged = True
                        lp_f = full_log_probs[t]
                        lp_c = comp_log_probs[t]
                        step_div.append(diverged)
                        step_agree.append(lp_f.argmax().item() == lp_c.argmax().item())
                        step_o5.append(topk_overlap(lp_f, lp_c, 5))
                        step_o10.append(topk_overlap(lp_f, lp_c, 10))
                        step_kl_fg.append(kl_div(lp_f, lp_c))

                    entry.update({
                        "full_toks":      full_toks,
                        "comp_toks":      comp_toks,
                        "full_pred":      full_pred,
                        "comp_pred":      comp_pred,
                        "em":             (full_pred == comp_pred),
                        "step_agree":     step_agree,
                        "step_diverged":  step_div,
                        "step_overlap5":  step_o5,
                        "step_overlap10": step_o10,
                        "step_kl_fg":     step_kl_fg,
                    })
                except Exception as e:
                    print(f"  WARNING FG [{task}][{idx}][{method}]: {e}")
                    entry["fg_error"] = str(e)

                # ── Option A: teacher-forced through y* ──────────────────────
                try:
                    lp_comp_tf = get_comp_log_probs(
                        model, comp_ids, ystar, pcfg, device, press=press)
                    if lp_comp_tf is not None:
                        n_tf = min(lp_full_stacked.shape[0], lp_comp_tf.shape[0])
                        step_kl_tf = [
                            kl_div(lp_full_stacked[t], lp_comp_tf[t])
                            for t in range(n_tf)
                        ]
                        entry["step_kl_tf"]  = step_kl_tf
                        entry["mean_kl_tf"]  = sum(step_kl_tf) / len(step_kl_tf)
                except Exception as e:
                    print(f"  WARNING TF [{task}][{idx}][{method}]: {e}")
                    entry["tf_error"] = str(e)

                results[ck_key] = entry if entry else None
                save_results(results, output_path)
                done += 1

                elapsed = time.time() - t0
                em_str  = "Y" if entry.get("em") else "N"
                kl_tf   = f"{entry.get('mean_kl_tf', float('nan')):.3f}"
                n_align = sum(1 for d in entry.get("step_diverged", []) if not d)
                n_ag    = sum(a for a, d in zip(entry.get("step_agree", []),
                                                entry.get("step_diverged", [])) if not d)
                agree_s = f"{n_ag}/{n_align}" if n_align else "n/a"
                print(
                    f"  {task} ex{idx:>3}/{n_ex} {method:<24}"
                    f"  em={em_str}  agree={agree_s}"
                    f"  KL_TF={kl_tf}  [{int(elapsed)}s]",
                    flush=True,
                )

    print(f"\nDone. Results → {output_path}")
    _summarize(results, tasks, methods, max_examples)


def _summarize(results, tasks, methods, n):
    import statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'Method':<26}  {'EM%':>5}  {'agree@0':>8}  {'KL_TF':>7}  {'KL_FG':>7}")
    print("-" * 60)
    for method in methods:
        em_vals = []; agree0 = []; kl_tf = []; kl_fg = []
        for task in tasks:
            for idx in range(n):
                e = results.get(f"{task}|{method}|{idx}")
                if not isinstance(e, dict): continue
                if "em" in e:          em_vals.append(float(e["em"]))
                if e.get("step_agree") and not e["step_diverged"][0]:
                    agree0.append(float(e["step_agree"][0]))
                if "mean_kl_tf" in e:  kl_tf.append(e["mean_kl_tf"])
                if e.get("step_kl_fg"):
                    kl_fg.append(statistics.mean(e["step_kl_fg"]))
        def _m(xs): return f"{sum(xs)/len(xs):.3f}" if xs else "  n/a"
        print(f"  {method:<24}  {_m(em_vals):>5}  {_m(agree0):>8}  {_m(kl_tf):>7}  {_m(kl_fg):>7}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_dir", default="lb_data_raw/data")
    p.add_argument("--output",   default="lb_results_base/free_gen_logits.pt")
    p.add_argument("--tasks",    default=",".join(DEFAULT_TASKS))
    p.add_argument("--methods",  default=",".join(DEFAULT_METHODS))
    p.add_argument("--n",        type=int, default=DEFAULT_N)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def main():
    args    = parse_args()
    tasks   = [t.strip() for t in args.tasks.split(",")   if t.strip() in ENGLISH_TASKS]
    _known  = set(_KVPRESS_METHODS) | set(METHOD_CONFIGS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in _known]

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device)
    model.eval()

    run_eval(
        model, tokenizer, tasks, methods,
        data_dir     = Path(args.data_dir),
        output_path  = Path(args.output),
        max_examples = args.n,
        device       = args.device,
    )


if __name__ == "__main__":
    main()

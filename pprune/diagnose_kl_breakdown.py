"""
diagnose_kl_breakdown.py
------------------------
Per-token KL breakdown for PyramidKV vs phrase method.
Isolates how much KL comes from answer tokens vs post-answer continuation.

Run on a small sample after the main eval clears the GPU.
"""
import argparse, json, os, torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tasks",   default="triviaqa,trec,2wikimqa,gov_report")
    p.add_argument("--n",       type=int, default=10)
    p.add_argument("--output",  default="lb_results_base/kl_breakdown.json")
    p.add_argument("--ystar_cache", default="lb_results_base/ystar_cache.pt")
    p.add_argument("--device",  default="cuda")
    return p.parse_args()

@torch.inference_mode()
def get_comp_logprobs_per_position(model, comp_ids, ystar_tokens, press, device):
    """Returns [n_gen, vocab] log-softmax tensor from compressed model."""
    comp_input = comp_ids.to(device)
    try:
        ctx = press(model) if press is not None else nullcontext()
        with ctx:
            out = model(input_ids=comp_input,
                        attention_mask=torch.ones_like(comp_input),
                        use_cache=True)
        kv = out.past_key_values
        del out

        logits_list = []
        n_gen = ystar_tokens.shape[0]
        for t in range(n_gen):
            tok = ystar_tokens[t:t+1].unsqueeze(0).to(device)
            step = model(input_ids=tok, past_key_values=kv, use_cache=True)
            logits_list.append(step.logits[0, -1, :].float())
            kv = step.past_key_values
        del kv
        return F.log_softmax(torch.stack(logits_list), dim=-1).cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

def main():
    args = parse_args()
    from kl_faith_eval import DATASET2PROMPT, DATASET2MAXLEN, ENGLISH_TASKS
    from kl_faith_eval_ystar import make_comp_ids, _KVPRESS_METHODS

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device)
    model.eval()

    ystar_cache = torch.load(args.ystar_cache, map_location="cpu", weights_only=True)

    press_pyramid = _KVPRESS_METHODS.get("pyramidkv")
    data_dir = Path("lb_data_raw/data")

    results = {}
    tasks = [t for t in args.tasks.split(",") if t in ENGLISH_TASKS]

    for task in tasks:
        dataset = json.load(open(data_dir / f"{task}.jsonl")
                           ) if False else [json.loads(l) for l in open(data_dir / f"{task}.jsonl")]
        template = DATASET2PROMPT[task]
        max_new = DATASET2MAXLEN.get(task, 64)
        results[task] = []

        for idx in range(min(args.n, len(dataset))):
            key = f"{task}|{idx}"
            if key not in ystar_cache:
                continue
            entry = ystar_cache[key]
            ystar_tokens = entry["tokens"]
            ref_lp = entry["log_p_full"]   # [n_gen, vocab]
            n_gen = ref_lp.shape[0]

            ex = dataset[idx]
            prompt = template.format(
                context=ex.get("context","").replace("NEWLINE_CHAR","\n"),
                input=ex.get("input","").replace("NEWLINE_CHAR","\n"))
            full_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if full_ids.shape[1] > 4096:
                full_ids = full_ids[:, -4096:]

            # Phrase method (cw128): make_comp_ids
            question_text = ex.get("input","").replace("NEWLINE_CHAR","\n")
            comp_ids = make_comp_ids(full_ids, "chunk_word128_t20", tokenizer, question_text,
                                     max_seq_comp=4096, model=model, device=args.device)
            phrase_lp = get_comp_logprobs_per_position(model, comp_ids, ystar_tokens, None, args.device)

            # PyramidKV
            pyr_lp = get_comp_logprobs_per_position(model, full_ids, ystar_tokens, press_pyramid, args.device)

            if phrase_lp is None or pyr_lp is None:
                continue

            # Per-position KL
            ref_lp_f = ref_lp.float()
            phrase_kl_per_pos = F.kl_div(phrase_lp.float(), ref_lp_f,
                                          reduction="none", log_target=True).sum(-1).tolist()
            pyr_kl_per_pos = F.kl_div(pyr_lp.float(), ref_lp_f,
                                       reduction="none", log_target=True).sum(-1).tolist()

            # Decode tokens
            decoded = [tokenizer.decode([t]) for t in ystar_tokens.tolist()]

            # Find answer end (first \n after token 0)
            ans_end = next((i for i,d in enumerate(decoded) if '\n' in d and i > 0), len(decoded))

            results[task].append({
                "idx": idx,
                "ans_end": ans_end,
                "tokens": decoded[:min(32, len(decoded))],
                "phrase_kl_per_pos": phrase_kl_per_pos,
                "pyr_kl_per_pos": pyr_kl_per_pos,
                "phrase_kl_ans": sum(phrase_kl_per_pos[:ans_end]),
                "phrase_kl_cont": sum(phrase_kl_per_pos[ans_end:]),
                "pyr_kl_ans": sum(pyr_kl_per_pos[:ans_end]),
                "pyr_kl_cont": sum(pyr_kl_per_pos[ans_end:]),
            })
            print(f"[{task}|{idx}] ans_end={ans_end}  "
                  f"pyr_ans={sum(pyr_kl_per_pos[:ans_end]):.2f}  "
                  f"pyr_cont={sum(pyr_kl_per_pos[ans_end:]):.2f}  "
                  f"phrase_ans={sum(phrase_kl_per_pos[:ans_end]):.2f}  "
                  f"phrase_cont={sum(phrase_kl_per_pos[ans_end:]):.2f}", flush=True)

        # Summary for task
        if results[task]:
            r = results[task]
            print(f"\n=== {task} summary (n={len(r)}) ===")
            print(f"  PyramidKV:  ans_KL={sum(x['pyr_kl_ans'] for x in r)/len(r):.3f}  "
                  f"cont_KL={sum(x['pyr_kl_cont'] for x in r)/len(r):.3f}")
            print(f"  Phrase cw128: ans_KL={sum(x['phrase_kl_ans'] for x in r)/len(r):.3f}  "
                  f"cont_KL={sum(x['phrase_kl_cont'] for x in r)/len(r):.3f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()

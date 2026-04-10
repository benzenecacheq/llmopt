"""
vn_diagnostic.py
----------------
Collect per-token V-norm and KQ alignment distributions across task types
to understand what drives the LCC regression vs QA/summarization tasks.

For each (task, example, layer, head) we record:
  - vn_raw   : raw V-norm (L2 of value vector) per token
  - kq       : normalized KQ alignment score per token (pre-RoPE, max over Q-buffer)
  - token_ids: integer token ids (decode later for analysis)

The data is saved as a pickle of a list of records:
  {
    "task":        str,
    "example_idx": int,
    "token_ids":   List[int],          # (seq_len,)
    "seq_len":     int,
    "layers": {
      layer_idx: {
        "vn_raw": np.ndarray,          # (num_kv_heads, seq_len)
        "kq":     np.ndarray,          # (num_kv_heads, seq_len)
      }
    }
  }

Usage
-----
    python vn_diagnostic.py \\
        --model meta-llama/Llama-3.1-8B \\
        --tasks lcc,hotpotqa,gov_report,passage_retrieval_en,2wikimqa \\
        --n_examples 5 \\
        --output vn_diagnostic.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import ScorerPress
from kvpress.utils import get_prerope_query_states
from additive_scorer_press import AdditiveScorerPress

# Reuse data-loading helpers from kvpress_benchmark
from kvpress_benchmark import DATASET2PROMPT, DATASET2MAXLEN, load_task, build_prompt


# ---------------------------------------------------------------------------
# Capturing press — runs full additive scoring but saves raw intermediates
# ---------------------------------------------------------------------------

@dataclass
class CapturingPress(AdditiveScorerPress):
    """
    Identical to AdditiveScorerPress but captures raw V-norm and KQ scores
    for every layer into self.layer_data before returning final scores.

    Set compression_ratio=0.01 so almost nothing is pruned and we can still
    see scores for essentially the full context.
    """

    layer_data: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def reset(self):
        self.layer_data = []

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_kv_heads, T, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        num_kv_groups = num_heads // num_kv_heads
        device = keys.device

        # 1. Pre-RoPE query buffer
        buf_len = min(self.q_buffer_size, T)
        q_raw = get_prerope_query_states(module, hidden_states[:, -buf_len:])

        # 2. KQ alignment
        k_expanded = keys.repeat_interleave(num_kv_groups, dim=1)
        kq = torch.matmul(q_raw, k_expanded.transpose(-1, -2)) / (head_dim ** 0.5)
        kq = kq.amax(dim=2)                                       # (bsz, num_heads, T)
        kq_min = kq.amin(dim=-1, keepdim=True)
        kq_max = kq.amax(dim=-1, keepdim=True)
        kq = (kq - kq_min) / (kq_max - kq_min + 1e-8)
        kq = kq.view(bsz, num_kv_heads, num_kv_groups, T).amax(dim=2)  # → KV heads

        # 3. V-norm (raw, un-normalized for diagnostic; also compute normalized)
        vn_raw = values.norm(dim=-1)                              # (bsz, num_kv_heads, T)
        vn = vn_raw / (vn_raw.amax(dim=-1, keepdim=True) + 1e-8)

        # 4. Save diagnostic data (batch dim 0 only; we always run bsz=1)
        layer_idx = getattr(module, "layer_idx", len(self.layer_data))
        self.layer_data.append({
            "layer_idx": layer_idx,
            "vn_raw": vn_raw[0].cpu().to(torch.float32).numpy(),  # (num_kv_heads, T)
            "kq":     kq[0].cpu().to(torch.float32).numpy(),      # (num_kv_heads, T)
        })

        # 5. Linear decay + additive combination (same as parent)
        positions = torch.arange(T, device=device, dtype=keys.dtype)
        decay = self.min_decay + (1.0 - self.min_decay) * positions / max(T - 1, 1)
        scores = self.score_alpha * kq + (1.0 - self.score_alpha) * vn * decay

        # 6. Always-keep guards
        sentinel = scores.amax() + 1.0
        if self.always_keep_first > 0:
            scores[:, :, :min(self.always_keep_first, T)] = sentinel
        if self.always_keep_last > 0:
            scores[:, :, T - min(self.always_keep_last, T):] = sentinel

        return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tasks", default="lcc,hotpotqa,gov_report,passage_retrieval_en,2wikimqa")
    p.add_argument("--n_examples", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=6144)
    p.add_argument("--data_dir", default="lb_data_raw/data")
    p.add_argument("--output", default="vn_diagnostic.pkl")
    p.add_argument("--q_buffer_size", type=int, default=128)
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model  : {args.model}")
    print(f"Tasks  : {tasks}")
    print(f"N      : {args.n_examples} examples/task")

    print(f"\nLoading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device
    )
    model.eval()

    # Minimal compression so we still invoke the scoring path but keep ~99% of tokens
    press = CapturingPress(
        compression_ratio=0.01,
        q_buffer_size=args.q_buffer_size,
    )

    records = []

    for task in tasks:
        print(f"\n=== {task} ===")
        examples = load_task(task, data_dir, args.n_examples)

        for ex_idx, ex in enumerate(examples):
            prompt = build_prompt(task, ex, tokenizer, args.max_seq_len)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            token_ids = inputs.input_ids[0].cpu().tolist()

            print(f"  example {ex_idx}: {seq_len} tokens", end=" ... ", flush=True)

            press.reset()
            try:
                with press(model):
                    _ = model(**inputs, use_cache=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("OOM, skipped")
                continue

            # Organise layer_data by layer_idx
            layers_dict = {d["layer_idx"]: {"vn_raw": d["vn_raw"], "kq": d["kq"]}
                           for d in press.layer_data}

            records.append({
                "task":        task,
                "example_idx": ex_idx,
                "token_ids":   token_ids,
                "seq_len":     seq_len,
                "layers":      layers_dict,
            })

            print(f"captured {len(layers_dict)} layers")
            torch.cuda.empty_cache()

    print(f"\nSaving {len(records)} records → {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump({"records": records, "model": args.model, "args": vars(args)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # Quick summary statistics
    print("\nQuick summary (mean V-norm std across layers and heads, by task):")
    print(f"  {'task':<25} {'vn_std_mean':>12} {'vn_raw_mean':>12} {'kq_mean':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    by_task: Dict[str, List] = {t: [] for t in tasks}
    for rec in records:
        vn_stds, vn_means, kq_means = [], [], []
        for layer in rec["layers"].values():
            vn_stds.append(layer["vn_raw"].std(axis=-1).mean())   # mean over heads
            vn_means.append(layer["vn_raw"].mean())
            kq_means.append(layer["kq"].mean())
        by_task[rec["task"]].append((
            float(np.mean(vn_stds)),
            float(np.mean(vn_means)),
            float(np.mean(kq_means)),
        ))

    for task in tasks:
        if not by_task[task]:
            continue
        arr = np.array(by_task[task])
        print(f"  {task:<25} {arr[:,0].mean():>12.4f} {arr[:,1].mean():>12.4f} {arr[:,2].mean():>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

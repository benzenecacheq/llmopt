"""
snapkv_positions.py
-------------------
Instruments SnapKVPress to record which KV positions are retained at each layer.
Runs prefill-only (no generation) on a sample of LongBench examples.

Output: lb_results_base/snapkv_positions.json
  {
    "meta": {"compression_ratio": 0.35, "window_size": 64, "n_per_task": 20},
    "examples": [
      {
        "task": "narrativeqa",
        "idx": 0,
        "seq_len": 3412,
        "layers": [
          {"layer": 0, "n_kept": 2219, "positions": [14, 87, ...]},
          ...
        ]
      },
      ...
    ]
  }

Usage:
    conda run -n llmopt python snapkv_positions.py [--tasks t1,...] [--n 20]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from kvpress.presses.snapkv_press import SnapKVPress
from torch import nn

MODEL_NAME = "meta-llama/Llama-3.1-8B"
OUT_PATH   = "lb_results_base/snapkv_positions.json"

PRESS_CHOICES = ["snapkv", "radar_post", "radar_pre", "gaussian_bell"]

TASKS_DEFAULT = [
    "narrativeqa",           # QA — key info distributed through narrative
    "gov_report",            # long summarisation
    "hotpotqa",              # multi-hop QA
    "passage_retrieval_en",  # retrieval — paragraph at a known position
    "trec",                  # classification — very short question at end
]


# ── Instrumented press mixin ─────────────────────────────────────────────────

class _InstrumentedMixin:
    """Mixin that records per-layer retained positions (mean over heads).

    Subclass together with any _ScorerPress (SnapKVPress, RadarPostRopePress,
    RadarPreRopePress) that exposes a score() method and uses the standard
    score → topk → gather compress() pattern.
    """

    def _init_log(self):
        self.__dict__.setdefault("_log", [])

    def reset(self):
        self._log = []

    @property
    def log(self):
        return self._log

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        self._init_log()
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        k_len  = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        top    = scores.topk(n_kept, dim=-1)
        raw_indices = top.indices  # (batch, num_kv_heads, n_kept)

        pos_mean = raw_indices[0].float().mean(dim=0).sort().values
        self._log.append({
            "layer": getattr(module, "layer_idx", len(self._log)),
            "seq_len": k_len,
            "n_kept": n_kept,
            "positions": pos_mean.cpu().tolist(),
        })

        idx_exp = raw_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys   = keys.gather(2, idx_exp).contiguous()
        values = values.gather(2, idx_exp).contiguous()
        return keys, values


@dataclass
class InstrumentedSnapKVPress(_InstrumentedMixin, SnapKVPress):
    """SnapKVPress that records per-layer retained positions."""
    pass


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",            default=",".join(TASKS_DEFAULT))
    parser.add_argument("--n",                type=int,   default=20)
    parser.add_argument("--compression_ratio",type=float, default=0.35,
                        help="eviction fraction (0.35 = evict 35%, keep 65%%)")
    parser.add_argument("--window_size",      type=int,   default=64)
    parser.add_argument("--press",            default="snapkv", choices=PRESS_CHOICES)
    parser.add_argument("--output",           default=None)
    parser.add_argument("--data_dir",         default="lb_data_raw/data")
    parser.add_argument("--max_seq",          type=int,   default=None,
                        help="override TASK_MAX_SEQ_FULL for all tasks")
    parser.add_argument("--model",            default=MODEL_NAME,
                        help="HuggingFace model name (default: Llama-3.1-8B)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"lb_results_base/{args.press}_positions.json"

    tasks    = [t.strip() for t in args.tasks.split(",")]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Resume
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        results = saved.get("examples", [])
        done    = {(e["task"], e["idx"]) for e in results}
        print(f"Resuming — {len(results)} examples already saved.")
    else:
        results, done = [], set()

    # Lazy-import heavy deps so --help is fast
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kl_faith_eval import DATASET2PROMPT, TASK_MAX_SEQ_FULL, DEFAULT_MAX_SEQ_FULL

    # Build instrumented press
    if args.press == "snapkv":
        press = InstrumentedSnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
        )
    elif args.press == "gaussian_bell":
        from kl_faith_eval_ystar import _GaussianBellPress

        @dataclass
        class InstrumentedGaussianBellPress(_InstrumentedMixin, _GaussianBellPress):
            pass

        press = InstrumentedGaussianBellPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
        )
    else:
        from kl_faith_eval_ystar import _RadarPostRopePress, _RadarPreRopePress

        @dataclass
        class InstrumentedRadarPostRopePress(_InstrumentedMixin, _RadarPostRopePress):
            pass

        @dataclass
        class InstrumentedRadarPreRopePress(_InstrumentedMixin, _RadarPreRopePress):
            pass

        cls = InstrumentedRadarPostRopePress if args.press == "radar_post" else InstrumentedRadarPreRopePress
        press = cls(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
        )

    model_name = args.model
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()

    for task in tasks:
        data_file = data_dir / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[{task}] SKIP — {data_file} not found"); continue

        with open(data_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        template    = DATASET2PROMPT.get(task, "{context}{input}")
        max_seq     = args.max_seq if args.max_seq else TASK_MAX_SEQ_FULL.get(task, DEFAULT_MAX_SEQ_FULL)
        n_ex        = min(args.n, len(dataset))
        print(f"\n[{task}]  n={n_ex}  max_seq={max_seq}")

        for idx in range(n_ex):
            if (task, idx) in done:
                continue
            ex = dataset[idx]
            prompt = template.format(**ex)
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=max_seq
            )["input_ids"].to(model.device)
            seq_len = input_ids.shape[1]

            if seq_len <= press.window_size + 10:
                print(f"  [{idx}] skip (seq_len={seq_len} too short)"); continue

            torch.cuda.empty_cache()
            press.reset()
            with torch.no_grad(), press(model):
                _ = model(input_ids, use_cache=True)

            if not press.log:
                print(f"  [{idx}] warning — nothing logged"); continue

            results.append({
                "task":    task,
                "idx":     idx,
                "seq_len": seq_len,
                "layers":  press.log.copy(),
            })
            done.add((task, idx))
            print(f"  [{idx}] seq_len={seq_len}  n_layers={len(press.log)}")

            with open(out_path, "w") as f:
                json.dump({"meta": {
                    "press": args.press,
                    "compression_ratio": args.compression_ratio,
                    "window_size": args.window_size,
                    "n_per_task": args.n,
                }, "examples": results}, f)

    print(f"\nFinished. {len(results)} examples → {out_path}")


if __name__ == "__main__":
    main()

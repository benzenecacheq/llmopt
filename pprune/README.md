# KV Cache Pruning for Llama

Prefill-time KV cache pruning that retains a fixed fraction of token slots
while preserving model quality on long-context tasks. Evaluated on LongBench
with Llama-3.1-8B and Mistral-7B-v0.3 at 65% retention.

**Best method**: `kq_post_rope` — post-RoPE KQ alignment with optional distance
decay. Achieves 104.2 perplexity faithfulness vs 95.5 for naive truncation, with
equivalent ground-truth scores (23.9 vs 23.4 average).

**Active paper**: `paper_faithfulness_prerope.md` — faithfulness-based evaluation
framework and method description. Supersedes `paper.md`.

## Files

| File | Purpose |
|---|---|
| `llama_pruned.py` | Core pruning module — patches Llama attention layers |
| `longbench_eval.py` | Resumable LongBench evaluation harness |
| `faithfulness_deep.py` | Perplexity and embedding faithfulness scoring |
| `mode_compare.py` | Quick mode/alpha/retention sweep without reloading the model |
| `paper_faithfulness_prerope.md` | **Active paper** — faithfulness evaluation framework |
| `paper.md` | Earlier paper (additive method, ground-truth evaluation only) |
| `needle_test.py` | Needle-in-a-haystack sanity check |
| `decay_screen.csh` | Quick screen for kq_post_rope + distance decay variants |
| `run_faithfulness_modes.sh` | Adds all method labels to lb_results_base checkpoint |

## Environment

All scripts require the `llmopt` conda environment:

```bash
conda run -n llmopt python longbench_eval.py ...
```

Or activate first:
```bash
conda activate llmopt
```

Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

## Quick Start — LongBench

Download LongBench data (one-time):
```bash
conda run -n llmopt python -c "
from huggingface_hub import hf_hub_download
import zipfile
p = hf_hub_download(repo_id='THUDM/LongBench', filename='data.zip', repo_type='dataset')
with zipfile.ZipFile(p) as z:
    z.extractall('lb_data_raw')
"
```

Run the full evaluation (takes ~12 hours on V100 32GB):
```bash
conda run -n llmopt python longbench_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --budget_fraction 0.65 \
    --score_mode kq_post_rope \
    --min_decay 1.0 \
    --max_seq_len 7168 \
    --output lb_results
```

Resume after interruption: re-run the same command. Completed examples are
checkpointed to `lb_results/checkpoint.json` and skipped on restart.

## Key Parameters (PrunedLlamaConfig)

| Parameter | Value | Description |
|---|---|---|
| `budget_fraction` | 0.65 | Fraction of tokens to retain per sequence |
| `score_mode` | kq_post_rope | Scoring mode (see below) |
| `min_decay` | 1.0 | Decay at oldest position; 1.0 = no decay |
| `q_buffer_size` | 128 | Tail Q vectors used for KQ alignment |
| `always_keep_first` | 16 | Unconditionally retained prefix tokens |
| `always_keep_last` | 16 | Unconditionally retained suffix tokens |

## Score Modes

| Mode | Formula | Notes |
|---|---|---|
| `kq_post_rope` | kq · decay | **Primary method.** Post-RoPE KQ, decay off by default (min_decay=1.0) |
| `kq_only` | kq (pre-RoPE) | Pre-RoPE ablation — no decay |
| `additive` | α·kq + (1-α)·vn·decay | Pre-RoPE KQ + V-norm. Earlier best method |
| `vn_decay` | vn · decay | V-norm only ablation |
| `snapkv` | attn weights (post-RoPE) | SnapKV baseline |
| `streaming` | sink + recency | StreamingLLM baseline |

For `kq_post_rope`, decay activates when `min_decay < 1.0`. The decay value
is the weight given to the oldest token; 0.7 means oldest token scores at
70% of its raw KQ value. A screen experiment (see README2.md) is testing
whether decay helps on faithfulness metrics.

## Faithfulness Results (Llama-3.1-8B, 65% retention)

| Method | GT↑ | Perplexity↑ | Embedding↑ | Lexical↑ |
|---|---|---|---|---|
| Naive_65pct | 23.4 | 95.5 | **90.7** | **57.9** |
| kq_post_rope | **23.9** | **104.2** | 85.7 | 46.0 |
| kq_only | 23.0 | 101.6 | 84.8 | 43.9 |
| SnapKV | 23.8 | 102.8 | 86.1 | 46.7 |
| vn_decay | 17.8 | 106.2 | 78.2 | 30.2 |
| Streaming | 13.4 | 68.2 | 62.2 | 11.9 |

Perplexity faithfulness: 100 = compressed output as likely as full-context output
under the full model. Embedding: 100 = identical, 50 = orthogonal.

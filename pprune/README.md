# Additive KV Cache Pruning for Llama

Prefill-time KV cache pruning that retains a fixed fraction of token slots
while preserving model quality on long-context tasks. Evaluated on LongBench
with Llama-3.1-8B at 65% retention.

## Files

| File | Purpose |
|---|---|
| `llama_pruned.py` | Core pruning module — patches Llama attention layers |
| `longbench_eval.py` | Resumable LongBench evaluation harness (full + naive + pruned) |
| `mode_compare.py` | Quick mode/alpha/retention sweep without reloading the model |
| `paper.md` | Conference-style paper describing the method and results |
| `needle_test.py` | Needle-in-a-haystack sanity check |
| `eval.py` | Earlier evaluation harness (needle / QA / summarization) |

## Installation

```bash
pip install torch transformers accelerate
pip install rouge-score datasets huggingface_hub
```

Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

## Quick Start — LongBench

Download LongBench data (one-time):
```bash
python -c "
from huggingface_hub import hf_hub_download
import zipfile, pathlib
p = hf_hub_download(repo_id='THUDM/LongBench', filename='data.zip', repo_type='dataset')
with zipfile.ZipFile(p) as z:
    z.extractall('lb_data_raw')
"
```

Run the full evaluation (takes ~12 hours on V100 32GB):
```bash
python longbench_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --budget_fraction 0.65 \
    --score_mode additive \
    --score_alpha 0.65 \
    --max_seq_len 7168 \
    --output lb_results
```

Resume after interruption: re-run the same command. Completed examples are
checkpointed to `lb_results/checkpoint.json` and skipped on restart.

## Quick Mode Comparison (no model reload)

```bash
python mode_compare.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --tasks passage_retrieval_en,hotpotqa,2wikimqa,gov_report,qmsum,multi_news \
    --max_examples 30 \
    --modes vn_decay,kq_only,additive \
    --alphas 0.6,0.65,0.7,0.9 \
    --retentions 0.50,0.65,0.80
```

## Key Parameters (PrunedLlamaConfig)

| Parameter | Best value | Description |
|---|---|---|
| `budget_fraction` | 0.65 | Fraction of tokens to retain per sequence |
| `score_mode` | additive | Scoring: `vn_decay`, `kq_only`, or `additive` |
| `score_alpha` | 0.65 | Weight on KQ in additive mode (0=vn only, 1=kq only) |
| `min_decay` | 0.7 | Distance decay value at position 0 (auto-scales with T) |
| `q_buffer_size` | 128 | Tail Q vectors used for KQ alignment |
| `always_keep_first` | 16 | Unconditionally retained prefix tokens |
| `always_keep_last` | 16 | Unconditionally retained suffix tokens |
| `decay_fn` | linear | `linear` or `exponential` |

## Scoring Function

```
score_i = α · kq_i + (1 − α) · vn_i · decay_i
```

- **kq_i**: max-pooled pre-RoPE dot product of tail Q-vectors against K_i (normalized)
- **vn_i**: L2 norm of V_i normalized by layer max
- **decay_i**: linear ramp from min_decay (oldest) to 1.0 (newest)

Pre-RoPE keys are used so position does not artificially suppress early tokens.

## LongBench Results (Llama-3.1-8B, 65% retention)

| Task | Full | Naive (4K) | Pruned (65%) |
|---|---|---|---|
| NarrativeQA | 5.5 | 19.5 | 3.1 |
| Qasper | 11.1 | 11.4 | 10.1 |
| MultifieldQA | 28.9 | 28.5 | 27.1 |
| HotpotQA | 9.9 | 10.9 | 9.9 |
| 2WikiMQA | 14.1 | 14.2 | 11.3 |
| MuSiQue | 6.9 | 6.7 | 5.1 |
| GovReport | 20.4 | 20.2 | 17.5 |
| QMSum | 10.3 | 14.6 | 8.1 |
| MultiNews | 1.1 | 0.8 | 1.4 |
| TREC | 70.0 | 71.0 | 65.0 |
| TriviaQA | 17.4 | 17.8 | 17.7 |
| SAMSum | 16.0 | 16.4 | 15.6 |
| PassageCount | 3.0 | 1.0 | 2.0 |
| **PassageRetrieval** | **44.0** | **18.0** | **27.0** |
| LCC | 68.1 | 66.5 | 57.4 |
| RepoBench-P | 55.6 | 54.7 | 49.0 |
| **Average** | **23.9** | **23.3** | **20.5** |

Key win: PassageRetrieval 27 vs naive 18 (+9 pts). Main regression: code tasks.

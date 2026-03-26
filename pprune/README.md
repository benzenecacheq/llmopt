# Per-Head KV Pruning for Llama

Experiments comparing standard Llama inference against a per-head importance
filter that prunes the KV cache during prefill, allowing a context of length x
to be processed with the memory/compute footprint of x/2.

Supports Llama 2 and Llama 3.

## Files

| File | Purpose |
|---|---|
| `head_filter.py` | Core algorithm — per-head importance scoring and mask generation |
| `llama_pruned.py` | Modified Llama inference engine — swaps attention layers (Llama 2 + 3) |
| `needle_test.py` | Needle-in-a-haystack evaluation |
| `eval.py` | Full three-way evaluation harness (needle / QA / summarization) |

## Installation

```bash
pip install torch transformers accelerate
pip install sacrebleu rouge-score          # metrics
pip install sentence-transformers          # optional: semantic similarity
```

You will need a HuggingFace account with access to `meta-llama/Llama-2-7b-hf`.
Set your token:
```bash
export HF_TOKEN=your_token_here
huggingface-cli login
```

## Quick Start — Needle Test

Runs 50 trials with context_len=1024, budget=512 (x/2 retention):

```bash
python eval.py \
    --task needle \
    --model meta-llama/Llama-2-7b-hf \
    --context_len 1024 \
    --budget 512 \
    --num_samples 50 \
    --budget_strategy entropy \
    --decay_fn exponential \
    --output needle_results.json
```

## QA Evaluation

Expects a JSONL file with records: `{"context": "...", "question": "...", "answer": "..."}`

```bash
python eval.py \
    --task qa \
    --data my_qa.jsonl \
    --context_len 1024 \
    --budget 512 \
    --use_sbert \
    --output qa_results.json
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--context_len` | 1024 | Full context length fed to the model |
| `--budget` | 512 | Retained token count (target = context_len/2) |
| `--budget_strategy` | entropy | `equal` or `entropy` (entropy-weighted per head) |
| `--decay_fn` | exponential | `linear` or `exponential` distance decay |
| `--decay_rate` | 0.002 | Steepness of distance decay |
| `--q_buffer` | 64 | Tail Q vectors tracked per head |
| `--always_keep` | 16 | Unconditionally retain this many tail tokens |

## How It Works

### Scoring (head_filter.py)

For each token j and each attention head h:

```
score(h, j) = kq_alignment(h, j)
            × v_norm(h, j)
            × distance_decay(j)
```

- **KQ alignment**: max dot product of K_h(j) against a rolling buffer of
  the last `q_buffer` Q vectors from the tail of the sequence
- **V-norm**: ||V_h(j)||, the maximum possible contribution of this token
- **Distance decay**: linear `1 - λ·dist` or exponential `e^{-λ·dist}`

### Budget Allocation

- **equal**: every head gets `total_budget / num_heads` slots
- **entropy**: heads with higher attention entropy (broader, more diffuse
  attention patterns) get proportionally more budget

### Inference (llama_pruned.py)

Patches each `LlamaAttention` layer in-place (shared weights, no retraining).
During prefill:
1. Stream all tokens through the filter
2. Per head, gather only retained K and V vectors
3. Run standard attention against the gathered (smaller) KV

During generation: standard KV cache, unmodified.

## Expected Results

On a V100 32GB with Llama2-7b-hf:

| Condition | VRAM | Needle accuracy (approx) |
|---|---|---|
| Full context x=1024 | ~16 GB | ~85-95% |
| Naive truncation x/2 | ~14 GB | ~40-60% (loses early needles) |
| Per-head filter x/2 | ~14 GB | ~70-90% (goal) |

Actual numbers will depend on needle position — early needles are the stress test.

## Tuning Tips

1. Start with `--decay_fn exponential --budget_strategy entropy` (the defaults)
2. If early-position accuracy is low, increase `--q_buffer` (64→128)
3. If recent context is suffering, increase `--always_keep` (16→32)
4. `--decay_rate 0.001` gives a flatter decay — better for long-range dependencies
5. Compare `equal` vs `entropy` budget strategy — the gap tells you how much
   head specialization matters for your task

# KV Cache Pruning — Developer Context

This document provides full technical context for resuming development.
It covers architecture decisions, what was tried, what worked, and what to do next.

---

## Current State (as of 2026-04-02)

- **Paper written**: `paper.md` — conference-style write-up of all findings
- **Best config**: additive α=0.65, budget_fraction=0.65, min_decay=0.7, linear decay
- **Evaluated on**: LongBench v1, Llama-3.1-8B (base, fp16), V100 32GB, 100 examples/task
- **Key result**: PassageRetrieval 27 vs naive 18 (+9 pts); code tasks regress vs naive

---

## Files

| File | Purpose |
|---|---|
| `llama_pruned.py` | Core module — `PrunedLlamaConfig`, `PrunedLlamaAttention`, `build_pruned_model` |
| `longbench_eval.py` | Resumable LongBench harness — two-pass (base then pruned), checkpoint per example |
| `mode_compare.py` | Quick ablation — patches pcfg in-place, no model reload between modes |
| `paper.md` | Conference paper |
| `needle_test.py` | Early needle-in-haystack sanity check |
| `eval.py` | Earlier harness (needle/QA/summarize) — superseded by longbench_eval.py |
| `lb_data_raw/data/` | LongBench JSONL files (extracted from data.zip) |
| `lb_results/` | Evaluation outputs (`results.json`, `checkpoint.json`) |

---

## Architecture: llama_pruned.py

### PrunedLlamaConfig

```python
@dataclass
class PrunedLlamaConfig:
    total_budget: int = 512          # used only if budget_fraction == 0
    q_buffer_size: int = 64          # tail Q vectors for KQ alignment
    budget_strategy: str = "entropy" # unused in current scoring path
    decay_fn: str = "linear"         # "linear" or "exponential"
    decay_rate: float = 0.0          # unused; min_decay used instead
    min_decay: float = 0.7           # decay value at position 0
    always_keep_last: int = 16       # unconditionally retained suffix tokens
    always_keep_first: int = 16      # unconditionally retained prefix tokens
    score_mode: str = "additive"     # "vn_decay", "kq_only", or "additive"
    score_alpha: float = 0.65        # weight on kq in additive mode
    budget_fraction: float = 0.0     # if > 0, overrides total_budget: int(T * fraction)
    filter_prefill_only: bool = True
```

### Scoring Function

```
score_i = α · kq_i + (1 − α) · vn_i · decay_i
```

Where:
- `kq_i = max_{j in tail} (Q_j_preRoPE · K_i_preRoPE) / √D`, normalized to [0,1] per head
- `vn_i = ‖V_i‖₂ / max_j ‖V_j‖₂` (normalized L2 value norm)
- `decay_i = min_decay + (1 − min_decay) · (i / (T−1))` — linear from min_decay at 0 to 1.0 at T-1
- Pre-RoPE K and Q are captured via hooks before rotary embedding is applied
- Post-RoPE K/V are still used for the actual attention computation

### GQA handling

Llama 3.x has 32 Q-heads and 8 KV-heads (q_per_kv=4). Scores are computed for
all Q-heads and aggregated per KV-head via max-pooling before selecting the
global budget positions.

### Budget selection

1. Always retain first `always_keep_first` and last `always_keep_last` tokens
2. Fill remaining budget with top-scoring non-protected tokens
3. Reconstruct causal attention mask using original (pre-pruning) positions

### Forward pass guard (important)

The guard in `forward()` computes `effective_budget` from `budget_fraction` if
set, otherwise falls back to `total_budget`. This was a critical bug fix: the
original code checked `total_budget < T` but `total_budget=999999`, so the
filter never ran when using `budget_fraction`.

```python
effective_budget = (int(T * self.pcfg.budget_fraction)
                    if self.pcfg.budget_fraction > 0.0
                    else self.pcfg.total_budget)
if is_prefill and self.pcfg.filter_prefill_only and ... and effective_budget < T:
    key_states, value_states, retained_pos = self._run_filter_prefill(
        q_raw, k_raw, key_states, value_states, effective_budget
    )
```

---

## Architecture: longbench_eval.py

Two-pass strategy to use one GPU efficiently:
- **Pass 1**: load base model, run "full" (no truncation) and "naive" (left-truncate to 4096 tokens) on all tasks
- **Pass 2**: load pruned model, run "pruned" on all tasks

Checkpointing: after every example, writes atomically to `{output}/checkpoint.json`.
On restart, skips already-completed (task, idx, method) triples.

OOM recovery: catches `torch.cuda.OutOfMemoryError`, halves `input_ids`, retries
up to 4 times. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` before running.

Retrieval scorer: extracts the number from both "Paragraph 17" and bare "17"
formats — critical because the base model outputs bare numbers.

---

## What Was Tried and What Worked

### Score modes

| Mode | Formula | Verdict |
|---|---|---|
| vn_decay | vn × decay | Fails on natural retrieval; OK on synthetic needle |
| kq_only | kq | Good on retrieval, unstable on some summarization |
| additive α=0.65 | 0.65·kq + 0.35·vn·decay | Best overall balance |

V-norm is high for numbers and rare words (works for synthetic needle), but near-zero
for common words and prose (fails when the needle is natural language).

KQ alignment uses pre-RoPE keys: without this, early tokens appear irrelevant
because RoPE rotates K vectors by position, creating artificially low dot products
regardless of semantic content.

Additive combination gives KQ and V-norm independent "voice". Multiplicative
combination approximates V-norm with a semantic tiebreaker — that was the old design
and it failed on retrieval.

### Budget parametrization

`budget_fraction` is context-length-independent (e.g. 0.65 means keep 65% of
each sequence's actual token count). The old `total_budget` int was context-length-
dependent and required manual calibration per task.

`min_decay` parametrizes decay by its value at the oldest position. The rate is
auto-derived as `rate = (1 − min_decay) / (T − 1)`, so the decay profile is
consistent regardless of T. Old fixed `decay_rate` would collapse early tokens
to near-zero at long contexts.

### Model choice

Instruct models are not well-suited to LongBench prompts (which are designed for
base models). Full scores were low (NarrativeQA=0) with instruct model.
Switch to `meta-llama/Llama-3.1-8B` (base) for definitive runs.

### Retrieval scoring fix

Original `retrieval_score` required "Paragraph 17" format. Base model outputs
bare "17". Fixed by regex that extracts the number from either format.

### Sequence length for retrieval

PassageRetrieval prompts are 11K-14K tokens. With `max_seq_len=7168`, the prompt
was truncated before the model saw most paragraphs. mode_compare.py uses
`max_seq_len=16000`; longbench_eval.py uses 7168 (a known tradeoff for V100 VRAM).

---

## Full LongBench Results

Model: Llama-3.1-8B (base), fp16, V100 32GB
Config: additive α=0.65, budget_fraction=0.65, min_decay=0.7, q_buffer_size=128,
        always_keep_first=16, always_keep_last=16, linear decay, max_seq_len=7168

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

The paper reports 86% of full-context performance recovered. The main loss is code
tasks (LCC -11, RepoBench -7 vs full). PassageRetrieval is the clear win (+9 vs naive).

NarrativeQA anomaly: naive (19.5) beats full context (5.5). Hypothesis: long story
contexts confuse the base model; truncation incidentally removes confusing material.

---

## Known Issues and Research Directions

### Code task regression (most important)

LCC and RepoBench-P are worse than naive truncation. Code completion is a recency
task — the immediately preceding lines matter most. KQ alignment may over-select
semantically interesting tokens (identifiers, keywords) at the expense of syntactically
necessary but low-salience tokens.

**Potential fixes:**
- Higher `always_keep_last` (e.g. 64 or 128) to unconditionally retain more recent context
- Lower `min_decay` (e.g. 0.3 or 0.1) for stronger recency bias
- Task-adaptive α: use α=0.9 or kq_only for code tasks, α=0.65 for QA

### Per-layer budgets

Early and late layers have different attention patterns. A flat 65% budget across
all layers is likely suboptimal. Retrieval heads (which do long-range lookup) are
concentrated in specific layers — those layers may need a larger budget.

### Larger/different models

Only tested on Llama-3.1-8B. Interesting to test:
- Llama-3.1-70B (if multi-GPU available)
- Mistral or Qwen2 families (different GQA ratios)
- A model with flash attention for speed comparison

### Efficiency measurement

No wall-clock timing or VRAM measurements were taken. A real paper needs:
- Prefill latency vs context length (full vs pruned)
- Peak VRAM vs retention fraction
- Throughput (tokens/sec) comparison

### Comparison with baselines in code

H2O, SnapKV, and StreamingLLM are not implemented here. Paper mentions them in
related work but does not benchmark against them directly — a stronger evaluation
would include at least one baseline.

---

## How to Resume Experiments

### Run a targeted ablation on specific tasks

```bash
python mode_compare.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --tasks lcc,repobench-p \
    --max_examples 30 \
    --modes additive \
    --alphas 0.65 \
    --retentions 0.65 \
    --always_keep_last 64 \
    --output mode_compare_code.json
```

### Run full LongBench with a different config

```bash
python longbench_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --budget_fraction 0.65 \
    --score_mode additive \
    --score_alpha 0.65 \
    --max_seq_len 7168 \
    --output lb_results_v2
```

### Check checkpoint progress

```bash
python -c "
import json
cp = json.load(open('lb_results/checkpoint.json'))
done = {(r['task'], r['idx'], r['method']) for r in cp}
print(f'{len(done)} examples done')
"
```

---

## Hardware Notes (V100 32GB)

- fp16 only — V100 does not support bfloat16
- No FlashAttention — requires Ampere or newer
- max_seq_len=7168 keeps peak VRAM below 32GB for Llama-3.1-8B
- For mode_compare.py with 3B model, max_seq_len=16000 works
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOM

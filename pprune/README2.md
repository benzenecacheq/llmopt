# KV Cache Pruning — Developer Context

This document provides full technical context for resuming development.
It covers architecture decisions, what was tried, what worked, and what to do next.

---

## Current State (as of 2026-04-30)

- **Active paper**: `paper_faithfulness_prerope.md` — faithfulness-based evaluation
  framework with kq_post_rope as the proposed method. Substantially revised today.
- **Best method**: `kq_post_rope` (post-RoPE KQ alignment, no decay at 104.2 perplexity
  faithfulness). Earlier work concluded `additive` was best, but that used ground-truth
  scoring which we now consider an unreliable metric for compression evaluation.
- **Experiment in progress**: `decay_screen.csh` — testing whether adding distance decay
  to kq_post_rope improves faithfulness metrics (perplexity faithfulness on 3 tasks ×
  20 examples, min_decay ∈ {0.9, 0.7, 0.5}). Results in `decay_screen_faith.json`.
- **Paper §4.2 is provisional**: currently describes the method with decay
  (`score_i = kq_i · decay_i`), but kq_post_rope was implemented without decay.
  The §4.2 description will be corrected once the decay screen results are in.
- **Evaluated on**: LongBench v1, Llama-3.1-8B (base, fp16) + Mistral-7B-v0.3,
  V100 32GB, 100 examples/task. All outputs in `lb_results_base/checkpoint.json`.
- **Conda environment**: `llmopt` — use `conda run -n llmopt python ...` for all scripts.

---

## Files

| File | Purpose |
|---|---|
| `llama_pruned.py` | Core module — `PrunedLlamaConfig`, `PrunedLlamaAttention`, `build_pruned_model` |
| `longbench_eval.py` | Resumable LongBench harness — checkpoint per example, OOM recovery |
| `faithfulness_deep.py` | Perplexity and embedding faithfulness scoring against full-context outputs |
| `mode_compare.py` | Quick ablation — patches pcfg in-place, no model reload between modes |
| `paper_faithfulness_prerope.md` | **Active paper** — faithfulness framework + kq_post_rope method |
| `paper.md` | Earlier paper (additive method, GT evaluation only) — superseded |
| `needle_test.py` | Early needle-in-haystack sanity check |
| `eval.py` | Earlier harness (needle/QA/summarize) — superseded by longbench_eval.py |
| `run_faithfulness_modes.sh` | Adds streaming/snapkv/vn_decay/kq_only/kq_post_rope to lb_results_base |
| `decay_screen.csh` | Quick faithfulness screen for kq_post_rope + decay variants |
| `lb_data_raw/data/` | LongBench JSONL files (extracted from data.zip) |
| `lb_results_base/` | Primary checkpoint (`checkpoint.json`, `results.json`, `faithfulness_deep.json`) |
| `lb_results_mistral/` | Mistral-7B-v0.3 results |
| `lb_results_streaming/` | Streaming baseline results |

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
    min_decay: float = 0.7           # decay value at oldest position
    always_keep_last: int = 16       # unconditionally retained suffix tokens
    always_keep_first: int = 16      # unconditionally retained prefix tokens
    score_mode: str = "additive"     # see score modes table below
    score_alpha: float = 0.65        # weight on kq in "additive" mode
    budget_fraction: float = 0.0     # if > 0, overrides total_budget: int(T * fraction)
    filter_prefill_only: bool = True
```

### Score Modes

| Mode | Key/Query space | Formula | Decay |
|---|---|---|---|
| `kq_post_rope` | Post-RoPE | kq · decay | Only when min_decay < 1.0 (default off) |
| `kq_only` | Pre-RoPE | kq | Never |
| `additive` | Pre-RoPE | α·kq + (1-α)·vn·decay | Always |
| `vn_decay` | — | vn · decay | Always |
| `snapkv` | Post-RoPE | pooled attn weights | Never |
| `streaming` | — | sink + recency window | Never |

**Important**: `kq_post_rope` was originally coded without decay (the decay
computation was only in the pre-RoPE `else` branch). As of 2026-04-30, decay
is now wired into the `kq_post_rope` branch and activates when `min_decay < 1.0`.
All existing checkpoint data for `kq_post_rope` was generated with min_decay=0.7
in the config but the code ignored it — those runs are equivalent to min_decay=1.0
(no decay). The decay screen will determine whether adding decay helps.

### kq_post_rope Scoring (Primary Method)

```
kq_i    = max_{j ∈ tail} (Q_j_postRoPE · K_i_postRoPE) / √D,  normalized to [0,1]
decay_i = min_decay + (1 − min_decay) · (i / (T−1))            [only if min_decay < 1.0]
score_i = kq_i · decay_i                                        [or just kq_i if no decay]
```

- Post-RoPE Q and K — same vectors the attention kernel uses
- Max-pooled over `q_buffer_size` tail queries (proxy for generation-phase queries)
- Distinct from SnapKV: raw dot products, not softmax attention weights

### GQA Handling

Llama 3.x has 32 Q-heads and 8 KV-heads (q_per_kv=4). Scores are computed for
all Q-heads and aggregated per KV-head via max-pooling before global budget selection.

### Budget Selection

1. Always retain first `always_keep_first` and last `always_keep_last` tokens
2. Fill remaining budget with top-scoring non-protected tokens
3. Reconstruct causal attention mask using original (pre-pruning) positions

### Forward Pass Guard (Important)

`effective_budget = int(T * budget_fraction)` if `budget_fraction > 0`, else
`total_budget`. Critical fix: original code used `total_budget=999999`, so the
filter never activated when using `budget_fraction`.

---

## Architecture: longbench_eval.py

Two-pass strategy for single-GPU efficiency:
- **Pass 1**: load base model, run "full" and "naive" on all tasks
- **Pass 2**: load pruned model, run each compressed method

Checkpointing: atomic write to `{output}/checkpoint.json` after every example.
On restart, skips already-completed `(task, idx, method)` triples.

OOM recovery: catches `torch.cuda.OutOfMemoryError`, halves `input_ids`, retries
up to 4 times. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` before running.

Key flags: `--method_label` sets the checkpoint key for a run (allows multiple
methods to share one checkpoint file); `--naive_fraction 0.0` skips the naive
baseline pass when adding new methods to an existing checkpoint.

---

## Architecture: faithfulness_deep.py

Computes two faithfulness metrics against stored full-context outputs (`full` key
in checkpoint):

- **Perplexity**: teacher-forced forward pass under the full-context model.
  `faith_ppl = exp(loss_full − loss_method) × 100`. Needs GPU and model loaded.
  100 = equally likely; >100 = more likely (valid); <100 = less likely (diverging).
- **Embedding**: cosine similarity via `all-MiniLM-L6-v2`, rescaled to [50, 100].
  CPU-only. 100 = identical embedding; 50 = orthogonal.

Results written to a separate JSON (e.g. `faithfulness_deep.json` or
`decay_screen_faith.json`). Checkpoints per task; safe to resume.

---

## What Was Tried and What Worked

### Ground-Truth Era (paper.md)

Early experiments used ground-truth (GT) scores as the primary metric. Under GT:
- `additive` α=0.65 appeared best (23.9 avg, led to paper.md)
- `kq_post_rope` looked comparable but not clearly better
- Decay appeared necessary — experiments without decay scored lower on GT
- This conclusion is now suspect: GT is insensitive to approximation quality

### Faithfulness Era (paper_faithfulness_prerope.md)

After introducing faithfulness metrics, the picture changed:

| Method | GT | Perplexity Faith. | Embedding Faith. |
|---|---|---|---|
| kq_post_rope | **23.9** | **104.2** | 85.7 |
| SnapKV | 23.8 | 102.8 | 86.1 |
| kq_only | 23.0 | 101.6 | 84.8 |
| naive_65pct | 23.4 | 95.5 | **90.7** |
| vn_decay | 17.8 | 106.2 | 78.2 |
| Streaming | 13.4 | 68.2 | 62.2 |

Key findings:
- `kq_post_rope` (post-RoPE, no decay) is the best overall method by faithfulness
- `kq_only` (pre-RoPE, no decay) is worse — post-RoPE beats pre-RoPE despite
  the theoretical argument that RoPE penalizes distant tokens
- `vn_decay` has highest perplexity but lowest embedding — "mode switching": finds
  high-probability outputs that are semantically different from the full model's outputs
- Adding V-norm to KQ alignment (pruned/additive) hurts both GT and faithfulness
- Naive truncation looks competitive on GT (23.4) but is measurably worse on
  perplexity faithfulness (95.5 vs 104.2) — a 9-point gap invisible to GT scoring

### Pre-RoPE vs Post-RoPE

The code comment originally said `kq_post_rope` was "the pre-RoPE method" — this
was wrong. Confirmed by code inspection:
- `kq_post_rope`: uses `key_states` (post-RoPE, rotated)
- `kq_only`: uses `k_raw` (pre-RoPE, captured before `apply_rotary_pos_emb`)
The §5.4 ablation is a clean comparison — both modes have no decay, differing
only in scoring space.

### Decay Screen (in progress as of 2026-04-30)

Early GT experiments concluded decay was not helpful. Since GT is an unreliable
metric for compression, we are re-evaluating using perplexity faithfulness.
`decay_screen.csh` tests min_decay ∈ {0.9, 0.7, 0.5} on passage_retrieval_en,
qmsum, samsum (20 examples each). Results in `decay_screen_faith.json`.

Decision rule: if any decay value beats no-decay kq_post_rope by >2 points on
average perplexity faithfulness → run full 16-task evaluation with that decay.
Otherwise → correct §4.2 in the paper to say method uses no decay.

---

## Paper Status (paper_faithfulness_prerope.md) — 2026-04-30

Revisions made today:
- **Title**: removed "with Pre-RoPE Scoring" (post-RoPE is the winning method)
- **Abstract, §1, §2**: reframed — pre-RoPE is a hypothesis we test, not a claim
- **§4**: expanded substantially — opening paragraph claims kq_post_rope as a novel
  method; §4.2 adds rationale for tail buffer, max-pooling, and decay;
  §4.3 rewritten as "post-RoPE vs pre-RoPE" comparison; §4.5 restructured into
  proposed method / external baselines / ablations
- **vn_decay**: GT (17.8) and Lexical (30.2) scores filled in — data was in
  `results.json` and `faithfulness_results.json` all along, just not tabulated
- **§1 contributions**: kq_post_rope explicitly claimed as novel method, distinct
  from SnapKV (same key space, different signal: raw dot products vs attn weights)

Still needs (pending decay screen):
- §4.2 final score formula: currently shows `score_i = kq_i · decay_i` but code
  has no decay for kq_post_rope. Will be corrected based on screen results.

---

## How to Resume Experiments

### Check decay screen results

```bash
python -c "
import json
r = json.load(open('decay_screen_faith.json'))
ppl = r.get('perplexity', {})
for task, scores in ppl.items():
    print(task, {k:round(v,1) for k,v in scores.items() if not k.startswith('_')})
"
```

### Add a new method to lb_results_base

```bash
conda run -n llmopt python longbench_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks passage_retrieval_en,qmsum,samsum \
    --max_examples 20 \
    --output lb_results_base \
    --budget_fraction 0.65 \
    --score_mode kq_post_rope \
    --min_decay 0.7 \
    --method_label kqpr_md0.7 \
    --naive_fraction 0.0
```

### Score faithfulness for new methods

```bash
conda run -n llmopt python faithfulness_deep.py \
    --model meta-llama/Llama-3.1-8B \
    --checkpoint lb_results_base/checkpoint.json \
    --output my_faith_results.json \
    --perplexity --embedding
```

### Run full 16-task LongBench with a new config

```bash
conda run -n llmopt python longbench_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --budget_fraction 0.65 \
    --score_mode kq_post_rope \
    --min_decay 0.7 \
    --max_seq_len 7168 \
    --method_label kqpr_md0.7 \
    --output lb_results_base \
    --naive_fraction 0.0
```

### Check checkpoint progress

```bash
python -c "
import json, collections
cp = json.load(open('lb_results_base/checkpoint.json'))
counts = collections.Counter(k.split('|')[2] for k in cp if k.count('|') == 2)
for method, n in sorted(counts.items()):
    print(f'{method}: {n}')
"
```

---

## Hardware Notes (V100 32GB)

- fp16 only — V100 does not support bfloat16
- No FlashAttention — requires Ampere or newer
- max_seq_len=7168 keeps peak VRAM below 32GB for Llama-3.1-8B
- For mode_compare.py with 3B model, max_seq_len=16000 works
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOM
- Perplexity faithfulness scoring is fast (forward passes only, no generation)

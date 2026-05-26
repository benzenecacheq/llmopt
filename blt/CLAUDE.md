# BLT Training — Session Context

## What this is
BLT (Bilinear Attention) is a research experiment replacing GPT-2's standard multi-head attention with a parameter-sharing bilinear variant. This is the **GD-only control condition** — all parameters trained with Adam (no alternative optimizer).

The key architectural idea: replace per-layer Wq and Wk with a single shared M matrix (768×768) initialized as the average of Wq @ Wk^T across all 12 GPT-2 layers. Since M is shared, X @ M @ X^T is computed once per layer (not once per head), saving 11/12 of the attention score computation vs standard MHA.

## Important decisions made
- **Scale factor**: `sqrt(d_head)` = `sqrt(64)` not `sqrt(d_model)` = `sqrt(768)`. The wrong scale was caught before the main run — it makes softmax temperature 3.5× off and would skew loss comparisons.
- **Baseline target**: Pretrained GPT-2 perplexity on WikiText-103 validation = **26.39** (measured with sliding-window eval, stride=512).
- **Training duration**: 2 epochs = ~50,300 steps (~15 hours on P100 at ~1.1s/step).

## Completed runs
- **BLT WikiText-103**: DONE. `run_seed42.pt`, 50,300 steps, final val_ppl = **21.50** (beats GPT-2 baseline of 26.39).
- **Baseline GPT-2 WikiText-103 fine-tune**: Stopped at step 33,930, val_ppl ~14.91. `run_baseline_seed42.pt`. Not resumed (plateaued).
- **BLT LAMBADA full-text fine-tune**: `run_lambada_seed42.pt`. Plateaued ~step 3500. LAMBADA benchmark acc 0.102 (worse than 0.114 before fine-tune).
- **BLT cloze fine-tune**: `run_cloze_seed42.pt`. Killed at step 7500. Catastrophic overfitting — train loss → 0 on 2,661 examples, LAMBADA benchmark acc collapsed to 0.065. Experiment abandoned.

## Benchmark results (lm-eval-harness)

| Task | GPT-2 (pretrained) | BLT (WikiText) | BLT (cloze, step 7500) |
|------|--------------------|----------------|------------------------|
| LAMBADA acc | 0.242 | 0.114 | 0.065 |
| LAMBADA ppl | 83.0 | 1307.5 | 590,704 |
| HellaSwag acc_norm | 0.291 | 0.275 | — |
| PIQA acc_norm | 0.560 | 0.541 | — |
| Winogrande acc | 0.502 | 0.507 | — |

Result files: `lm_eval_baseline.json`, `lm_eval_blt.json`, `lm_eval_cloze_blt.json`.

BLT is close to GPT-2 on all tasks except LAMBADA. Every fine-tuning intervention has made LAMBADA worse (domain shift + catastrophic forgetting). Cloze training was completely ineffective — 2,661 examples is too small; the model memorizes them without generalizing.

## Planned: pg19 training (data hypothesis test)
WikiText-103 is encyclopedia text; LAMBADA is narrative fiction. The LAMBADA failure
may be data mismatch rather than architecture. pg19 (Project Gutenberg pre-1919 books,
~10.5B tokens, `emozilla/pg19`) is the right domain. Plan: train both 1-M and 2-M on
pg19, compare LAMBADA scores. If both beat WikiText-trained models, data was the problem.

**To run 1-M on pg19**:
```
conda run -n blt python train.py --seed 42 --dataset pg19 \
  --save-path run_pg19_seed42.pt --log-file run_pg19_seed42.log \
  --lr 5e-5 --max-steps 50300 --warmup-steps 200
```

**To run 2-M on pg19**:
```
conda run -n blt python train.py --seed 42 --dataset pg19 --num-m-groups 2 \
  --save-path run_2m_pg19_seed42.pt --log-file run_2m_pg19_seed42.log \
  --lr 5e-5 --max-steps 50300 --warmup-steps 200
```

## Next experiment: Two-M (GQA-like) architecture
**Hypothesis**: The shared-M limitation means all 12 heads use the same attention pattern. Two M matrices (6 heads each) give more head diversity, analogous to GQA grouping.

**Implementation** (already in `model.py`):
- `BLT2Attention`: 2 shared M matrices (M1, M2), each 768×768
- Group 1 (heads 0-5): attention via M1, value projection Wv1 (768→384)
- Group 2 (heads 6-11): attention via M2, value projection Wv2 (768→384)
- Outputs concatenated (768) then projected through Wo
- M1 initialized from Wq[:, 0:384] @ Wk[:, 0:384].T averaged over layers; M2 from heads 6-11

**To run**:
```
conda run -n blt python train.py --seed 42 --num-m-groups 2 \
  --save-path run_2m_seed42.pt --log-file run_2m_seed42.log \
  --lr 5e-5 --max-steps 50300 --warmup-steps 200
```
Then evaluate:
```
conda run -n blt python blt_lm_eval.py \
  --checkpoint run_2m_seed42.pt --num-m-groups 2 \
  --output lm_eval_2m_blt.json
```

## Fixed bugs (sessions 2026-05-26)
- **`set(model.parameters())` ordering bug**: `set()` is unordered, so optimizer state indices shift between runs, corrupting resume. Fixed with stable id-based deduplication in train.py.
- **evaluate.py**: Added `dataset` parameter for LAMBADA val ppl monitoring.
- **train.py**: `--finetune` flag added (weights only, fresh optimizer/scheduler).
- **LAMBADA dataset OOM**: `chunk_size=1` forced when `dataset='lambada'`.
- **blt_lm_eval.py `device` property**: `LM` base class has `device` as read-only; fixed to `self._device`.

## Python environment
Must use the `blt` conda environment. The default conda env has `mytorch` on the path which shadows `torch` and `transformers`.

## Files
- `model.py` — BLTAttention (1-M), BLT2Attention (2-M), build_blt_model(num_m_groups=1|2)
- `train.py` — training harness. Key flags: `--resume`, `--finetune`, `--dataset`, `--baseline`, `--num-m-groups`
- `evaluate.py` — sliding-window perplexity (WikiText-103 or LAMBADA val); compute_cloze_accuracy
- `blt_lm_eval.py` — lm-eval-harness wrapper; supports `--num-m-groups`
- `monitor_plateau.py` — watches a training log and fires a shell command on plateau
- `generate.py` — sanity-check text generation from a checkpoint
- `run_seed42.pt/.log` — BLT WikiText-103 (50,300 steps, val_ppl=21.50) — **primary 1-M checkpoint**
- `run_baseline_seed42.pt/.log` — stopped baseline fine-tune
- `run_lambada_seed42.pt/.log` — LAMBADA full-text fine-tune (plateaued, abandoned)
- `run_cloze_seed42.pt/.log` — cloze fine-tune (killed step 7500, catastrophic overfit)
- `lm_eval_baseline.json` — pretrained GPT-2 benchmark scores
- `lm_eval_blt.json` — BLT after WikiText-103 training
- `lm_eval_cloze_blt.json` — BLT after cloze fine-tune (worse on everything)

## Broader project context
This `blt` branch lives alongside `pprune/` (a KV cache pruning paper). BLT is a separate experiment, likely for comparison purposes.

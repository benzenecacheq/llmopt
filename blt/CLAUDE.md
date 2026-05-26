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
- **Baseline GPT-2 WikiText-103 fine-tune**: Stopped at step 33,930, val_ppl ~14.91. `run_baseline_seed42.pt`. Not resumed (plateaued, decided to focus on LAMBADA).

## Benchmark results (lm-eval-harness)
Stored in `lm_eval_baseline.json` (pretrained GPT-2) and `lm_eval_blt.json` (BLT after WikiText-103 training).

| Task | GPT-2 (pretrained) | BLT |
|------|--------------------|-----|
| LAMBADA acc | 0.242 | 0.114 |
| LAMBADA ppl | 83.0 | 1307.5 |
| HellaSwag acc_norm | 0.291 | 0.275 |
| PIQA acc_norm | 0.560 | 0.541 |
| Winogrande acc | 0.502 | 0.507 |

BLT is close to GPT-2 on all tasks except LAMBADA, where it collapses. Hypothesis: combination of domain mismatch (BLT trained on WikiText-103 encyclopedia text; LAMBADA is book/narrative text) and possible architectural limitation (shared M restricts attention diversity across heads). Running LAMBADA fine-tune to separate these factors empirically.

## Currently running: LAMBADA fine-tune
```
conda run -n blt python train.py --seed 42 --dataset lambada \
  --finetune run_seed42.pt --save-path run_lambada_seed42.pt \
  --log-file run_lambada_seed42.log --lr 1e-5 --max-steps 61500 --warmup-steps 100
```
- At ~step 860 when last checked (step 0 val_ppl = 19.69 on WikiText-103 val)
- Checkpoint every 500 steps to `run_lambada_seed42.pt`
- Monitor running: `monitor_lambada.log` — **BUT THIS IS BROKEN** (see below)

## Fixed bugs (session 2026-05-26)
- **`set(model.parameters())` ordering bug**: `set()` is unordered, so optimizer state indices shift between runs, corrupting resume. Fixed by using stable id-based deduplication (`unique_params` list) everywhere in train.py.
- **evaluate.py**: Added `dataset` parameter so LAMBADA val ppl is monitored during LAMBADA fine-tuning (not WikiText-103 val).
- **train.py**: `--finetune` flag added to load model weights only with fresh optimizer/scheduler (use this when switching datasets; `--resume` for mid-run recovery on same dataset).
- **LAMBADA dataset OOM**: `cimec/lambada` docs average ~90K tokens each; fixed by forcing `chunk_size=1` when `dataset='lambada'`.

## Known issue: plateau monitor watches wrong metric
`monitor_plateau.py` watches val_ppl from `evaluate.py`, which always evaluates on **WikiText-103 validation**. During LAMBADA fine-tuning, WikiText-103 val_ppl rises due to domain shift — the monitor will fire prematurely (~step 2000) thinking it plateaued.

**TODO**: Kill monitor, fix `evaluate.py` to support a `--dataset` argument for LAMBADA val evaluation, restart training with proper monitoring. We are only at ~step 860 so restart cost is low.

## To restart LAMBADA fine-tune after fixing evaluate.py
1. Kill any running monitor_plateau.py and train.py processes
2. Fix evaluate.py to support LAMBADA val split
3. Run:
```
conda run -n blt python train.py --seed 42 --dataset lambada \
  --finetune run_seed42.pt --save-path run_lambada_seed42.pt \
  --log-file run_lambada_seed42.log --lr 1e-5 --max-steps 61500 --warmup-steps 100
```
4. Run monitor with correct val metric

## To resume LAMBADA fine-tune (if it crashed mid-run, after evaluate.py is fixed)
```
conda run -n blt python train.py --seed 42 --dataset lambada \
  --resume run_lambada_seed42.pt --save-path run_lambada_seed42.pt \
  --log-file run_lambada_seed42.log --lr 1e-5 --max-steps 61500
```

## Python environment
Must use the `blt` conda environment. The default conda env has `mytorch` on the path which shadows `torch` and `transformers`.

## Files
- `model.py` — BLTAttention and build_blt_model()
- `train.py` — training harness. Key flags: `--resume` (full state), `--finetune` (weights only, fresh optimizer), `--dataset {wikitext103,lambada}`, `--baseline` (use GPT-2 instead of BLT)
- `evaluate.py` — sliding-window perplexity on WikiText-103 validation (needs LAMBADA support added)
- `blt_lm_eval.py` — lm-eval-harness wrapper for benchmark evals
- `monitor_plateau.py` — watches a training log and fires a shell command on plateau
- `generate.py` — sanity-check text generation from a checkpoint
- `run_seed42.log` / `run_seed42.pt` — completed BLT WikiText-103 run
- `run_baseline_seed42.log` / `run_baseline_seed42.pt` — stopped baseline fine-tune
- `run_lambada_seed42.log` / `run_lambada_seed42.pt` — LAMBADA fine-tune (in progress)
- `monitor_lambada.log` — plateau monitor log (currently broken, watching wrong metric)
- `lm_eval_baseline.json` / `lm_eval_blt.json` — benchmark results

## Broader project context
This `blt` branch lives alongside `pprune/` (a KV cache pruning paper). BLT is a separate experiment, likely for comparison purposes.

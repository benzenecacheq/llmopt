# BLT Training — Session Context

## What this is
BLT (Bilinear Attention) is a research experiment replacing GPT-2's standard multi-head attention with a parameter-sharing bilinear variant. This is the **GD-only control condition** — all parameters trained with Adam (no alternative optimizer).

The key architectural idea: replace per-layer Wq and Wk with a single shared M matrix (768×768) initialized as the average of Wq @ Wk^T across all 12 GPT-2 layers. Since M is shared, X @ M @ X^T is computed once per layer (not once per head), saving 11/12 of the attention score computation vs standard MHA.

## Important decisions made
- **Scale factor**: `sqrt(d_head)` = `sqrt(64)` not `sqrt(d_model)` = `sqrt(768)`. The wrong scale was caught before the main run — it makes softmax temperature 3.5× off and would skew loss comparisons.
- **Baseline target**: Pretrained GPT-2 perplexity on WikiText-103 validation = **26.39** (measured with sliding-window eval, stride=512).
- **Training duration**: 2 epochs = ~50,300 steps (~15 hours on P100 at ~1.1s/step).
- **Logging**: All output goes to `run_seed42.log` (file-buffered, line by line). Nothing to stdout. Monitor with `tail -f run_seed42.log`.

## Currently running
Training run: `conda run -n blt python train.py --seed 42 --save-path run_seed42.pt`

- 50,300 steps (2 epochs over WikiText-103)
- Checkpoints every 500 steps to `run_seed42.pt` (overwrites)
- Loss logged every 10 steps, val perplexity every 500 steps

## To resume after an outage
```
conda run -n blt python train.py --seed 42 --save-path run_seed42.pt --resume run_seed42.pt
```

## Python environment
Must use the `blt` conda environment. The default conda env has `mytorch` on the path which shadows `torch` and `transformers`.

## Files
- `model.py` — BLTAttention and build_blt_model()
- `train.py` — training harness with resume support
- `evaluate.py` — sliding-window perplexity on WikiText-103 validation
- `run_seed42.log` — live training log
- `run_seed42.pt` — latest checkpoint (saved every 500 steps)

## Broader project context
This `blt` branch lives alongside `pprune/` (a KV cache pruning paper). BLT is a separate experiment, likely for comparison purposes.

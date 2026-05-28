# BLT Training — Session Context

## What this is
BLT (Bilinear Attention) is a research experiment replacing GPT-2's standard multi-head attention with a parameter-sharing bilinear variant. This is the **GD-only control condition** — all parameters trained with Adam (no alternative optimizer).

The key architectural idea: replace per-layer Wq and Wk with a single shared M matrix (768×768). The attention score between positions i and j is x_i^T M x_j / scale, computed as (X @ M) @ X^T. M is shared across all 12 layers; Wv and Wo remain per-layer.

## Important decisions made
- **Scale factor**: `sqrt(d_head)` = `sqrt(64)` not `sqrt(d_model)` = `sqrt(768)`. The wrong scale was caught before the main run — it makes softmax temperature 3.5× off and would skew loss comparisons.
- **Baseline target**: Pretrained GPT-2 perplexity on WikiText-103 validation = **26.39** (measured with sliding-window eval, stride=512).
- **M initialization**: Can be warm-started from average of Wq @ Wk^T across GPT-2 layers, OR randomly initialized N(0, 1/sqrt(D)). Random init works fine — M learns well from scratch.

## Key lessons learned (2026-05-28)
- **Local minima trap**: Fine-tuning from GPT-2 pretrained weights keeps the model in an encyclopedia-text basin. LAMBADA fine-tuning made scores *worse* due to catastrophic forgetting. Random M init + OWT training avoids this.
- **BLT works**: 48% fewer attention parameters, better WikiText-103 ppl than pretrained GPT-2, and LAMBADA well above GPT-2 baseline when trained on appropriate data.
- **LAMBADA failure was domain mismatch, not architecture**: WikiText-trained BLT scored 0.114; OWT-trained BLT with random M scored 0.182 at step 126K (still training). Head diversity from multiple M matrices is NOT the bottleneck — data domain is.
- **Loss/benchmark mismatch**: Cross-entropy on next-token prediction optimizes equally over all tokens. LAMBADA tests hard, long-range predictions that are a tiny fraction of training signal. See Token Weighting paper (arXiv 2503.09202, NAACL 2025) for a principled fix.

## Completed runs
- **BLT WikiText-103**: DONE. `run_seed42.pt`, 50,300 steps, final val_ppl = **21.50** (beats GPT-2 baseline of 26.39).
- **Baseline GPT-2 WikiText-103 fine-tune**: Stopped at step 33,930, val_ppl ~14.91. `run_baseline_seed42.pt`. Not resumed (plateaued).
- **BLT LAMBADA full-text fine-tune**: `run_lambada_seed42.pt`. Plateaued ~step 3500. LAMBADA benchmark acc 0.102 (worse than 0.114 before fine-tune). Abandoned.
- **BLT cloze fine-tune**: `run_cloze_seed42.pt`. Killed at step 7500. Catastrophic overfitting — train loss → 0 on 2,661 examples, LAMBADA benchmark acc collapsed to 0.065. Abandoned.
- **BLT 2-M WikiText-103**: Monitored to step 20K (val_ppl=20.95, LAMBADA acc=0.106). No final checkpoint on disk.

## Active run (2026-05-28)
**BLT 1-M, OpenWebText, random M init** — `run_owt_randm_seed42.pt/.log`
- Command: `conda run -n blt python train.py --seed 42 --dataset openwebtext --random-m --save-path run_owt_randm_seed42.pt --log-file run_owt_randm_seed42.log --lr 5e-5 --max-steps 250000 --warmup-steps 200`
- Dataset: first 1M docs of Skylion007/openwebtext (~1B tokens, ~1 epoch at 250K steps)
- Status at last check: **step ~178,000 of 250,000**, val_ppl ~48.6, LAMBADA cloze acc ~0.55 (200-sample estimate)
- Full lm-eval at step 126K: LAMBADA acc=**0.182**, HellaSwag=0.280, PIQA=0.572, Winogrande=0.503
- Result file: `lm_eval_owt_randm_126k.json`
- To resume if interrupted: `--resume run_owt_randm_seed42.pt`

## Benchmark results (lm-eval-harness)

| Task | GPT-2 (pretrained) | BLT 1-M (WikiText) | BLT 2-M (step 20K) | BLT OWT random-M (step 126K) |
|------|--------------------|--------------------|---------------------|-------------------------------|
| LAMBADA acc | 0.242 | 0.114 | 0.106 | **0.182** |
| LAMBADA ppl | 83.0 | 1307.5 | 975.6 | 243.0 |
| HellaSwag acc_norm | 0.291 | 0.275 | 0.271 | 0.280 |
| PIQA acc_norm | 0.560 | 0.541 | 0.547 | **0.572** |
| Winogrande acc | 0.502 | 0.507 | 0.499 | 0.503 |

Result files: `lm_eval_baseline.json`, `lm_eval_blt.json`, `lm_eval_cloze_blt.json`, `lm_eval_2m_blt.json`, `lm_eval_lambada_blt.json`, `lm_eval_owt_randm_126k.json`.

## Python environment
Must use the `blt` conda environment. **PyTorch downgraded to 2.3.1+cu121** (transformers 5.x requires PyTorch ≥ 2.4, but V100 GPU is CC 7.0 which is not supported by PyTorch ≥ 2.4). Transformers downgraded to 4.46.3.

GPU: Tesla V100-DGXS-16GB (16GB VRAM, CC 7.0).

## New flags added (2026-05-28)
- `train.py --random-m`: initialize M ~ N(0, 1/sqrt(D)) instead of Wq@Wk^T average
- `train.py --dataset openwebtext`: loads first 1M docs from Skylion007/openwebtext
- `train.py --lambada-eval-every N` (default 2000): logs LAMBADA cloze acc alongside val_ppl
- `save_checkpoint()`: now writes atomically via .tmp + os.replace; keeps .bak as fallback

## Architecture variants
- `model.py build_blt_model(num_m_groups=1)`: single shared M (original BLT)
- `model.py build_blt_model(num_m_groups=2)`: two shared M matrices, heads 0-5 and 6-11
- `model.py build_blt_model(random_m=True)`: random M initialization

## Future directions discussed
- **Grouped Wv**: share Wv across groups of heads (GQA-style) to reduce bandwidth further. G groups of value heads, each D×d, concatenated then projected by Wo. Reduces per-layer attention bandwidth by ~3.6× vs current BLT.
- **Token weighting loss**: upweight tokens requiring long-range context using a short-context reference model (arXiv 2503.09202). Most promising fix for LAMBADA/benchmark mismatch.
- **KV cache**: BLT's cache stores raw hidden states x_j as keys (no Wk multiply needed) + standard values. Not yet implemented.

## Files
- `model.py` — BLTAttention (1-M), BLT2Attention (2-M), build_blt_model(num_m_groups, random_m)
- `train.py` — training harness. Key flags: `--resume`, `--finetune`, `--dataset`, `--baseline`, `--num-m-groups`, `--random-m`, `--lambada-eval-every`
- `evaluate.py` — sliding-window perplexity (WikiText-103 or LAMBADA val); compute_cloze_accuracy
- `blt_lm_eval.py` — lm-eval-harness wrapper; supports `--num-m-groups`
- `paper_blt.md` — draft paper covering BLT architecture, results, and related work
- `run_seed42.pt/.log` — BLT WikiText-103 (50,300 steps, val_ppl=21.50)
- `run_owt_randm_seed42.pt/.log` — **active run**: OWT random-M, step ~178K of 250K
- `lm_eval_owt_randm_126k.json` — full benchmark at step 126K of active run

## Broader project context
This `blt` branch lives alongside `pprune/` (a KV cache pruning paper). BLT is a separate experiment exploring parameter-efficient attention alternatives.

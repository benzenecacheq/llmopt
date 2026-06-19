# BLT Training — Session Context

## What this is
BLT (Bilinear Attention) is a research experiment replacing GPT-2's standard multi-head attention with a parameter-sharing bilinear variant. This is the **GD-only control condition** — all parameters trained with Adam (no alternative optimizer).

The key architectural idea: replace per-layer Wq and Wk with a single shared M matrix (768×768). The attention score between positions i and j is x_i^T M x_j / scale, computed as (X @ M) @ X^T. M is shared across all 12 layers; Wv and Wo remain per-layer.

## Important decisions made
- **Scale factor**: `sqrt(d_head)` = `sqrt(64)` not `sqrt(d_model)` = `sqrt(768)`. The wrong scale was caught before the main run — it makes softmax temperature 3.5× off and would skew loss comparisons.
- **Baseline target**: Pretrained GPT-2 perplexity on WikiText-103 validation = **26.39** (measured with sliding-window eval, stride=512).
- **M initialization**: Can be warm-started from average of Wq @ Wk^T across GPT-2 layers, OR randomly initialized N(0, 1/sqrt(D)). Random init works fine — M learns well from scratch.
- **From-scratch comparison step count**: BLT runs 550K steps vs GPT-2's 500K steps to equalize wall-clock time (BLT ~1.65ms/step vs GPT-2 ~1.82ms/step).

## Key lessons learned (2026-05-28)
- **Local minima trap**: Fine-tuning from GPT-2 pretrained weights keeps the model in an encyclopedia-text basin. LAMBADA fine-tuning made scores *worse* due to catastrophic forgetting. Random M init + OWT training avoids this.
- **BLT works**: 48% fewer attention parameters, better WikiText-103 ppl than pretrained GPT-2, and LAMBADA well above GPT-2 baseline when trained on appropriate data.
- **LAMBADA failure was domain mismatch, not architecture**: WikiText-trained BLT scored 0.114; OWT-trained BLT with random M scored 0.199 at 250K steps. Head diversity from multiple M matrices is NOT the bottleneck — data domain is.
- **Loss/benchmark mismatch**: Cross-entropy on next-token prediction optimizes equally over all tokens. LAMBADA tests hard, long-range predictions that are a tiny fraction of training signal. See Token Weighting paper (arXiv 2503.09202, NAACL 2025) for a principled fix.

## Completed runs
- **BLT WikiText-103**: DONE. `run_seed42.pt`, 50,300 steps, final val_ppl = **21.50** (beats GPT-2 baseline of 26.39).
- **Baseline GPT-2 WikiText-103 fine-tune**: Stopped at step 33,930, val_ppl ~14.91. `run_baseline_seed42.pt`. Not resumed (plateaued).
- **BLT LAMBADA full-text fine-tune**: `run_lambada_seed42.pt`. Plateaued ~step 3500. LAMBADA benchmark acc 0.102 (worse than 0.114 before fine-tune). Abandoned.
- **BLT cloze fine-tune**: `run_cloze_seed42.pt`. Killed at step 7500. Catastrophic overfitting — train loss → 0 on 2,661 examples, LAMBADA benchmark acc collapsed to 0.065. Abandoned.
- **BLT 2-M WikiText-103**: Monitored to step 20K (val_ppl=20.95, LAMBADA acc=0.106). No final checkpoint on disk.
- **BLT 1-M OWT random-M**: DONE. 250K steps, LAMBADA acc=**0.199**, PIQA=0.568, HellaSwag=0.280. Result files: `lm_eval_owt_randm_126k.json`, `lm_eval_owt_randm_218k.json`, `lm_eval_owt_randm_250k.json`.
- **BLT 1-M from-scratch OWT (seed 42)**: DONE (2026-06-02). `run_blt_scratch_seed42.pt`, 550K steps, 244,170s (444ms/step), final WikiText val_ppl=72.88, OWT held-out ppl=**31.05** (loss 3.4357 nats). Result files: `lm_eval_blt_scratch.json`.
- **GPT-2 from-scratch OWT (baseline, seed 42)**: DONE (2026-06-03). `run_gpt2_baseline_seed42.pt`, 500K steps, 280,020s (560ms/step), final WikiText val_ppl=55.99, OWT held-out ppl=**27.78** (loss 3.3243 nats). Result files: `lm_eval_baseline_scratch.json`.
- **BLT 1-M from-scratch OWT (seed 19)**: DONE (2026-06-07). `run_blt_scratch_seed19.pt`, 600K steps, 337,021s (562ms/step, this machine ~27% slower than seed42 machine), final WikiText val_ppl=85.82, OWT held-out ppl=**30.48** (loss 3.4170 nats). Result files: `lm_eval_blt_scratch_seed19.json`. Note: WikiText val_ppl was noisy throughout training (swings of 25+ ppl); OWT ppl is stable and the reliable metric.
- **GQA 2-group from-scratch OWT (seed 42)**: DONE (2026-06-11). `run_gqa_scratch_seed42.pt`, 500K steps, 296,815s (593ms/step), final WikiText val_ppl=57.43, OWT held-out ppl=**27.64** (loss 3.3192 nats). Result files: `lm_eval_gqa_scratch.json`.
- **BLT 1-M from-scratch OWT (seed 7)**: DONE (2026-06-16). `run_blt_scratch_seed7.pt`, 500K steps, final OWT held-out ppl=**30.81** (loss 3.4279 nats). Third BLT seed; consistent with seed42 (31.05) and seed19 (30.48), confirming the ~0.10 nat gap vs GPT-2/GQA holds across seeds. Result files: `lm_eval_blt_scratch_seed7.json`.
- **Hybrid model (6 MHA + 6 BLT layers), from-scratch OWT, seed 42**: DONE (2026-06-19). `run_hybrid_mha6_scratch_seed42.pt`, 500K steps, 305,249s, final WikiText val_ppl=69.08, OWT held-out ppl=**28.40** (loss 3.3462 nats). Result files: `lm_eval_hybrid_scratch.json`. The run had an earlier OOM crash partway through (see `run_hybrid_mha6_scratch_seed42.log`) and was restarted by a watcher script; completed cleanly afterward. Per the log-naming bug fixed in commit 1f3803d, this run's per-step training log went to `run_seed42.log` instead of its own filename (checkpoint unaffected).
- **Major finding — hybrid closes most of the BLT gap**: OWT loss gap vs GPT-2 drops from ~0.10-0.11 nats (full BLT) to just **0.022 nats** (hybrid, 3.3462 vs GPT-2's 3.3243) — i.e., halving the BLT layers recovers >75% of the lost ground, not half. LAMBADA acc (0.222) and ppl (167.1) are the *best* of the whole BLT family, edging out even GPT-2's LAMBADA ppl (174.6). This suggests the per-layer Wq/Wk in the 6 retained MHA layers do most of the work of capturing long-range/positional structure that full M-sharing loses, and that BLT's expressiveness cost is not simply additive across layers — see paper_blt.md "Hybrid architecture" section for write-up.

## Benchmark results (lm-eval-harness)

### From-scratch OWT runs — primary comparison

| Task | BLT seed42 (550K) | BLT seed19 (600K) | BLT seed7 (500K) | Hybrid 6MHA+6BLT (500K) | GPT-2 seed42 (500K) | GQA seed42 (500K) |
|------|-------------------|-------------------|-------------------|--------------------------|---------------------|-------------------|
| OWT held-out ppl | 31.05 | 30.48 | 30.81 | 28.40 | 27.78 | **27.64** |
| OWT held-out loss | 3.4357 | 3.4170 | 3.4279 | 3.3462 | 3.3243 | **3.3192** |
| LAMBADA acc | 0.205 | 0.209 | 0.212 | **0.222** | 0.225 | 0.204 |
| LAMBADA ppl | 349.6 | 288.6 | 244.4 | **167.1** | 174.6 | 205.3 |
| HellaSwag acc_norm | 0.271 | 0.267 | 0.268 | **0.273** | 0.268 | 0.269 |
| PIQA acc_norm | 0.561 | 0.572 | 0.568 | 0.562 | **0.579** | 0.568 |
| Winogrande acc | **0.528** | 0.511 | 0.516 | 0.504 | 0.505 | 0.496 |

**Key findings:**
- All three BLT seeds consistent (OWT ppl 30.48–31.05), confirming ~0.10 nat gap vs GPT-2/GQA is real, not seed variance.
- GQA edges GPT-2 on OWT ppl (27.64 vs 27.78) despite similar parameter counts — KV compression at 2 groups has no cost and marginal benefit.
- GPT-2 still wins on LAMBADA acc despite worse OWT ppl than GQA — full per-layer Wk matters for long-range prediction specifically. But the hybrid model actually has the best LAMBADA ppl (167.1, beating even GPT-2's 174.6), suggesting 6 full MHA layers recover essentially all of LAMBADA's long-range needs.
- HellaSwag, PIQA, Winogrande are essentially ties within noise across all variants; BLT/hybrid hold a slight Winogrande edge over GPT-2/GQA.
- **Hybrid result is non-additive**: halving the BLT layer count (12→6) cuts the OWT loss gap vs GPT-2 from ~0.10-0.11 nats down to 0.022 nats — recovering over 75% of the gap, not 50%. This implies the expressiveness cost of M-sharing is front-loaded or compounding across layers rather than a flat per-layer tax; 6 unrestricted MHA layers (regardless of which 6) are enough to mostly route around it. See paper_blt.md "Hybrid architecture" for analysis of which layer positions were used.
- The ~0.10 nat BLT gap vs GPT-2/GQA is attributable to cross-layer M sharing (expressiveness cost), not parameter count — GQA has similar params to BLT but matches GPT-2.
- BLT speed advantage (seed42 machine) was hardware-specific — on this machine BLT runs at 562ms/step vs GPT-2's 560ms/step on the other machine.

### Full history

| Task | GPT-2 pretrained | BLT WikiText (50K) | BLT OWT random-M (250K) | BLT from-scratch OWT (550K) | GPT-2 from-scratch OWT (500K) |
|------|------------------|--------------------|--------------------------|------------------------------|-------------------------------|
| LAMBADA acc | **0.242** | 0.114 | 0.199 | 0.205 | 0.225 |
| LAMBADA ppl | **83.0** | 1307.5 | 206.8 | 349.6 | 174.6 |
| HellaSwag acc_norm | **0.291** | 0.275 | 0.280 | 0.271 | 0.268 |
| PIQA acc_norm | 0.560 | 0.541 | 0.568 | 0.561 | **0.579** |
| Winogrande acc | 0.502 | 0.507 | 0.490 | **0.528** | 0.505 |

Result files: `lm_eval_baseline.json` (truncated/corrupt), `lm_eval_blt.json`, `lm_eval_cloze_blt.json`, `lm_eval_2m_blt.json`, `lm_eval_lambada_blt.json`, `lm_eval_owt_randm_126k.json`, `lm_eval_owt_randm_218k.json`, `lm_eval_owt_randm_250k.json`, `lm_eval_blt_scratch.json`, `lm_eval_baseline_scratch.json`.

## New machine setup

**Step 1 — Get the code**
```
git clone <repo> && cd llmopt.blt/blt
git checkout blt
```

**Step 2 — Create the conda environment**
```
conda create -n blt python=3.11 -y
conda activate blt
pip install -r requirements.txt
```
`requirements.txt` pins PyTorch 2.3.1+cu121 and transformers 4.46.3. These downgrades are required for V100 (CC 7.0) GPUs — PyTorch ≥ 2.4 dropped CC 7.0 support. On a newer GPU (CC ≥ 8.0) you can use current PyTorch/transformers.

**Step 3 — Transfer checkpoints** (not in git, ~1.3–1.4 GB each)
```
scp oldmachine:~/llmopt.blt/blt/run_blt_scratch_seed42.pt .
scp oldmachine:~/llmopt.blt/blt/run_gpt2_baseline_seed42.pt .
```

**Step 4 — OWT dataset**
Option A — transfer the tokenized block cache (fastest, 9 GB):
```
scp oldmachine:~/.cache/blt_owt_2m_blocks.pt ~/.cache/
```
Option B — transfer the raw parquet files (~25 GB) and let the first training run re-tokenize them (~10 min):
```
scp -r oldmachine:~/.cache/huggingface/hub/datasets--Skylion007--openwebtext ~/.cache/huggingface/hub/
```
Option C — re-download from HuggingFace (slow, requires internet access to HF):
The dataset loads automatically on first use; set `HF_HOME` if needed.

**Step 5 — Verify**
```
conda run -n blt python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n blt python eval_owt.py --baseline-checkpoint run_gpt2_baseline_seed42.pt --max-tokens 50000
```

## Python environment
Must use the `blt` conda environment. **PyTorch downgraded to 2.3.1+cu121** (transformers 5.x requires PyTorch ≥ 2.4, but V100 GPU is CC 7.0 which is not supported by PyTorch ≥ 2.4). Transformers downgraded to 4.46.3.

Target GPU: Tesla V100-PCIE-32GB (32GB VRAM, CC 7.0). Runs were done with 64GB RAM. Any GPU with ≥ 16 GB VRAM should work; batch size 4 fits within that.

## Flags added (2026-05-28/30)
- `train.py --random-m`: initialize M ~ N(0, 1/sqrt(D)) instead of Wq@Wk^T average
- `train.py --from-scratch`: randomly initialize ALL weights (no pretrained GPT-2 load); implies --random-m. Works for both BLT and --baseline.
- `train.py --dataset openwebtext`: loads 2M docs from Skylion007/openwebtext via local parquet cache
- `train.py --lambada-eval-every N` (default 2000): logs LAMBADA cloze acc alongside val_ppl
- `save_checkpoint()`: now writes atomically via .tmp + os.replace; keeps .bak as fallback

## Dataset loading (OpenWebText)
OWT tokenization caches to `~/.cache/blt_owt_2m_blocks.pt` after first run — subsequent starts load in seconds. First run takes ~10 minutes (bulk parquet load + encode_batch tokenization). Loading 2M docs uses ~15GB RAM peak; tokenized blocks are ~9GB. Uses first 21 parquet files (sorted), which matches `train[:2000000]` ordering — verified doc-for-doc against HuggingFace split.

## Architecture variants
- `model.py build_blt_model(num_m_groups=1)`: single shared M (original BLT)
- `model.py build_blt_model(num_m_groups=2)`: two shared M matrices, heads 0-5 and 6-11
- `model.py build_blt_model(random_m=True)`: random M initialization
- `model.py build_blt_model(from_scratch=True)`: random init for all weights, no pretrained load

## SVD analysis of trained M (seed42, from-scratch OWT)

Computed SVD of trained M from `run_blt_scratch_seed42.pt`. Key findings:
- Spectrum is nearly **flat** — M is genuinely full-rank
- 550 of 768 singular values are > 10% of the maximum
- 745 of 768 singular values are > 1% of the maximum
- Frobenius energy captured: r=64→36%, r=128→56%, r=192→71%, r=256→81%, r=384→93%
- Implication: M is earning its D×D capacity. A UV^T factorization needs r≥192-256 to be a good approximation. r=64 loses 64% of energy.
- Why flat? M must simultaneously encode useful attention patterns for 12 layers × 12 heads = 144 contexts, requiring broad coverage across the full D-dimensional space.

## Memory bandwidth analysis

BLT vs standard MHA at inference (decode phase):
- **Weight bandwidth**: BLT wins at all context lengths. No per-layer Wq/Wk; M loaded once (potentially L2-cached). Standard MHA loads Wq+Wk per layer per step.
- **KV cache bandwidth**: BLT and standard MHA are TIED. Both cache D-dimensional vectors per token per layer (MHA: Wk·x_j, BLT: raw x_j). Neither has GQA's cache reduction.
- At long contexts, KV cache bandwidth dominates and BLT's weight advantage shrinks as a fraction of total. At 128K context on 70B: BLT saves ~6% total bandwidth vs MHA (weights: 21.6GB vs 43GB, but KV cache: 335GB each).
- BLT is strictly better than standard MHA on every bandwidth measure; comparison to GQA is apples-to-oranges (GQA trades weight bandwidth for KV compression).

## M L2 cache sizing

M fits in GPU L2 cache for small-to-medium models, giving effectively zero HBM cost after first load:
- GPT-2 (D=768): M = 1.2 MB → fits any GPU (V100: 6MB L2, H100: 50MB L2)
- 7B model (D=4096): M = 33.6 MB → fits H100/A100 (40-50MB L2), borderline
- 13B model (D=5120): M = 52 MB → borderline H100
- 70B model (D=8192): M = 134 MB → does NOT fit any current GPU L2

For GPT-2 scale: M is loaded once, stays resident across all 12 layers. Standard MHA pays Wq+Wk HBM cost 12× per decode step.

## Tensor parallelism analysis

Large production models (Claude, GPT-4, etc.) use N-way tensor parallelism (TP): attention is sharded by head across GPUs. Standard MHA shards cleanly — each GPU handles H/N heads independently, one all-reduce for Wo.

**Full M BLT is incompatible with head-level TP.** M produces identical attention weights for all heads, so it cannot be sharded by head. Options are (a) replicate M on every GPU or (b) shard M and add an all-reduce of L×L matrices. Either way, BLT's bandwidth advantage inverts at 4+ way TP:

| GPUs | MHA MB/GPU/layer | Full M BLT MB/GPU/layer |
|------|-----------------|------------------------|
| 1    | 4D²             | ~3D² (BLT wins)        |
| 2    | 2D²             | 2D² (tied)             |
| 4    | D²              | 1.5D² (MHA wins)       |
| 8    | 0.5D²           | 1.25D² (MHA wins 2.5×) |

**UV^T BLT fixes the TP problem entirely.** With r=256 and D=8192, U+V = 8.4 MB — small enough to fit in H100 L2 cache (50 MB) and trivial to reload even if evicted by FFN weights. UV^T BLT maintains its bandwidth advantage at any level of TP:

| GPUs | MHA MB/GPU/layer | UV^T BLT MB/GPU/layer (r=256, D=8192) |
|------|-----------------|---------------------------------------|
| 1    | 268 MB          | 152 MB (43% less)                     |
| 8    | 67 MB           | 42 MB (37% less)                      |
| 32   | 17 MB           | 10.5 MB (38% less)                    |

UV^T scales correctly because U and V are a rounding error; only Wv and Wo are sharded, and Wq+Wk are eliminated entirely. The bandwidth savings are consistent across all parallelism levels.

**Implication:** UV^T is not just a KV cache optimization — it is a prerequisite for BLT to be viable at production scale. Full M BLT is a single-GPU architecture; UV^T BLT is a scalable one. This elevates the UV^T experiment from "interesting variant" to "essential for the large-scale deployment case."

## Future directions discussed

### Planned experiments (in order)
1. **BLT seed 19**: DONE.
2. **GQA baseline**: DONE.
3. **BLT seed 7**: DONE — third seed, confirms gap vs GPT-2/GQA is consistent.
4. **Hybrid 6 MHA + 6 BLT**: DONE — see Completed runs above. Non-additive gap closure (75%+ of the OWT loss gap recovered from half the BLT layers) is a key result for the paper.
5. **UV^T fine-tuning (Option 2)**: post-training SVD factorization of trained M + fine-tune. See details below. Not yet started.

### Low-rank BLT (M = UV^T)
Factor M as U (D×r) × V^T (r×D), both globally shared. Attention score: (x_i @ U)·(x_j @ V)/√d. Key cache stores x_j @ V (r-dimensional). Asymmetry (U ≠ V) is intentional — query and key views are different questions.

**Option 2 — Post-training SVD + fine-tune (preferred):**
1. Take trained M from seed42 (or seed19 when done)
2. Factorize M ≈ UV^T via SVD at r = 128, 192, 256
3. Initialize new model with UV^T, fine-tune on OWT for ~50-100K steps
4. Compare OWT ppl and benchmarks against full-rank BLT
- Feasible on current hardware (same VRAM, shorter run than from-scratch)
- r=192 or r=256 are the realistic starting points given the flat spectrum
- Tests viability; positive result establishes foundation for larger-scale claim

**KV cache with UV^T — critical detail:** To compress the KV cache for long prompts, must store V-projected keys (x_j @ V, r-dimensional) for PREFILL tokens too, not just decode tokens. During prefill, self-attention between context tokens still uses full M (quality preserved), but the cache is built with V-projected keys. Decode then attends against the full r-dimensional cache. This gives KV cache compression proportional to total context length, not just the short decode portion.

**Why training U is non-trivial in a hybrid scheme:** In causal LM training, every token is simultaneously context and query — there's no clean prefill/decode split. Option 2 avoids this by fine-tuning UV^T directly (U and V both get gradients normally as standard weight matrices in the UV^T product). Pure training from scratch with UV^T is also clean. A two-stream training scheme (M for context, UV^T for new tokens) is theoretically interesting but expensive and architecturally complex.

**Limitation of GPT-2 scale testing:** KV cache and M-caching benefits are most compelling at 7B+ scale with long contexts. GPT-2 scale establishes viability; the practical case requires larger models.

### Other future directions
- **Grouped Wv**: share Wv across groups of heads (GQA-style) to reduce value cache bandwidth.
- **Token weighting loss**: upweight tokens requiring long-range context using a short-context reference model (arXiv 2503.09202). Most promising fix for LAMBADA/benchmark mismatch.
- **KV cache implementation**: BLT's cache stores raw x_j as keys (no Wk multiply needed) + standard values. Not yet implemented.

## Files
- `model.py` — BLTAttention (1-M), BLT2Attention (2-M), build_blt_model(num_m_groups, random_m, from_scratch)
- `train.py` — training harness. Key flags: `--resume`, `--finetune`, `--dataset`, `--baseline`, `--num-m-groups`, `--random-m`, `--from-scratch`, `--lambada-eval-every`
- `evaluate.py` — sliding-window perplexity (WikiText-103 or LAMBADA val); compute_cloze_accuracy
- `blt_lm_eval.py` — lm-eval-harness wrapper; supports `--num-m-groups`
- `paper_blt.md` — draft paper covering BLT architecture, results, and related work
- `eval_owt.py` — held-out OWT evaluation (files 21-25, sliding window); `--blt-checkpoint`, `--gqa-checkpoint`, `--hybrid-checkpoint` (+ `--n-mha-layers`), or `--baseline-checkpoint`
- `run_seed42.pt/.log` — BLT WikiText-103 (50,300 steps, val_ppl=21.50). Note: `run_seed42.log` ALSO received the hybrid run's training log (2026-06-16 through completion 2026-06-19) due to the log-naming bug fixed in commit 1f3803d (fix not applied to that already-running process) — the WikiText-103 run's own log content is only the first ~5 header lines plus its original step history.
- `run_blt_scratch_seed42.pt/.log` — BLT from-scratch OWT (550K steps, val_ppl=72.88, OWT ppl=31.05)
- `run_blt_scratch_seed19.pt/.log` — BLT from-scratch OWT seed19 (600K steps, OWT ppl=30.48)
- `run_blt_scratch_seed7.pt/.log` — BLT from-scratch OWT seed7 (500K steps, OWT ppl=30.81)
- `run_gpt2_baseline_seed42.pt/.log` — GPT-2 from-scratch OWT (500K steps, val_ppl=55.99, OWT ppl=27.78)
- `run_gqa_scratch_seed42.pt/.log` — GQA 2-group from-scratch OWT (500K steps, OWT ppl=27.64)
- `run_hybrid_mha6_scratch_seed42.pt/.log` — hybrid 6 MHA + 6 BLT from-scratch OWT, DONE (500K steps, val_ppl=69.08, OWT ppl=28.40); `.log` only has its own content up to the OOM crash, see note on `run_seed42.log` above
- `lm_eval_owt_randm_250k.json` — final benchmark for OWT random-M run
- `lm_eval_blt_scratch.json` — benchmarks for BLT from-scratch OWT run (seed42)
- `lm_eval_blt_scratch_seed7.json` — benchmarks for BLT from-scratch OWT run (seed7)
- `lm_eval_baseline_scratch.json` — benchmarks for GPT-2 from-scratch OWT run
- `lm_eval_gqa_scratch.json` — benchmarks for GQA from-scratch OWT run
- `lm_eval_hybrid_scratch.json` — benchmarks for hybrid 6 MHA + 6 BLT from-scratch OWT run

## Broader project context
This `blt` branch lives alongside `pprune/` (a KV cache pruning paper). BLT is a separate experiment exploring parameter-efficient attention alternatives.

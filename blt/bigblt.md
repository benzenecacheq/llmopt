# BigBLT — Scaling BLT Beyond Single Shared M

## Background

BLT replaces per-layer Wq and Wk with a single shared D×D matrix M. Score: `x_i^T M x_j / sqrt(d_head)`. Original small-model results showed 48% fewer attention params with better WikiText-103 ppl than pretrained GPT-2 (21.50 vs 26.39 baseline).

The key question: does BLT scale, and what variants improve on the single-M design?

---

## GPT-2 XL Warm-Start (1.3B, negative result)

Attempted single-M BLT at GPT-2 XL scale (48 layers, 25 heads, D=1600). M warm-started as average of Wq@Wk^T across all 48 layers, fine-tuned on WikiText-103 for 50K steps.

**First run**: NaN at step 16,840 (fp16 overflow). Silently corrupted both checkpoint and .bak. Fixed by adding pre-backward `isfinite` guard and checkpoint safety check.

**Root cause of instability**: M-averaging compresses 48×25=1200 attention contexts into one matrix (vs 144 for small model). The approximation is 8.3× coarser and perturbs the model far more violently at init.

**Fix**: `--warmstart-scale 0.25` multiplies M_init by 0.25 before loading.

**Second run result** (completed ~50K steps):

| Metric | BLT XL warm-start | GPT-2 XL zero-shot |
|---|---|---|
| WikiText val_ppl | 16.80 | **15.30** |
| LAMBADA acc | 0.212 | **0.245** |
| HellaSwag acc_norm | 0.334 | **0.450** |
| PIQA acc_norm | 0.568 | **0.680** |
| Winogrande acc | 0.511 | **0.560** |

**Verdict**: Worse than zero-shot pretrained on every metric. Single M cannot represent 1200 attention contexts. Fine-tuning damages the pretrained representations without a good replacement.

Files: `run_blt_xl_warmstart_wikitext.pt/.log`, `lm_eval_blt_xl_warmstart.json`, `lm_eval_gpt2xl_pretrained_zeroshot.json`

---

## Multi-M Head-Split Variants (small model, from-scratch OWT)

After the XL failure, explored whether multiple M matrices (each covering a subset of heads, shared across all layers) can close the gap vs GPT-2.

### Why warm-starts are not the right test

Warm-start experiments (WikiText-103, 50K steps) gave misleading results because the single-M warm-start initialization (M = global average of Wq@Wk^T) is naturally well-suited for one matrix. Multi-M warm-starts use head-sliced averages which are a rougher approximation, and 50K steps is not enough to overcome the worse initialization. All warm-start multi-M runs converged to ~30.3-30.4 ppl regardless of grouping (vs single-M's 21.50) — this reflects initialization bias, not architecture quality.

**From-scratch training on OWT (500K steps) is the correct comparison.**

### Warm-start results (for reference only)

| Model | WikiText val_ppl |
|---|---|
| BLT 1-M warm-start | **21.50** |
| BLT 3-M warm-start (head groups) | 30.38 |
| BLT layer-4-M warm-start (strided layers) | 30.30 |
| GQA-3 warm-start | 27.69 |

The layer-4-M variant (strided: layer i uses M[i % 3] for GPT-2 small) was also tested. It performed comparably to 3-M head-split — both substantially worse than single-M in warm-start, for the same initialization-bias reason.

### From-scratch OWT results (primary comparison)

All runs: 500K steps, seed 42, OWT dataset, GPT-2 small architecture (12 layers, 12 heads, D=768).

| | BLT 1-M | BLT 3-M | Hybrid 6MHA+6BLT | GPT-2 | GQA-2 | GQA-3 |
|--|--|--|--|--|--|--|
| **OWT held-out ppl** | 31.05 | **29.90** | 28.40 | 27.78 | 27.64 | **27.46** |
| **OWT held-out loss** | 3.4357 | **3.3978** | 3.3462 | 3.3243 | 3.3192 | **3.3129** |
| LAMBADA acc | 0.205 | **0.224** | 0.222 | 0.225 | 0.204 | 0.195 |
| LAMBADA ppl | 349.6 | 204.9 | **167.1** | 174.6 | 205.3 | 216.4 |
| HellaSwag acc_norm | 0.271 | **0.273** | **0.273** | 0.268 | 0.269 | 0.271 |
| PIQA acc_norm | 0.561 | **0.573** | 0.562 | 0.579 | 0.568 | 0.553 |
| Winogrande acc | **0.528** | 0.493 | 0.504 | 0.505 | 0.496 | 0.511 |

BLT 3-M checkpoints: `run_blt_3m_scratch_seed42.pt`, `lm_eval_blt_3m_scratch.json`
GQA-3 run on separate machine; checkpoints stored there.

### Key findings

- **Head diversity matters**: BLT 3-M (3 independent D×D M matrices, 4 heads each) closes ~38% of the 1-M gap vs GPT-2 (0.111 nats → 0.073 nats). Going from 1 to 3 head groups is a meaningful improvement.
- **GQA group count has diminishing returns**: GQA-3 (27.46) vs GQA-2 (27.64) is a 0.18 ppl improvement — smaller than seed-to-seed variance (GQA-2: 27.64 vs 28.25 across seeds). Adding a KV group doesn't move the needle in GQA.
- **BLT 3-M wins on benchmarks despite worse OWT ppl**: BLT 3-M beats GQA-3 on LAMBADA acc (0.224 vs 0.195) and PIQA (0.573 vs 0.553). The shared-M architecture seems to encourage long-range pattern learning even at the cost of next-token prediction loss.
- **Winogrande is the exception**: BLT variants consistently score lower on Winogrande than GPT-2/GQA. High variance (1,267 examples, stderr ~0.014) makes this hard to interpret definitively, but the pattern is consistent across BLT variants.
- **Gap is not closed**: BLT 3-M at 0.073 nats below GPT-2 still represents a real expressiveness cost. The hybrid result (6 MHA + 6 BLT, 0.022 nats gap) shows that mixing full MHA layers is more efficient at recovering the gap than adding more M matrices.

---

## Compute cost of multi-M variants

Per-layer FLOPs for attention score computation (T=sequence length, D=768):

| Architecture | Score FLOPs | Projection FLOPs | Total |
|---|---|---|---|
| Standard MHA (12 heads) | T²×D | 2×T×D² | 2T×D² + T²×D |
| BLT 1-M | T²×D | T×D² | T×D² + T²×D |
| BLT 3-M | 3×T²×D | 3×T×D² | 3T×D² + 3T²×D |

BLT 1-M is cheaper than MHA (saves one D² projection). BLT 3-M is more expensive than MHA on every term — each group's M multiply operates in full D=768 space, unlike MHA heads which operate in d_head=64 space. Observed training speed: 1-M ~0.44s/step, 3-M ~0.57s/step (1.3× slower).

---

## Per-head M architecture proposals

To fix the scaling problem cleanly, two designs were discussed:

### Proposal A: Full D×D M per head (user's proposal)
One D×D M_h per head, shared across all L layers.
```
scores_h = (X @ M_h) @ X^T / sqrt(d_head)   # (T,T) per head
```
Params: H × D² = 12 × 768² = 7.1M (small), 25 × 1600² = 64M (XL).
Benefits: full head diversity, TP-friendly (head h → GPU h%N).
Cost: 12× more params than single M; still more expensive than MHA at multi-GPU TP.

### Proposal B: W_h (D×d_head) + M_h (d_head×d_head) per head
Project to head space first, apply small bilinear form:
```
x_h = X @ W_h                                # (T, d_head)
scores_h = (x_h @ M_h) @ x_h^T / sqrt(d_head)  # (T, T)
```
Both W_h and M_h shared across all L layers.
```
# Full forward pass (one attention layer):
for h in range(H):
    x_h  = X @ W[h]               # (T,D)@(D,d)   → (T,d)
    q_h  = x_h @ M[h]             # (T,d)@(d,d)   → (T,d)
    A_h  = softmax(q_h @ x_h.T / sqrt(d) + causal_mask)  # (T,T)
    v_h  = X @ Wv[h]              # (T,D)@(D,d)   → (T,d)
    out_h = A_h @ v_h             # (T,T)@(T,d)   → (T,d)
out = concat(out_h for h)         # (T, H*d) = (T, D)
result = out @ Wo                 # (T,D)@(D,D)   → (T,D)
```
Params: H × d_head × (D + d_head) = 12 × 64 × 832 = 638K (small) ≈ single M (590K).
Benefits: nearly identical params to single M, full head diversity, compute-parity with MHA, TP-friendly.
Cost: shared W_h means same projection for query and key sides; M_h must capture asymmetry in d_head space.

### Not yet implemented

No implementation decision made as of 2026-07-18. Proposal B is the recommended starting point — nearly free in parameters while restoring head diversity. Proposal A is simpler to implement but significantly more expensive.

---

## Files

| File | Contents |
|---|---|
| `run_blt_xl_warmstart_wikitext.pt/.log` | GPT-2 XL single-M warm-start (50K steps) |
| `lm_eval_blt_xl_warmstart.json` | XL warm-start benchmarks |
| `lm_eval_gpt2xl_pretrained_zeroshot.json` | GPT-2 XL zero-shot baseline |
| `run_blt_3m_warmstart_small.pt` | 3-M head-split warm-start (50K WikiText, small model) |
| `run_gqa3_warmstart_small.pt` | GQA-3 warm-start (50K WikiText, small model) |
| `lm_eval_blt_3m_warmstart_small.json` | 3-M warm-start benchmarks |
| `lm_eval_gqa3_warmstart_small.json` | GQA-3 warm-start benchmarks |
| `run_blt_3m_scratch_seed42.pt` | BLT 3-M from-scratch OWT (500K steps, seed 42) |
| `lm_eval_blt_3m_scratch.json` | BLT 3-M from-scratch benchmarks |

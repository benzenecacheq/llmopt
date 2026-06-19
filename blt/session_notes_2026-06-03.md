# BLT Session Notes — 2026-06-03

## Overfitting check

Added `--train-set` flag to `eval_owt.py` to evaluate on training files (0-4) instead of held-out files (21-25).

Results (500K tokens, 5 files):

| Split | BLT from-scratch (550K) | GPT-2 from-scratch (500K) |
|-------|-------------------------|---------------------------|
| Train-set loss | 3.3872 nats | 3.2689 nats |
| Train-set ppl  | 29.58       | 26.28       |
| Held-out loss  | 3.4357 nats | 3.3243 nats |
| Held-out ppl   | 31.05       | 27.78       |

**Conclusion:** No overfitting. Train/held-out gap is ~0.05 nats for both models. The ~0.11 nat advantage for GPT-2 is consistent across both splits.

## Timing analysis

Actual per-step times from log files (not the estimates in CLAUDE.md):

| Model | Steps | Wall-clock | ms/step |
|-------|-------|------------|---------|
| BLT from-scratch | 550K | 244,170s (67.8h) | ~444ms |
| GPT-2 from-scratch | 500K | 280,020s (77.8h) | ~560ms |

BLT is ~26% faster per step (CLAUDE.md had estimated ~10%). At equal wall-clock time, BLT could have run ~631K steps vs the 550K actually run — 81K steps left on the table. The original step-count equalization was based on inaccurate timing estimates.

## Statistical validity concern

With n=1 per architecture, the ~0.11 nat gap cannot be distinguished from run-to-run seed variance. Multiple seeds needed to confirm.

## New run: BLT seed 19

Started a second BLT from-scratch OWT run to address both issues (seed variance + step count):

- Seed: 19
- Max steps: 600K
- Flags: `--from-scratch --dataset openwebtext --eval-every 1000 --lambada-eval-every 5000 --save-path run_blt_scratch_seed19.pt`
- Log: `run_seed19.log` (rename to `run_blt_scratch_seed19.log` when done)
- Per-step time on this machine: ~482ms (this machine ~10% slower than original)
- 600K steps at 482ms ≈ 289,200s (~80h) — comparable wall-clock to GPT-2's 500K steps on the faster machine

**Rationale:** If longer training closes the gap, BLT's inference-time savings (26% faster per forward pass) outweigh the one-time training cost differential.

## Next steps

1. Let seed 19 run to 600K steps
2. Run lm-eval benchmarks on seed 19 checkpoint
3. Consider running GPT-2 seed 19 for a proper paired comparison
4. Implement GQA baseline (2-group) after seed 19 completes
5. Run UV^T fine-tuning experiment (Option 2) — see below

---

# Architecture discussion — 2026-06-04

## Memory bandwidth analysis

BLT vs standard MHA (no GQA):
- Weight bandwidth: BLT strictly wins at all context lengths (~2× reduction, no per-layer Wq/Wk)
- KV cache: BLT and MHA are tied (both D-dimensional per token per layer)
- At long contexts, KV cache dominates and BLT's relative advantage shrinks
- M L2 caching: GPT-2 M=1.2MB fits any GPU L2; 7B M=33.6MB fits H100 (50MB L2); 70B M=134MB does not fit

## SVD analysis of trained M

M from `run_blt_scratch_seed42.pt` has a nearly flat singular value spectrum — genuinely full-rank.
- 550/768 singular values > 10% of max
- r=256 captures only 81% of Frobenius energy
- M is using its full D×D capacity, not converging to a natural low-rank solution
- Implication: UV^T factorization needs r≥192-256 to be a reasonable approximation

## Low-rank UV^T direction

Three options for training UV^T:
1. Pure UV^T from scratch — clean, but loses full-rank capacity
2. **Post-training SVD + fine-tune (preferred)** — factorize trained M, fine-tune UV^T for ~50-100K steps. Feasible on current hardware. Start with r=192 or r=256.
3. Two-stream training (M for prefill, UV^T for decode) — theoretically interesting, training is non-trivial, deferred

Key KV cache insight: to compress the cache for long prompts, must store V-projected keys (x_j @ V) for PREFILL tokens too — not just decode tokens. This gives r-dimensional key cache for the entire context, not just the short generated portion. Prefill self-attention still uses full M for quality; only the cache changes.

Asymmetry (U ≠ V) is intentional and important — query and key views are different.

## Experiment sequencing

1. Seed 19 BLT (in progress, ~600K steps)
2. GQA 2-group GPT-2 baseline (after seed 19)
3. UV^T fine-tuning Option 2 (after GQA baseline)

Note: GPT-2 scale establishes viability of UV^T; practical inference case requires 7B+ models which exceed current hardware.

## Tensor parallelism — UV^T is essential at scale

Full M BLT cannot be sharded by head in tensor parallel inference — M produces the same attention weights for all heads, so it must be replicated on every GPU. At 8-way TP on 70B, full M BLT uses 2.5× MORE bandwidth than standard MHA. BLT's single-GPU advantage inverts at 4+ way TP.

UV^T fixes this completely. U+V at r=256, D=8192 = 8.4 MB — fits in H100 L2 cache (50MB). Only Wv and Wo are sharded; U and V are a rounding error. UV^T BLT is 37-43% better than standard MHA at any TP level.

Key reframing: UV^T is not just a KV cache optimization — it is a prerequisite for BLT to be viable at production scale. Full M BLT is a single-GPU architecture. This elevates the UV^T experiment significantly.

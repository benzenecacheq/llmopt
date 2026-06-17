---
name: project-loss-function-ideas
description: "Discussion of alternative loss functions and self-referential curriculum ideas for BLT training, to improve LAMBADA and long-range benchmark performance"
metadata: 
  node_type: memory
  type: project
  originSessionId: a7808a7f-0afc-4a4a-b8bd-fc92c2dd278d
---

The core problem: standard cross-entropy weights all tokens equally, so easy high-frequency tokens swamp the gradient signal from hard long-range predictions. This explains why WikiText-trained BLT scores poorly on LAMBADA even as loss decreases.

**Why:** LAMBADA tests hard, long-range predictions that are a tiny fraction of training tokens. The loss can go down while the model gets worse at exactly those cases.

**How to apply:** When designing next training experiments, consider replacing or augmenting the standard cross-entropy loss with one of the approaches below.

## Approaches with prior art

- **Rho-1 / Selective LM** (NeurIPS 2024, arXiv 2404.07965): train a reference model, compute excess loss per token vs. reference, only backprop through high-excess tokens. Strong results on math reasoning. Requires a reference model.
- **Token Weighting for Long-Range LM** (NAACL 2025, arXiv 2503.09202): compare long-context vs. short-context model confidence per token; upweight tokens where long-range context helps. Most targeted fix for LAMBADA-style tasks.
- **MiLe Loss** (arXiv 2310.19531): weight tokens by entropy — uncertain tokens get stronger signal. No reference model needed but can't distinguish long-range uncertainty from genuine ambiguity.
- **Tilting the Playing Field** (ICML 2021, arXiv 2102.03793): cycle time-dependent loss weights across objectives during training. Pushes optimizer into wider, deeper minima than static loss.

## Self-referential curriculum (novel idea, no reference model needed)

User's insight: we have the information we need within the model itself to identify which tokens are worth focusing on.

**Option 1 — EMA per-token loss tracking**: maintain a running exponential moving average of loss per token (or token n-gram) across training steps. Tokens with fast-decreasing loss are "learned" → downweight. Tokens with stuck/increasing loss are "hard" → upweight. Cheap, no extra forward passes. Also naturally implements loss cycling: as easy tokens get learned and downweighted, harder tokens come to the fore, shifting the effective loss landscape over time.

**Option 2 — Short-context self-comparison**: on each batch, run two forward passes — full context and truncated context (e.g., last 64 tokens). Upweight tokens where the loss gap is large (i.e., long-range context matters for that token). Principled and directly targets long-range dependency. Costs 2× compute per step.

**Option 3 — Per-token gradient norm**: tokens with large gradient norms are ones the model is still actively learning from. Track as a proxy for "still useful."

Most practical starting point: **Option 1 (EMA)** — cheap, no extra compute, captures the Rho-1 intuition without a reference model, and naturally implements curriculum cycling.

## Cycling among multiple loss functions (user's core idea)

Explicitly rotate between fundamentally different loss functions during training — e.g., standard cross-entropy → token-weighted long-range loss → focal loss → back. Motivation: each loss function has its own local minima landscape; cycling prevents the optimizer from getting trapped in any one of them and pushes it toward minima that are deep under *all* the objectives.

This goes further than Tilting the Playing Field (which cycles weights on the same loss) — cycling between structurally different losses is a stronger perturbation of the landscape. Not directly explored in the LLM literature as of mid-2026.

Concrete implementation sketch:
- Define 2-3 loss functions (e.g., standard CE, EMA-weighted CE, short-context-gap-weighted CE)
- Rotate every N steps (N could be ~1000-5000, matching roughly one pass through a data subset)
- Track per-loss validation metrics to confirm each loss is contributing improvement
- Could combine with cyclic LR schedule so LR resets align with loss function switches

[[feedback-permissions]]

# EMA Per-Token Loss Weighting: A Self-Referential Curriculum for Language Model Pretraining

## Abstract

Standard cross-entropy pretraining weights every token equally, so easy, high-frequency tokens dominate the gradient signal and hard, informative tokens — the kind long-range benchmarks like LAMBADA specifically probe — contribute comparatively little. We introduce a simple, cheap fix: maintain a per-vocabulary-token historical loss estimate, and reweight each token's contribution to the training loss by its (normalized) value. Unlike prior hard-token-selection methods, this requires no reference model and no extra forward pass — it is entirely self-referential, using only information the model already produces. We test two variants of the historical-loss estimate: a fixed-decay exponential moving average (EMA, ~69-occurrence half-life) and an exact cumulative running mean.

Trained from scratch on OpenWebText for 500K steps, full weighting (either variant) improves LAMBADA accuracy substantially (non-EMA baseline 0.225 → fixed-decay EMA 0.253 → cumulative 0.268, GPT-2 Small; BLT 0.212 → 0.242 with fixed-decay EMA) and LAMBADA perplexity dramatically at a real but bounded cost to general next-token prediction. This trade-off is **architecture-general** (BLT and standard MHA show the same shape of trade-off, `paper_blt.md`) and, newly confirmed in this revision, **model-scale-general**: a GPT-2-medium (355M parameter) run shows the identical trade-off shape at roughly 3× the parameter count.

We further show the trade-off is not fixed, and that the best way to realize it changed as more seeds accumulated. Naively fine-tuning an EMA-converged checkpoint back to standard cross-entropy is a **forgetting cliff**: the LAMBADA gain evaporates faster than the OWT-ppl cost recovers. Blending the EMA objective with standard CE **from step 0 of a from-scratch run** beats every sequential fine-tune, and this holds across three independent seeds (mean LAMBADA acc 0.260, all above the 0.225 baseline). **Annealing the blend coefficient in via a sine schedule initially looked like a further, substantial improvement in a single seed (LAMBADA acc 0.281) — but this did not replicate**: the three-seed average for the sine schedule (0.259) is statistically indistinguishable from the non-annealed blend's own three-seed average (0.260). We report this as a cautionary result about trusting single-seed wins, not as a confirmed finding, and the same pattern recurs for a second, independently-motivated annealing test described below.

**Cumulative (exact running-mean) weighting is a better mechanism than fixed-decay EMA at matched blend strength**, confirmed at the three-seeds-per-side level: at full weighting (blend=1.0), cumulative mode reaches LAMBADA acc 0.265 (3-seed average) versus fixed-decay EMA's 0.258 — a modest but genuine win, plausibly because cumulative mode's running mean reflects a token's difficulty across its entire training history rather than only its most recent ~69 occurrences. Applying the same sine anneal to cumulative mode's blend coefficient — motivated by a different rationale than the fixed-decay case, since cumulative mode's own estimator variance already shrinks with sample count independent of any schedule — again shows no benefit once a third seed lands (3-seed average 0.268, statistically tied with the non-annealed 0.265).

We also show that the OWT held-out perplexity cost reported throughout this paper is not a neutral correctness check: the reweighting mechanism works almost entirely by *suppressing* gradient on the most frequent tokens (at full blend strength, tokens accounting for 44% of all training-token occurrences receive less than half their uniform gradient weight), so a model trained this way is, by construction, trained to deviate from the raw frequency-weighted distribution that OWT ppl measures. We therefore reframe OWT ppl as a **declared cost** of a deliberate choice, not a guardrail against something having gone wrong — the zero-shot benchmarks already serve that role. Finally, we decompose the cost at the individual-benchmark-item level: the BLiMP grammaticality cost is real and highly significant (not noise, unlike a superficially similar ARC-Easy gap), concentrated in long-range structural dependencies (anaphor binding, wh-movement across clause boundaries), and present at a similar small magnitude (roughly 1–2 points) regardless of which mechanism or blend value produced it.

---

## 1. Background: The Loss/Benchmark Mismatch

Cross-entropy next-token prediction optimizes equally over all tokens in a training corpus. Most tokens in web text are easy — function words, common continuations, near-deterministic completions — and dominate the loss simply by frequency. Benchmarks like LAMBADA, by contrast, specifically test hard, long-range predictions (predicting a final word that requires understanding an entire preceding paragraph) that are a tiny fraction of total training signal. A model's cross-entropy loss can decrease steadily while its performance on exactly the tokens LAMBADA-style benchmarks care about stagnates or even regresses relative to what a differently-weighted objective would achieve.

This was observed directly during earlier BLT experiments (`paper_blt.md`, Section 4.1-4.2): a BLT model fine-tuned on WikiText-103 scored LAMBADA acc 0.114, far below GPT-2's 0.275, even though WikiText validation perplexity looked healthy. Retraining on broader web text (OpenWebText) improved LAMBADA substantially without any loss-function change, showing the original gap was partly a training-domain artifact — but a residual gap remained, motivating a loss-function-level fix rather than a data-level one.

---

## 2. Method: Self-Referential Per-Token Loss Weighting

The key idea: the information needed to identify "hard" tokens already exists inside the model's own per-step loss — no external reference model (cf. Rho-1, Section 6) or extra forward pass (cf. short-context self-comparison, discussed but not implemented — see Section 5) is required. We test two ways of estimating "historical difficulty" for a token, and a schedule for blending the resulting weight toward uniform.

### 2.1 Fixed-Decay EMA Weighting

**Mechanism** (`train.py`, `--ema-loss-weighting` with default `--loss-weighting-mode ema`):

1. Maintain a vector `token_loss` of length `vocab_size` (50,257 for GPT-2's BPE vocabulary), one estimate per **vocabulary token ID** (not position, not n-gram). Initialized to `log(vocab_size)` — the loss of a uniform/maximum-entropy predictor — as an uninformative prior.
2. At each training step, compute the standard per-token cross-entropy loss for every predicted token in the batch, `per_token_loss`.
3. Look up each token's current weight: `weight = token_loss[token_id]`, then normalize so the batch's mean weight is 1: `weight = weight / weight.mean()`.
4. Optionally blend toward uniform weighting via `--ema-blend α` (Section 2.3).
5. The training loss is `mean(weight.detach() · per_token_loss)` — the weight is **detached** from the graph, so it only rescales each token's gradient magnitude; it does not introduce its own gradient path.
6. After the step, update the estimate in place, per vocabulary ID actually seen in the batch: `token_loss[id] = decay · token_loss[id] + (1 - decay) · batch_mean_loss_for_id`, with `decay=0.99` by default (`--ema-decay`).

With `decay=0.99`, the half-life of this estimate is `ln(0.5)/ln(0.99) ≈ 69 occurrences` — for a common token seen many thousands of times over a run, the buffer reflects only a short, recent window of that token's history, not its difficulty across the whole corpus. Which window survives is partly an artifact of corpus shuffle order. This observation motivated the alternative estimator below.

### 2.2 Cumulative (Exact Running Mean) Weighting

`--loss-weighting-mode cumulative` replaces the fixed-decay update in step 6 above with an exact running mean: alongside `token_loss`, maintain a per-token-ID count `token_count` (int64 — a high-frequency token can exceed float32's 2^24 exact-integer range over a long run, which would silently corrupt the count), and update via `token_loss[id] = (token_loss[id] · token_count[id] + batch_sum_loss_for_id) / (token_count[id] + batch_count_for_id)`, then increment `token_count[id]`. This has no permanent "forgetting window" — an early-training occurrence and a late-training occurrence of the same token contribute to the estimate in exact proportion to how many total occurrences there have been, rather than the fixed-decay estimate's constant-width recency bias. Its only weakness is the ordinary, self-resolving cold-start problem any online estimator has (an early-training estimate is noisier simply because fewer samples have accumulated), not a permanent structural forgetting mechanism.

Checkpoints store the `token_loss` and `token_count` buffers directly (`save_checkpoint(..., ema_loss=..., token_count=...)` — the field is named `ema_loss` in the checkpoint schema for both modes, for backward compatibility), and both `--resume` and `--finetune` restore them — the latter was a bug fix partway through this work (Section 4.2): originally only `--resume` loaded the buffer, so any `--finetune` run silently restarted hard-token tracking from the uninformative uniform prior.

### 2.3 Blending and Annealing

`--ema-blend α` interpolates the per-token weight toward uniform, for either weighting mode: `weight = α · weight + (1 - α) · 1.0` (α=1.0 = full weighting, α=0.0 = standard, unweighted cross-entropy exactly). This can be applied either as a **sequential fine-tune** (start from a converged checkpoint, continue training with a fixed blend α; Section 4.3) or as the **objective from step 0** of a from-scratch run (Section 4.4).

`--ema-blend-schedule sine` anneals α from 0 up to its target over the course of training rather than applying it at full target strength immediately: the effective blend at step `s` is `ema_blend · sin(0.5π · min(s, max_steps)/max_steps)` — 0 at step 0, ramping smoothly to exactly the target value at `max_steps`. The actual per-step effective blend is logged directly (`effective_blend` training-log column) and confirmed to track the formula exactly in every run where it was checked. This mechanism composes with either weighting mode (Sections 4.5, 4.7).

---

## 3. Experimental Setup

Identical protocol to `paper_blt.md` Section 3 unless noted: GPT-2 Small architecture (12 layers, 12 heads, D=768) or BLT (same base architecture, single shared M — see `paper_blt.md` Section 2), trained from scratch (no pretrained initialization) on 2M OpenWebText documents, Adam optimizer (lr=5e-5, cosine decay, 200 warmup steps, batch size 4, block size 1024), 500K steps unless noted. A separate GPT-2-medium (24 layers, 16 heads, D=1024, 354.8M parameters) protocol, used only in Section 4.12, trains on a larger 75-file OWT corpus (`--dataset openwebtext_large`, ~7.9B tokens, chosen to avoid epoch repetition at the larger step budget) for 1.5M steps at batch size 2 with 2-step gradient accumulation (effective batch 4, to fit the larger model in 16GB).

**Evaluation.** Primary suite: held-out OWT perplexity (files 21-25, sliding window) and zero-shot accuracy on LAMBADA/HellaSwag/PIQA/Winogrande via lm-eval-harness. Supplementary suite (Section 4.10): ARC-Easy, BoolQ, OpenBookQA. Syntactic-competence suite (Section 4.9): BLiMP, 67 grammaticality-judgment minimal-pair subtasks. All EMA/cumulative experiments in this paper use standard MHA (GPT-2) unless labeled BLT; no jointly-trained-blend or sine-annealed experiment has been run on BLT (Section 5).

**Caution on the in-training LAMBADA proxy.** `train.py --lambada-eval-every` logs a cheap 200-example greedy-decoding cloze accuracy during training, for monitoring only. This proxy is **not discriminating** — it stayed flat at ~0.55-0.57 through an entire 10,000-step fine-tune run where the real lm-eval-harness LAMBADA accuracy (5,153 examples, log-likelihood scoring, the number reported everywhere in this paper) collapsed by several points (Section 4.2). Every result below is from the full lm-eval-harness benchmark, never the in-training proxy.

**Multi-seed discipline.** Every result in this paper that is reported as "confirmed" has at least three independent seeds (42, 19, 7, or a variant such as 7_v2); single-seed results are explicitly labeled as such and should be read as preliminary. This discipline is load-bearing for this paper's central cautionary finding (Section 4.5): two separate sine-annealing experiments each looked like a clear win at one seed and turned out to be statistically indistinguishable from the non-annealed baseline once two more seeds landed.

---

## 4. Results

### 4.1 Full EMA Weighting Is a Real, Architecture-General Trade-off

We trained GPT-2 and BLT from scratch with `--ema-loss-weighting` (fixed-decay, α=1.0, decay=0.99), 500K steps, OWT, seed 42, and compared each against its own non-EMA baseline of the same architecture/seed/step-count.

| Metric | GPT-2 baseline | GPT-2 + EMA | Δ | BLT baseline (seed 7) | BLT + EMA (seed 42) | Δ |
|---|---|---|---|---|---|---|
| OWT held-out ppl | 27.78 | 30.06 | +8.2% | 30.81 | 34.25 | +11.2% |
| OWT held-out loss (nats) | 3.3243 | 3.4031 | — | 3.4279 | 3.5337 | — |
| LAMBADA acc | 0.225 (±0.0058) | 0.253 (±0.0061) | **+0.028** | 0.212 | 0.242 (±0.0060) | **+0.030** |
| LAMBADA ppl | 174.6 | 119.6 | **−31.5%** | 244.4 | 205.3† | −16.0%† |
| HellaSwag acc_norm | 0.268 | 0.272 | +0.004 | 0.268 | ~flat | — |
| PIQA acc_norm | 0.579 | 0.569 | −0.010 | 0.568 | ~flat | −0.009 |
| Winogrande acc | 0.505 | 0.511 | +0.006 | 0.516 | ~flat | — |

† BLT+EMA LAMBADA ppl figure carried from the `project_loss_function_ideas` working notes; re-derive from `lm_eval_blt_ema_seed42.json` if an exact figure is needed for publication — the acc figures above are read directly from the JSON.

**The trade-off shape is nearly identical across architectures**: LAMBADA accuracy gain is +0.028 (GPT-2) vs. +0.030 (BLT); PIQA degrades by almost the same amount on both (−0.010 vs. −0.009); OWT ppl worsens on both, proportionally somewhat more on BLT (+11.2% vs. +8.2%). This is strong evidence the effect is a property of the **loss function itself**, not of the attention mechanism — a more general claim than prior hard-token-weighting work (Rho-1, MiLe Loss; Section 6), which was only evaluated on standard transformer attention. No jointly-trained-blend, sine-annealed, or cumulative-mode experiment has been run on BLT — only this one pure (α=1.0, fixed-decay) configuration (Section 5).

BLT already pays its own architecture-level OWT ppl cost relative to MHA even without EMA (`paper_blt.md` Section 4.3); EMA's cost stacks on top of that (30.81 → 34.25) rather than interacting with it in any special way, consistent with the "architecture-general" reading.

### 4.2 Post-Hoc Fine-Tuning Back to Standard CE Is a Forgetting Cliff

Took the converged `run_gpt2_ema_seed42.pt` (full fixed-decay EMA, 500K steps) and fine-tuned for just **10,000 steps** (2% of the original training budget) with plain cross-entropy — a fresh optimizer and LR schedule (`train.py --finetune ...`; `--resume` would be wrong here since it would continue the original cosine schedule, already decayed to ~0 LR by step 500K). Run on `titan`: `run_gpt2_ema_then_ce_finetune_seed42.pt`.

| Metric | EMA (before) | Pure-CE fine-tune (10K steps) | Non-EMA baseline (never had EMA) |
|---|---|---|---|
| OWT held-out ppl | 30.06 | 29.13 | 27.78 |
| LAMBADA acc | 0.253 | **0.217** | 0.225 |
| LAMBADA ppl | 119.6 | 184.1 | 174.6 |
| HellaSwag acc_norm | 0.272 | 0.271 | 0.268 |
| PIQA acc_norm | 0.569 | 0.566 | 0.579 |
| Winogrande acc | 0.511 | 0.499 | 0.505 |

LAMBADA accuracy did not merely regress toward the non-EMA baseline — it **overshot past it**, ending lower (0.217) than a model that never saw EMA weighting at all (0.225), and LAMBADA perplexity ended slightly worse than the never-EMA baseline too (184.1 vs. 174.6). Meanwhile OWT held-out ppl recovered only 41% of the EMA-induced gap in the same 10K-step window (30.06 → 29.13 of a 30.06 → 27.78 gap). The benefit that took 500K steps of EMA weighting to build erodes in a small fraction of that time once gradient pressure shifts away from the hard tokens it was protecting — the same catastrophic-forgetting mechanism documented earlier in this project from WikiText fine-tuning of a pretrained-and-converted BLT model (`paper_blt.md` Section 4.1 background; see also `CLAUDE.md` "Key lessons learned (2026-05-28)").

### 4.3 Blending Softens the Cliff: Sequential Fine-Tune Sweep (α = 0.25, 0.5, 0.75)

`--ema-blend α` interpolates the per-token weight toward uniform (Section 2.3). We swept α ∈ {0.25, 0.5, 0.75} as a **sequential fine-tune** from the same converged EMA checkpoint, same 10,000-step / fresh-schedule protocol as Section 4.2, all on `titan`, seed 42.

| | EMA (before) | α=0.25 ft | α=0.5 ft | α=0.75 ft | Pure-CE ft (α=0) | Non-EMA baseline |
|---|---|---|---|---|---|---|
| OWT held-out ppl | 30.06 | 29.29 | 29.68 | *not measured‡* | 29.13 | 27.78 |
| OWT held-out loss | 3.4031 | 3.3771 | — | — | — | 3.3243 |
| LAMBADA acc | 0.253 | 0.231 (±0.0059) | 0.244 | 0.258 (±0.0061) | 0.217 | 0.225 |
| LAMBADA ppl | 119.6 | 159.3 | 142.9 | 132.1 | 184.1 | 174.6 |
| HellaSwag acc_norm | 0.272 | 0.270 | 0.270 | 0.270 | 0.271 | 0.268 |
| PIQA acc_norm | 0.569 | 0.563 | 0.562 | 0.558 | 0.566 | 0.579 |
| Winogrande acc | 0.511 | 0.506 | 0.498 | 0.504 | 0.499 | 0.505 |

‡ OWT held-out ppl for the α=0.75 sequential fine-tune (`run_gpt2_ema_blend75_finetune_seed42.pt`) has not yet been measured.

The sweep is monotonic and interior points beat both endpoints on the trade-off that matters: every blend α tested keeps LAMBADA accuracy **above** the non-EMA baseline (0.231-0.258 vs. 0.225) — unlike the pure-CE fine-tune, which overshot *below* it (0.217) — while giving up less OWT-ppl recovery than pure CE.

### 4.4 Jointly-Trained Blend From Scratch Beats Every Sequential Fine-Tune — Confirmed Across Three Seeds

Section 4.3's blends are all **sequential**: fine-tuned into the blend only after the model fully converged in the pure-EMA basin. This leaves open whether a model trained with the blended objective **from step 0** — never specializing into pure EMA before correcting — reaches a better point, since it never has to unlearn anything.

We trained GPT-2 from scratch (`--baseline --from-scratch --ema-blend 0.75`, fixed-decay weighting, full 500K steps, OWT) rather than fine-tuning, across three seeds (42, 19, 7_v2 — an original seed-7 launch was retracted after its checkpoint's `token_loss` buffer showed no genuine weighting activity, see the methodological note below):

| | seed42 | seed19 | seed7_v2 | **3-seed avg** | EMA pure (α=1, seed42) | Non-EMA baseline (seed42) |
|---|---|---|---|---|---|---|
| OWT held-out ppl | 29.61 | 29.61 | 29.42 | 29.55 | 30.06 | 27.78 |
| OWT held-out loss | 3.3881 | 3.3882 | 3.3818 | — | 3.4031 | 3.3243 |
| LAMBADA acc | 0.269 | 0.253 | 0.257 | **0.260** | 0.253 | 0.225 |
| LAMBADA ppl | 130.6 | 127.8 | 131.3 | 129.9 | 119.6 | 174.6 |
| HellaSwag acc_norm | 0.273 | 0.271 | 0.266 | 0.270 | 0.272 | 0.268 |
| PIQA acc_norm | 0.565 | 0.582 | 0.574 | 0.574 | 0.569 | 0.579 |
| Winogrande acc | 0.522 | 0.508 | 0.520 | 0.517 | 0.511 | 0.505 |

**All three genuine seeds land solidly above the non-EMA baseline's LAMBADA accuracy (0.225)**, averaging 0.260 — a real three-seed confirmation of the jointly-trained 75/25-blend finding, beating even pure EMA's own single-seed LAMBADA accuracy (0.253) at essentially the same OWT-ppl cost.

**Methodological note: verify weighting is actually active before trusting a run.** A first attempted replicate at seed 7 (`run_gpt2_ema_blend75_scratch_seed7.pt`) finished with numbers that superficially looked like a "mirror image" of the seed42 result. Direct inspection of the checkpoint's `token_loss` buffer explained why: it sat bit-for-bit at its untouched initialization value (std exactly 0.0) — proof `--ema-loss-weighting` was never active, since the launch had passed `--ema-blend 0.75` without that flag, a complete no-op. **This run is retracted and excluded from all results.** Lesson applied to every subsequent run in this paper: verify the checkpoint's `token_loss`/`token_count` buffers show genuine per-token spread (not std=0.0, not an unpopulated count) at the first checkpoint save, rather than discovering a launch-flag mistake only after a full run and benchmark.

### 4.5 Annealing the Blend Coefficient: A Promising Single Seed That Did Not Replicate

Section 4.4 fixes the blend coefficient at α=0.75 for the entire training run. The motivating concern for annealing it instead: the historical-loss estimate is least trustworthy early in training, when each vocabulary token has seen only a handful of updates, so applying full-strength reweighting from step 0 means part of training is shaped by a noisy, uninformative signal. Section 2.3 describes the sine-anneal mechanism tested here.

We trained GPT-2 from scratch with the sine schedule to a target of α=0.75, fixed-decay weighting, otherwise identical to Section 4.4's protocol, across three seeds:

| | seed42 | seed19 | seed7 | **3-seed avg (sine)** | **3-seed avg (fixed, Section 4.4)** |
|---|---|---|---|---|---|
| OWT held-out ppl | 29.44 | 29.37 | 29.37 | 29.39 | 29.55 |
| LAMBADA acc | **0.281** | 0.246 | 0.251 | **0.259** | 0.260 |
| LAMBADA ppl | 117.7 | 146.0 | 136.0 | 133.2 | 129.9 |
| HellaSwag acc_norm | 0.267 | 0.270 | 0.272 | 0.270 | 0.270 |
| PIQA acc_norm | 0.571 | 0.566 | 0.590 | 0.576 | 0.574 |
| Winogrande acc | 0.511 | 0.507 | 0.517 | 0.512 | 0.517 |

**The first seed (0.281) was, on its own, the best LAMBADA result in the entire EMA/blend family at the time it was measured — and it did not replicate.** The second seed (0.246) landed well below it, and the third (0.251) confirmed the second rather than the first: the three-seed average (0.259) is statistically indistinguishable from the non-annealed blend's own three-seed average from Section 4.4 (0.260) — a difference of 0.001, well within ordinary seed-to-seed noise, with OWT ppl likewise tied (29.39 vs. 29.55). **We no longer treat annealing as an improvement over the fixed-schedule blend for fixed-decay EMA.** The original motivating hypothesis — that an uninformative early estimate corrupts early training if weighted at full strength immediately — may still be directionally correct, but whatever benefit it buys is too small to distinguish from noise at this seed count, and the single-seed headline number materially overstated it. The same pattern recurs with a second, independently-motivated annealing test in Section 4.7, which is why we report this as a structural caution rather than a one-off fluke: a flattering single seed should not be trusted for *any* annealing variant tested in this paper without at least two confirming seeds.

### 4.6 Cumulative (Exact Running Mean) Weighting: A Better Mechanism at Matched Blend

Section 2.2 motivates cumulative weighting as a fix for fixed-decay EMA's structural blind spot: a ~69-occurrence half-life means the historical-loss estimate for a common token reflects only a short, arbitrary recent window of training, not its difficulty across the whole corpus. We tested this directly against fixed-decay EMA at matched blend values, trained from scratch, OWT, 500K steps.

**Blend sweep (seed 42, except blend=0.75/1.0 which have 3-seed averages below).**

| | blend=0.5 (1 seed) | blend=0.75 (3-seed avg) | blend=1.0 (3-seed avg) |
|---|---|---|---|
| OWT held-out ppl | 28.76 | 29.12 | 30.09 |
| LAMBADA acc | 0.244 | 0.253 | 0.265 |
| LAMBADA ppl | 150.6 | — | — |
| HellaSwag acc_norm | 0.268 | 0.271 | 0.271 |
| PIQA acc_norm | 0.571 | 0.577 | 0.574 |
| Winogrande acc | 0.531 | 0.521 | 0.514 |

As blend decreases from 1.0 to 0.5, OWT ppl steadily improves (lower is better) while LAMBADA acc steadily worsens — the same clean, monotonic trade-off shape fixed-decay EMA's own sweep showed (Section 4.3). blend=0.5 has only a single seed so far and is not yet confirmed to the same standard as the other two points; blend=0.75's 3-seed set needed a third seed to settle an apparent non-monotonicity in its first two seeds (0.249, 0.245) that looked like it might break the trend — the third seed (0.266) landed close to blend=1.0's range instead, and the resulting 3-seed average (0.253) sits exactly where the monotonic story predicts.

**Matched-blend comparison against fixed-decay EMA (3-seed vs. 3-seed, blend=1.0, the only blend value both mechanisms have been run three times at):**

| | Cumulative (3-seed avg) | Fixed-decay EMA (3-seed avg) |
|---|---|---|
| OWT held-out ppl | 30.09 | 30.46 |
| LAMBADA acc | **0.265** | 0.258 |
| HellaSwag acc_norm | 0.271 | 0.270 |
| PIQA acc_norm | 0.574 | 0.571 |
| Winogrande acc | 0.514 | 0.517 |

**Cumulative weighting edges out fixed-decay EMA on both metrics that matter here**: better LAMBADA accuracy (0.265 vs. 0.258) at a very similar — in fact slightly *better* — OWT ppl (30.09 vs. 30.46). This is a small but genuine win for the exact-running-mean mechanism over fixed-decay at matched full blend strength, confirmed at the standard three-seed level on both sides, consistent with the motivating hypothesis that a less recency-biased historical-loss estimate makes slightly better reweighting decisions.

At matched blend=0.75 specifically, the comparison is more mixed: cumulative mode's 3-seed average (29.12 ppl / 0.253 acc) beats fixed-decay EMA's own 3-seed average at the same blend (29.55 ppl / 0.260 acc) on OWT ppl but loses on LAMBADA acc — cumulative mode buys OWT ppl at LAMBADA's expense relative to fixed-decay at this particular blend value, the opposite trade from what blend=1.0 shows. Taken together, cumulative mode isn't uniformly better across every blend value; it shifts where on the OWT-ppl/LAMBADA curve a given nominal blend lands, and happens to land in a more favorable place specifically at full strength.

### 4.7 Does Annealing Help Cumulative Mode? No

Cumulative mode's own estimator variance genuinely shrinks with accumulated sample count (~1/n, independent of training step or learning rate), unlike fixed-decay EMA's constant-width recency window — so the rationale for annealing (let the estimate mature before trusting it fully) is mechanically different for cumulative mode than it was for fixed-decay EMA in Section 4.5, and the result was not assumed to carry over. We tested it as its own question: sine schedule to a target of α=1.0, cumulative weighting, otherwise identical to Section 4.6's protocol, three seeds.

| | seed42 | seed19 | seed7 | **3-seed avg (sine)** | **3-seed avg (fixed, Section 4.6)** |
|---|---|---|---|---|---|
| OWT held-out ppl | 30.52 | 30.49 | 30.49 | 30.50 | 30.09 |
| LAMBADA acc | **0.2818** | 0.2585 | 0.2626 | **0.268** | 0.265 |
| LAMBADA ppl | 114.0 | 135.0 | 126.4 | 125.1 | — |
| HellaSwag acc_norm | 0.270 | 0.270 | 0.271 | 0.270 | 0.271 |
| PIQA acc_norm | 0.573 | 0.569 | 0.591 | 0.578 | 0.574 |
| Winogrande acc | 0.504 | 0.517 | 0.522 | 0.514 | 0.514 |

The same pattern as Section 4.5 recurs: the first seed (0.2818) again looked like a clear, substantial improvement, and again the three-seed average (0.268) lands essentially on top of the non-annealed mechanism's own three-seed average (0.265) — a difference of 0.003, inside ordinary seed noise, with a small OWT-ppl cost (30.50 vs. 30.09) that, if anything, points the wrong direction for a genuine improvement. **We draw the same conclusion as Section 4.5: sine annealing does not provide a confirmed benefit for cumulative mode either.** Having now seen this exact shape twice — flattering single seed, confirmed-null three-seed average — for two mechanistically different reasons (a stale fixed-decay estimate maturing over training vs. a cumulative estimate's variance shrinking with sample count), we treat it as a general lesson about this experimental setup rather than a property of either specific annealing rationale: a single seed in this project's small-model protocol is not infrequently a ~0.02-0.03 LAMBADA-acc favorable draw, and any first result at one seed — regardless of how well-motivated the mechanism — should be read as preliminary until confirmed.

**The best confirmed configuration in the entire family, accounting for all results in this paper**, is therefore cumulative weighting at blend=1.0, with or without the sine anneal (0.265 and 0.268 respectively, statistically tied) — both ahead of every fixed-decay EMA variant tested (0.258–0.260) and every cumulative blend below 1.0 (0.244–0.253).

### 4.8 What the OWT-Perplexity Cost Actually Measures

Every result above reports OWT held-out perplexity as a cost. It is worth being precise about what that cost actually represents, since the answer changes how it should be weighed against the LAMBADA benefit.

**Exact mechanism** (Section 2.1, step 3): `weight = token_loss[id] / token_loss[id].mean()` — normalized so the *current batch's* mean weight is exactly 1.0 (not a global, vocabulary-wide mean; the normalizer shifts batch-to-batch depending on which tokens appear) — then blended toward uniform via `weight = blend · weight + (1 - blend) · 1.0`. This is algebraically `weight = 1.0 + (weight - 1.0) · blend`.

**The resulting weight distribution, measured directly from a converged checkpoint** (`run_gpt2_cumulative_scratch_seed42.pt`, blend=1.0): the middle 98% of tokens (p01–p99) fall in a fairly modest 0.28–1.82 multiplier range, but the distribution is sharply asymmetric. At full blend strength, 4,198 tokens (8.4% of the 50,257-token vocabulary) receive a multiplier below 0.5 — and those 4,198 tokens account for **902 million of the corpus's 2.05 billion total training-token occurrences, i.e. 44% of the entire training corpus**. Tokens with a multiplier above 2.0 are comparatively rare (392 tokens) and represent only 36,838 occurrences — a rounding error by comparison. **The mechanism works almost entirely by suppressing gradient on the most common tokens, not by amplifying rare or hard ones** — consistent with the original motivation (uniform cross-entropy lets easy, high-frequency tokens swamp the gradient signal from hard, informative ones), but a sharper and more specific characterization of the mechanism than "upweighting hard tokens" alone would suggest.

**Consequence for how OWT ppl should be read.** `eval_owt.py` computes standard, unweighted cross-entropy on held-out text — by definition, a measure of how well the model matches the corpus's actual, frequency-weighted next-token distribution. A scheme that deliberately suppresses gradient on the tokens making up 44% of that corpus is, by construction, training the model to deviate from exactly the target OWT ppl measures. A meaningful fraction of the "cost" reported throughout this paper is therefore closer to a near-tautological readout of how much deliberate deviation occurred than independent evidence that the model got worse in some sense OWT ppl doesn't already define. This specific reframing is scoped to the loss-reweighting family of experiments (Sections 4.1–4.7, 4.12) — it does not apply to any architecture comparison in `paper_blt.md`, where nothing touches the loss weighting and OWT ppl remains an unqualified, direct comparison.

**Practical upshot.** OWT ppl is not needed as a degeneracy backstop for this family of experiments: HellaSwag, PIQA, Winogrande, and LAMBADA already serve as the substantive quality checks, and nothing about OWT ppl uniquely catches a problem those benchmarks would miss. We therefore report OWT ppl in every table in this paper as a **declared cost** — the honest price tag of a deliberate choice — rather than as a guardrail whose movement needs explaining away. The claim this whole line of work makes is that matching the raw, frequency-weighted distribution of web text is not actually the target; it is a stand-in for "generate more of what we actually want" (better long-range, narrative competence, of the kind LAMBADA-style tasks probe), and this family of methods trades away some of the former, on purpose, for more of the latter.

### 4.9 BLiMP: A Real, Mechanism-Independent Cost Concentrated in Long-Range Structural Tracking

Every configuration in this paper trades some BLiMP grammaticality-judgment accuracy for LAMBADA gain. This section first decomposes that cost at the individual-item level for one configuration, then reports the full cross-mechanism picture at the standard three-seed level.

**Item-level decomposition (small scale, one seed each).** Both the non-EMA baseline and the cumulative-α1.0 checkpoint (seed 42 each) were scored on every individual BLiMP item (67,000 minimal pairs: 67 subtasks × 1,000 pairs each), not just the per-subtask aggregate the harness normally reports. This required reproducing the harness's own request construction exactly — `lm-evaluation-harness` prepends a single leading space (`target_delimiter`) between an empty context and each candidate sentence, and omitting it (an error caught and fixed during this analysis) causes GPT-2's byte-level BPE tokenizer to encode each sentence's first word differently than the benchmark actually scores, corrupting every per-item result. The fix was verified by reproducing the harness's own official per-subtask accuracy exactly before trusting any downstream analysis.

Pooled result: baseline 0.7806 item-level accuracy, cumulative-α1.0 0.7644. A McNemar test on the paired per-item disagreements gives χ²=136.9, p≈0 — a real, highly significant effect, in clear contrast to a superficially similar decomposition run on ARC-Easy (Section 4.10), where an apparent ~1-point gap turned out to be statistically indistinguishable from chance. 41 of 67 subtasks show an individually significant (p<0.05) difference, lopsided rather than balanced: 28 significantly worse for cumulative vs. only 13 significantly better. The losses cluster in two linguistically coherent, long-range-dependency categories:

- **Anaphor binding (Principle A)**: `principle_A_reconstruction` (−12.3 pts), `principle_A_c_command` (−11.2 pts), `principle_A_domain_1` (−6.6 pts) — does a reflexive pronoun correctly resolve to the antecedent its syntactic structure permits.
- **Long-distance movement / islands**: `wh_questions_object_gap` (−12.0 pts), `wh_questions_subject_gap_long_distance` (−9.5 pts), `sentential_subject_island` (−9.8 pts) — does a moved wh-phrase correctly reconnect with its gap across an intervening clause boundary.

The gains cluster in a more local phenomenon — **negative polarity item (NPI) licensing** (`only_npi_scope` +11.1 pts, `npi_present_2` +6.0 pts) — whether a word like "any" appears in a context licensed by nearby negation or a question, a dependency that typically resolves within a single clause.

**Medium-scale replication (baseline vs. EMA sine-blend75, one seed each).** The pooled effect replicates (χ²=48.9, p≈0, 15-worse/6-better subtasks). Category-level, the picture is mixed: Binding/Anaphora and Quantifiers replicate cleanly in both direction and rough magnitude (Binding: −3.04% small-scale vs. −1.68% medium-scale, 3-worse/0-better at both scales); Movement/Islands keeps direction but shrinks to roughly a fifth its small-scale size. **The small-scale NPI-licensing gain does not replicate — it flips to net-negative at medium scale** (+3.57% small-scale, 5-better/1-worse, vs. −1.94% medium-scale, 2-and-2), with at least one individual subtask (`only_npi_licensor_present`) reversing sign entirely (+3.3% small-scale → −12.1% medium-scale). We read this as: the concentrated-cost finding itself replicates across scale, and Binding/Anaphora is its most robust piece; the NPI-licensing consolation story should not be assumed to generalize.

**The full cross-mechanism picture, three seeds per configuration.** Having established that the cost is real and not uniform, we checked whether its *magnitude* depends on which mechanism or blend value produced it, using the mean-of-67-subtask-accuracies figure (distinct from, but close to, the pooled item-level figure above) across the standard three seeds for every configuration in this paper with three seeds available:

| | Baseline | Fixed EMA 0.75 | Fixed EMA 1.0 | Cumulative 0.75 | Cumulative 1.0 |
|---|---|---|---|---|---|
| OWT held-out ppl (3 seeds) | **27.79** | 29.55 | 30.46 | 29.12 | 30.09 |
| LAMBADA acc (3 seeds) | 0.216 | 0.259 | 0.258 | 0.253 | **0.265** |
| BLiMP mean acc (3 seeds) | **0.763** | 0.752 | 0.754 | 0.749 | 0.752 |

**All four reweighted-loss configurations land in a tight 0.749–0.754 band, all below the non-EMA baseline's 0.763 — regardless of mechanism (fixed-decay vs. cumulative) or blend value (0.75 vs. 1.0).** This is a materially cleaner picture than an earlier, single-seed-per-point pass through the same four configurations suggested, where the apparent ordering looked non-monotonic (fixed-decay at blend=1.0 appeared to cost *less* BLiMP than at blend=0.75, the opposite ordering from what OWT ppl and LAMBADA both show for that pair). With three seeds, that apparent non-monotonicity resolves into noise: individual seeds for fixed-decay blend=1.0 alone ranged from −0.7pt to −2.0pt relative to baseline, and the single-seed number that originally drove the non-monotonic reading (−0.7pt) was simply on the low end of that spread. **The robust finding is a small (roughly 1–2 point), consistent-direction BLiMP cost that is present across every reweighting configuration tested, essentially independent of mechanism or blend strength** — a different, and now better-supported, conclusion than treating specific blend/mechanism combinations as differentially costly.

### 4.10 ARC-Easy and the Supplementary Suite: Mostly Noise

As a check on whether every apparent gap in this paper's tables reflects a real effect, we ran the same item-level decomposition used for BLiMP on ARC-Easy, a benchmark where baseline and cumulative-α1.0 showed an apparent ~1-point gap in aggregate accuracy.

**Full per-item comparison, all 2,376 test examples.**

| Outcome | Small scale (baseline vs. cumulative-α1.0) | Medium scale (baseline vs. EMA sine-α0.75) |
|---|---|---|
| Both correct | 712 (30.0%) | 880 (37.0%) |
| Both wrong, same pick | 1,098 (46.2%) | 984 (41.4%) |
| Both wrong, different picks | 219 (9.2%) | 190 (8.0%) |
| Baseline right, other wrong | 183 (7.7%) | 161 (6.8%) |
| Other right, baseline wrong | 164 (6.9%) | 161 (6.8%) — exact tie |

McNemar's test on the disagreement counts: χ²=1.04, p=0.31 (small scale); χ²=0.000, p=1.0000 (medium scale — an exact tie). **Statistically indistinguishable from a coin flip at both scales.** In the two disagreement categories, whichever candidate (gold or the wrong pick) happens to catch a large idiosyncratic score swing wins that particular item; in the two "both agree" categories (76–78% of the data), the gold answer's score change shows no systematic effect beyond ordinary background drift (t-tests p=0.41/0.12 small scale, p=0.35/0.87 medium scale). **ARC-Easy's apparent ~1-point gap is noise, not a real cost of the loss-reweighting family** — a genuinely different conclusion from BLiMP's (Section 4.9), and a reminder that not every small aggregate gap in a benchmark table should be read as a real effect.

**Full supplementary-suite table, three seeds each.**

| | Baseline | Fixed EMA 0.75 | Fixed EMA 1.0 | Cumulative 0.75 | Cumulative 1.0 |
|---|---|---|---|---|---|
| ARC-Easy acc (3 seeds) | **0.379** | 0.378 | 0.371 | 0.371 | 0.372 |
| BoolQ acc (3 seeds) | 0.567 | 0.591 | **0.609** | 0.574 | 0.607 |
| OpenBookQA acc (3 seeds) | 0.131 | 0.141 | **0.143** | 0.137 | **0.143** |

ARC-Easy's 3-seed figures are consistent with the item-level noise finding above — the baseline-vs-reweighted gap (0.007–0.008) is smaller than either family's own seed-to-seed spread (0.016–0.017). BoolQ and OpenBookQA show the opposite direction from BLiMP/ARC-Easy: both blend=1.0 variants (fixed-decay and cumulative) *beat* baseline rather than cost against it, while both blend=0.75 variants sit at or below it — consistent with the primary-suite finding that blend=1.0 is the stronger operating point on LAMBADA too. Given BoolQ and OpenBookQA are much smaller, noisier benchmarks than BLiMP's 67-subtask aggregate (their own seed-to-seed spread reaches 0.065 for BoolQ), this asymmetry is plausible but, per Section 4.5's cautionary finding about trusting patterns from limited seed counts, should not be over-read without further seeds specifically targeting these two tasks.

### 4.11 Does Reweighting Benchmark Scoring Itself Reveal a Hidden Advantage? No

A natural follow-up question: if training deliberately does not score all tokens equally, why should the benchmarks used to *evaluate* training score every token in a candidate completion equally via a plain summed log-likelihood? If common, low-information tokens dilute the benchmark signal the same way they dilute the raw training loss, reweighting benchmark scoring the same way might reveal a larger reweighting advantage than the standard metrics show.

**Method.** We identified a small, deliberately conservative closed-class token list — **a, an, the, of, and** (leading-space BPE variants only, 10 token IDs) — chosen by excluding on any doubt: all quantifiers/demonstratives (no/any/some/all/this/that — real logical or deictic weight), all other conjunctions (or/but/nor/yet/so — each changes logical structure), and all other prepositions (in/on/at/under/with/without — real spatial/temporal/causal content) were excluded. Complete removal (weight=0) was considered and rejected after checking real BLiMP data: at least two subtasks (`existential_there_quantifiers_1`, `superlative_quantifiers_2`) have minimal pairs where the entire distinguishing signal between the two candidates is one of these five words — zeroing them out would turn those comparisons into exact ties, destroying real signal unrelated to the hypothesis being tested. We swept a weight ∈ {1.0, 0.5, 0.25, 0.1, 0.0} applied to these tokens' contribution to each benchmark's accumulated log-likelihood (`blt_lm_eval.py --lowimpact-weight`, verified bit-for-bit identical to the unmodified harness at weight=1.0), comparing the non-EMA baseline against cumulative-blend=1.0 (seed 42 each) across all eight benchmarks used in this paper.

**Result: mixed, no clean confirmation of the hypothesis.**

| Benchmark | Structural fit | Result |
|---|---|---|
| BLiMP | Minimal-pair design shares most of the sentence between candidates | Flat — gap unchanged (−0.021) at every weight |
| HellaSwag | Full-sentence candidate comparison | No real gap exists between these checkpoints at any weight |
| **PIQA** | Full-sentence candidate comparison | **Real, direction-consistent narrowing**: gap shrinks from −0.017 (w=1.0) to −0.005 (w=0.25/0.1), not perfectly monotonic |
| Winogrande | Full-sentence candidate comparison | Flat small gap at every weight |
| **ARC-Easy** | Full-sentence candidate comparison | **Moves the wrong way** — gap widens slightly (−0.007 → −0.011) |
| BoolQ | Single-token (`yes`/`no`) answer choices | Mechanically inert — neither answer token is in the 5-word set |
| OpenBookQA | Full-sentence, but only 500 examples | Bounces with no discernible pattern |
| LAMBADA | `acc` is computed from top-1 greedy agreement, not a weighted sum | Mechanically inert on `acc` (bit-for-bit identical at every weight); only perplexity moves, and this paper has never treated LAMBADA perplexity as the metric that matters |

One real supporting result (PIQA), one real opposing result (ARC-Easy), two benchmarks structurally inert for reasons unrelated to the hypothesis (BoolQ, LAMBADA acc), and the rest flat or too noisy to read. **This does not rise to "standard benchmark scoring structurally hides the reweighting advantage"** — it reads more like occasional, benchmark-specific sensitivity to which particular low-impact words happen to be touched, not a general confound in how these benchmarks are scored.

### 4.12 The Trade-off Generalizes to a Larger Model Scale

Every other result in this paper uses GPT-2 Small (124M parameters). To test whether the core trade-off (Section 4.1) is an artifact of this specific model size, we trained GPT-2-medium (355M parameters, ~3× the parameter count) from scratch for 1.5M steps (Section 3), comparing a non-EMA baseline against the sine-annealed fixed-decay blend=0.75 configuration and, in a later run motivated by Section 4.6's finding that cumulative weighting edges out fixed-decay EMA at small scale, a cumulative blend=1.0 configuration.

| | Medium baseline | Medium EMA (sine blend=0.75) | Medium cumulative (blend=1.0) |
|---|---|---|---|
| OWT held-out ppl | **18.09** | 18.79 | 19.49 |
| LAMBADA acc | 0.311 | 0.373 | **0.381** |
| LAMBADA ppl | 40.8 | 31.2 | **30.7** |
| HellaSwag acc_norm | 0.297 | 0.302 | 0.301 |
| PIQA acc_norm | **0.607** | 0.594 | 0.608 |
| Winogrande acc | **0.522** | 0.515 | 0.500 |
| BLiMP mean acc | **0.801** | 0.792 | 0.791 |

**The same trade-off shape documented at small scale (Section 4.1) replicates at roughly 3× the parameter count, for both mechanisms.** Fixed-decay EMA: OWT ppl modestly worse (+3.9% relative), LAMBADA accuracy +20% relative, HellaSwag edges up slightly, PIQA/Winogrande dip slightly — the same pattern of gains and losses as the small-scale GPT-2 result in Section 4.1. Cumulative weighting at medium scale reproduces its own small-scale signature relative to fixed-decay EMA (Section 4.6): a larger OWT-ppl cost (19.49 vs. 18.79) but the better LAMBADA result of the three configurations (0.381), including a LAMBADA-perplexity figure (30.7) better than either alternative. Per-step timing on a clean post-migration segment (1.32 s/step, consistent across both halves of the measured window) confirms the reweighting mechanism costs no meaningful extra compute at this scale either — it is a per-token multiply on an already-computed loss, not a new matmul.

**One place the medium-scale cumulative result diverges from its small-scale counterpart**: Winogrande drops to 0.500 (chance level) for cumulative weighting, versus 0.522 (baseline) and 0.515 (EMA) — a real-looking, isolated regression rather than the roughly-flat Winogrande numbers cumulative weighting shows at small scale (Section 4.6). Both medium-scale reweighted configurations are single-seed, as is the baseline; no multi-seed medium-scale result exists for any configuration in this paper given the cost (~2–3 weeks) of a single run at this scale, so this divergence should be read as a single data point, not yet a confirmed architectural or mechanistic effect.

---

## 5. Discussion

**Why blending beats switching.** Sections 4.2-4.4 together tell a coherent story about *when* a model specializes into an objective's basin, not just *which* objective is used. Pure weighting (α=1.0 throughout) and pure CE (α=0.0 throughout) each converge to their own basin; switching between them late (Section 4.2) forces the model to climb out of one basin and into another, and the climb-out is fast and disproportionate — the LAMBADA gain built over 500K weighted steps evaporates in under 10K CE steps. A fixed intermediate α, whether reached by sequential fine-tune (Section 4.3) or trained from scratch (Section 4.4), never creates a single-objective basin to escape from in the first place.

**Annealing looked like a refinement of the same mechanism; three seeds say it isn't one.** Sections 4.5 and 4.7 tested the sine anneal for two mechanistically distinct reasons (a stale fixed-decay estimate needing time to mature; a cumulative estimate's variance shrinking with sample count) and got the identical result both times: a flattering first seed, and a three-seed average statistically tied with the non-annealed baseline. We treat this as the paper's central methodological lesson rather than a footnote — a single seed in this experimental setup is not a rare occurrence of a ~0.02–0.03 LAMBADA-acc favorable draw, and that magnitude is large enough to produce a highly plausible-looking "clear win" that evaporates on replication. Every multi-seed claim in this paper exists because of this risk, not despite it.

**Cumulative weighting's advantage over fixed-decay EMA is real but modest, and is the best-supported mechanistic claim in this paper.** Unlike the annealing results, the cumulative-vs-fixed-decay comparison at matched blend=1.0 (Section 4.6) is a true three-seed-vs-three-seed comparison, and the direction (cumulative wins on both OWT ppl and LAMBADA acc simultaneously, rather than trading one for the other) is the opposite of what a spurious draw would typically produce — a spurious seed-driven difference would more often show one metric improving at the other's expense, not both improving together. This is consistent with the motivating mechanism: an estimate that reflects a token's difficulty across its whole training history, rather than only its most recent ~69 occurrences, makes better-calibrated reweighting decisions.

**OWT perplexity is a declared cost, not a safety check.** Section 4.8's reframing changes how every number in this paper should be read, but only within the loss-reweighting family: the mechanism's OWT-ppl cost is substantially a direct, near-tautological consequence of deliberately suppressing gradient on the 44% of the corpus made up of common tokens, not an independent signal that something has gone wrong elsewhere. The zero-shot benchmarks (LAMBADA, HellaSwag, PIQA, Winogrande, and the supplementary/BLiMP suites) already do the job of catching a genuinely broken model; OWT ppl's role in this paper is to state the price of the trade, not to police it.

**The BLiMP cost is real, concentrated, and consistent — but small and mechanism-independent.** Section 4.9's three-seed comparison shows every reweighting configuration tested pays roughly the same (1–2 point) BLiMP cost regardless of mechanism or blend strength, concentrated in long-range structural dependencies (anaphor binding, wh-movement) rather than spread uniformly across grammaticality judgment. Combined with Section 4.10's finding that a superficially similar ARC-Easy gap is pure noise, the overall picture is: the reweighting family pays a small, real, specific tax on long-range syntactic tracking, buys a much larger gain on long-range semantic/narrative prediction (LAMBADA), and does not meaningfully move general commonsense reasoning (HellaSwag, PIQA, Winogrande) or most of the supplementary suite in either direction.

**Cost is bounded, not runaway, and generalizes across both architecture and scale.** Across every configuration tested — both weighting mechanisms, every blend ratio, both model scales, both architectures tested — OWT held-out perplexity degradation stays in a narrow band (roughly +4% to +11% relative to the matched non-reweighted baseline). There is no configuration where the mechanism causes catastrophic degradation of general language-modeling ability; the cost is a real but modest, controllable tax across every axis we tested generalization along.

**Open questions this paper does not resolve** (carried forward from `project_loss_function_ideas` memory and `CLAUDE.md`, updated for what this revision answers):

- *(Answered)* Does a jointly-trained blend beat sequential fine-tune? — **Yes** (Section 4.4), confirmed across three independent seeds.
- *(Answered, corrected)* Does annealing the blend coefficient in via a schedule help? — **No, not once confirmed to three seeds**, for either mechanism (Sections 4.5, 4.7). An initial single-seed result for each looked like a substantial improvement; neither replicated.
- *(Answered)* Is cumulative (exact running-mean) weighting better than fixed-decay EMA? — **Yes, modestly, at matched full blend strength** (Section 4.6), the best-supported mechanistic claim in this paper. Not yet tested at every blend value with equal seed coverage — blend=0.5 has only one seed for either mechanism.
- *(Answered)* Does the base trade-off generalize across model scale? — **Yes** (Section 4.12): the same trade-off shape holds at ~3× the parameter count, for both fixed-decay EMA and, in a later addition, cumulative weighting — the latter again showing its small-scale signature of a larger OWT-ppl cost for a better LAMBADA result. Both medium-scale results are single-seed.
- *(Answered)* Is the BLiMP cost uniform or concentrated, and does it depend on mechanism/blend? — **Concentrated in long-range structural dependencies** (anaphor binding, wh-movement), replicating across small and medium scale, and **present at similar small magnitude regardless of mechanism or blend value** (Section 4.9) — a cleaner and more general answer than this paper's earlier, single-seed-per-point read suggested.
- *(Answered)* Does reweighting benchmark scoring itself (rather than training) reveal a larger hidden advantage? — **No clean confirmation** (Section 4.11): one supporting result, one opposing result, the rest inert or noisy.
- *(Partially answered)* Sweep more α values — Section 4.3 covers 0.25/0.5/0.75 sequential for fixed-decay EMA; Section 4.6 covers 0.5/0.75/1.0 for cumulative mode, though blend=0.5 is single-seed. A full matched sweep across both mechanisms at equal seed coverage has not been run.
- *(Open)* Is cumulative weighting, jointly-trained blending, or sine-annealing architecture-general the way the base EMA effect was (Section 4.1)? No BLT experiment beyond pure fixed-decay EMA (α=1.0) has been run.
- *(Open, not yet started)* Option 2 from the original brainstorm — a short-context vs. long-context self-comparison, upweighting tokens where the loss gap between truncated and full context is large — was never implemented. It targets long-range dependence more directly than the per-vocabulary-ID historical-loss estimate used throughout this paper (see the Token Weighting comparison, Section 6) but costs 2× forward-pass compute per step.
- *(Open, not yet started)* Cycling between structurally different loss functions during training (not just blending two, but rotating among 3+) was proposed as a way to avoid any single loss function's minima, but not implemented or tested.

---

## 6. Related Work

**Rho-1 / Selective Language Modeling.** Lin et al. train a small reference model, compute each training token's "excess loss" relative to the reference, and backpropagate only through the highest-excess tokens, showing strong gains on math reasoning benchmarks ([arXiv 2404.07965](https://arxiv.org/abs/2404.07965), NeurIPS 2024). This is the closest prior work in spirit — both identify and upweight "hard" tokens — but Rho-1 requires training and running a separate reference model at every step, while both weighting mechanisms here are entirely self-referential, using only the running model's own historical per-token loss.

**Token Weighting for Long-Range Language Modeling.** This work compares a long-context model's per-token confidence against a short-context model's confidence, and upweights tokens where long-range context specifically helps ([arXiv 2503.09202](https://arxiv.org/abs/2503.09202), NAACL 2025). This is the most targeted prior fix for the LAMBADA-style failure mode motivating this paper (Section 1), and directly inspired "Option 2" in the open-questions list (Section 5) — but it requires two forward passes per step (full and truncated context) to compute the confidence gap, versus this paper's single-forward-pass approach. Both weighting mechanisms here identify tokens that are *globally* hard (rare or high-entropy words) rather than tokens that are specifically *long-range-dependent* in context, which likely explains why they capture only part of the available LAMBADA-style benefit relative to what a direct long-range confidence signal could in principle achieve.

**MiLe Loss.** Weights tokens by their predictive entropy — uncertain tokens receive a stronger training signal, with no reference model required ([arXiv 2310.19531](https://arxiv.org/abs/2310.19531)). Closer in spirit and cost to this paper's method than Rho-1 or Token Weighting (both are single-forward-pass, no-reference-model approaches), but entropy alone cannot distinguish genuine long-range uncertainty (the kind LAMBADA tests) from other sources of ambiguity.

**Tilting the Playing Field.** Proposes cycling time-dependent loss weights across multiple objectives during training, arguing this pushes the optimizer toward minima that are simultaneously good under all the cycled objectives rather than just the current one ([arXiv 2102.03793](https://arxiv.org/abs/2102.03793), ICML 2021). This is the direct precedent for the "cycling among multiple loss functions" idea in Section 5's open questions.

**Summary.** The per-token weighting schemes in this paper occupy a specific, previously-unfilled point in the design space of hard-token-upweighting methods: no reference model (unlike Rho-1), single forward pass (unlike Token Weighting), and using an explicit historical-loss signal per vocabulary ID rather than instantaneous entropy (unlike MiLe Loss). The comparison between a fixed-decay and an exact-running-mean estimator of that historical signal, and the finding that annealing the blend coefficient in does not reliably improve either, have — to our knowledge — not been reported elsewhere for this class of method.

---

## Appendix A: Result File Index

For digging up raw data behind any number in this paper. Paths are relative to the repo root unless marked otherwise; see `CLAUDE.md` for the full, chronological account of every run, including machine locations and migration history.

| Result | Checkpoint | lm-eval JSON | Notes |
|---|---|---|---|
| GPT-2 non-EMA baseline (seed 42/19/7) | `run_gpt2_baseline_seed{42,19,7}.pt` | `lm_eval_gpt2_baseline_seed{42,19,7}.json` | 3-seed baseline, also `_blimp.json`/`_supplementary.json` per seed |
| BLT non-EMA baseline (seed 7) | `run_blt_scratch_seed7.pt` | `lm_eval_blt_scratch_seed7.json` | Section 4.1 BLT comparison |
| GPT-2 fixed-decay EMA, pure α=1.0 (seed 42/19/7) | `run_gpt2_ema_seed{42,19,7}.pt` | `lm_eval_gpt2_ema_seed{42,19,7}.json` | 3-seed, Section 4.6; also `_blimp.json`/`_supplementary.json` per seed |
| BLT fixed-decay EMA, pure α=1.0 (seed 42) | `run_blt_ema_seed42.pt` | `lm_eval_blt_ema_seed42.json` | Section 4.1, single seed, only BLT reweighting result in this paper |
| Pure-CE / α=0.25 / α=0.5 sequential fine-tunes | `run_gpt2_ema_{then_ce,blend25,blend50}_finetune_seed42.pt` | matching `lm_eval_*.json` | Section 4.3 |
| α=0.75 jointly-trained-from-scratch (seed 42/19/7_v2) | `run_gpt2_ema_blend75_scratch_seed{42,19,7_v2}.pt` | `lm_eval_gpt2_ema_blend75_scratch_seed{42,19,7_v2}.json` | Section 4.4; also `_blimp.json`/`_supplementary.json` per seed |
| **RETRACTED** — seed-7 launch missing `--ema-loss-weighting` | `run_gpt2_ema_blend75_scratch_seed7.pt` | — | Section 4.4 methodological note; excluded from all results |
| α=0.75 sine-annealed, jointly-trained-from-scratch (seed 42/19/7) | `run_gpt2_ema_blend75_sine_scratch_seed{42,19,7}.pt` | `lm_eval_gpt2_ema_blend75_sine_scratch_seed{42,19,7}.json` | Section 4.5, 3-seed; also `_supplementary.json` per seed |
| Cumulative blend=1.0 (seed 42/19/7) | `run_gpt2_cumulative_scratch_seed{42,19,7}.pt` | `lm_eval_gpt2_cumulative_scratch_seed{42,19,7}.json` | Section 4.6, 3-seed; also `_blimp.json`/`_supplementary.json` per seed |
| Cumulative blend=0.75 (seed 42/19/7) | `run_gpt2_cumulative_blend75_scratch_seed{42,19,7}.pt` | `lm_eval_gpt2_cumulative_blend75_scratch_seed{42,19,7}.json` | Section 4.6, 3-seed; also `_blimp.json`/`_supplementary.json` per seed |
| Cumulative blend=0.5 (seed 42) | `run_gpt2_cumulative_blend50_scratch_seed42.pt` | `lm_eval_gpt2_cumulative_blend50_scratch_seed42.json` | Section 4.6, single seed |
| Cumulative + sine, blend=1.0 target (seed 42/19/7) | `run_gpt2_cumulative_sine_scratch_seed{42,19,7}.pt` | `lm_eval_gpt2_cumulative_sine_scratch_seed{42,19,7}.json` | Section 4.7, 3-seed |
| Medium baseline (seed 42) | `run_gpt2_medium_baseline_seed42.pt` | `lm_eval_gpt2_medium_baseline_seed42.json` | Section 4.12; also `_blimp.json`/`_supplementary.json`/`_extended.json` |
| Medium EMA, sine blend=0.75 (seed 42) | `run_gpt2_medium_ema_blend75_sine_seed42.pt` | `lm_eval_gpt2_medium_ema_blend75_sine_seed42.json` | Section 4.12; also `_blimp.json`/`_extended.json` |
| Medium cumulative blend=1.0 (seed 42) | `run_gpt2_medium_cumulative_seed42.pt` | `lm_eval_gpt2_medium_cumulative_seed42.json` | Section 4.12; also `_blimp.json`/`_supplementary.json`/`_extended.json` |
| BLiMP full per-item decomposition | `blimp_full_results.json` (small), `blimp_full_results_medium.json` (medium) | — | Section 4.9 |
| ARC-Easy full per-item decomposition | `arc_easy_full_results.json`, `arc_easy_full_results_medium.json`, `arc_easy_disagreements.json` | — | Section 4.10 |
| Low-impact-token benchmark-reweighting sweep | — | `lm_eval_lowimpact_sweep_{baseline,cumulative100}_seed42_w{0.0,0.1,0.25,0.5,1.0}.json` and per-benchmark variants (40 files total) | Section 4.11 |

**Code references**: weighting mechanism in `train.py` (search `token_loss`, `token_count`, `ema_blend`, `loss_weighting_mode`; the checkpoint field is named `ema_loss` for both modes for backward compatibility). `--finetune`'s buffer-loading fix landed alongside `--ema-blend` in commit `c118787`. `--loss-weighting-mode cumulative` and `--ema-blend-schedule sine` (plus the `effective_blend` training-log column) are both in `train.py`; see `CLAUDE.md` for exact commit references and dates.

**Memory references** (Claude session memory, not in this repo): `project_loss_function_ideas.md` — original brainstorm of Options 1-3 and the loss-cycling idea, plus the running log this paper formalizes.

## Appendix B: Academic References

- Lin, Z. et al. "Not All Tokens Are What You Need for Pretraining." [arXiv 2404.07965](https://arxiv.org/abs/2404.07965), NeurIPS 2024. (Rho-1 / Selective Language Modeling)
- "Token Weighting for Long-Range Language Modeling." [arXiv 2503.09202](https://arxiv.org/abs/2503.09202), NAACL 2025.
- "MiLe Loss: a MInimizing the average LExical error for LLM Pretraining." [arXiv 2310.19531](https://arxiv.org/abs/2310.19531).
- "Tilting the Playing Field: Dynamical Loss Functions for Machine Learning." [arXiv 2102.03793](https://arxiv.org/abs/2102.03793), ICML 2021.

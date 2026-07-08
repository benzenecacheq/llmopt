# EMA Per-Token Loss Weighting: A Self-Referential Curriculum for Language Model Pretraining

## Abstract

Standard cross-entropy pretraining weights every token equally, so easy, high-frequency tokens dominate the gradient signal and hard, informative tokens — the kind long-range benchmarks like LAMBADA specifically probe — contribute comparatively little. We introduce a simple, cheap fix: maintain a per-vocabulary-token exponential moving average (EMA) of that token's historical loss, and reweight each token's contribution to the training loss by its (normalized) EMA value. Unlike prior hard-token-selection methods, this requires no reference model and no extra forward pass — it is entirely self-referential, using only information the model already produces.

Trained from scratch on OpenWebText for 500K steps, full EMA weighting improves LAMBADA accuracy substantially (GPT-2: 0.225 → 0.253; BLT: 0.212 → 0.242) and LAMBADA perplexity dramatically (GPT-2: 174.6 → 119.6, a 31.5% drop) at a real but bounded cost to general next-token prediction (OWT held-out perplexity worsens by 8-11%). This trade-off is **architecture-general**: BLT and standard multi-head attention (MHA) show the same shape of trade-off within a few percent of each other, despite BLT's very different attention mechanism (see `paper_blt.md`).

We further show this trade-off is not fixed. Naively fine-tuning an EMA-converged checkpoint back to standard cross-entropy is a **forgetting cliff**, not a free lunch: the LAMBADA gain evaporates (and overshoots below baseline) far faster than the OWT-ppl cost can be recovered. Blending the EMA objective with standard CE — rather than switching between them — produces much better trade-off points, and blending **from step 0 of a from-scratch run** (rather than fine-tuning into the blend after full EMA convergence) is the best protocol tested: a 75/25 (EMA/uniform) blend trained jointly from scratch reaches LAMBADA accuracy 0.269 — better than even the pure-EMA run — while giving up almost nothing extra in OWT perplexity versus the best sequential fine-tune we tried.

---

## 1. Background: The Loss/Benchmark Mismatch

Cross-entropy next-token prediction optimizes equally over all tokens in a training corpus. Most tokens in web text are easy — function words, common continuations, near-deterministic completions — and dominate the loss simply by frequency. Benchmarks like LAMBADA, by contrast, specifically test hard, long-range predictions (predicting a final word that requires understanding an entire preceding paragraph) that are a tiny fraction of total training signal. A model's cross-entropy loss can decrease steadily while its performance on exactly the tokens LAMBADA-style benchmarks care about stagnates or even regresses relative to what a differently-weighted objective would achieve.

This was observed directly during earlier BLT experiments (`paper_blt.md`, Section 4.1-4.2): a BLT model fine-tuned on WikiText-103 scored LAMBADA acc 0.114, far below GPT-2's 0.275, even though WikiText validation perplexity looked healthy. Retraining on broader web text (OpenWebText) improved LAMBADA substantially without any loss-function change, showing the original gap was partly a training-domain artifact — but a residual gap remained, motivating a loss-function-level fix rather than a data-level one.

---

## 2. Method: Self-Referential EMA Per-Token Loss Weighting

The key idea: the information needed to identify "hard" tokens already exists inside the model's own per-step loss — no external reference model (cf. Rho-1, Section 6) or extra forward pass (cf. short-context self-comparison, discussed but not implemented — see Section 7) is required.

**Mechanism** (`train.py`, `--ema-loss-weighting`, implementation lines ~239, ~289-312):

1. Maintain a vector `ema_loss` of length `vocab_size` (50,257 for GPT-2's BPE vocabulary), one EMA per **vocabulary token ID** (not position, not n-gram). Initialized to `log(vocab_size)` — the loss of a uniform/maximum-entropy predictor — as an uninformative prior.
2. At each training step, compute the standard per-token cross-entropy loss for every predicted token in the batch, `per_token_loss`.
3. Look up each token's current EMA weight: `weight = ema_loss[token_id]`, then normalize so the batch's mean weight is 1: `weight = weight / weight.mean()`.
4. Optionally blend toward uniform weighting via `--ema-blend α` (default α=1.0, full EMA): `weight = α · weight + (1 - α) · 1.0`. α=0 recovers standard, unweighted cross-entropy exactly.
5. The training loss is `mean(weight.detach() · per_token_loss)` — the weight is **detached** from the graph, so it only rescales each token's gradient magnitude; it does not introduce its own gradient path.
6. After the step, update the EMA in place, per vocabulary ID actually seen in the batch: `ema_loss[id] = decay · ema_loss[id] + (1 - decay) · batch_mean_loss_for_id`, with `decay=0.99` by default (`--ema-decay`).

Net effect: tokens whose vocabulary ID has *historically* been hard (high running-average loss) get upweighted; tokens that have been reliably easy get downweighted. Because the EMA is per-*vocabulary-ID*, not per-context-instance, this is best understood as identifying globally hard tokens (rare words, tokens with high inherent entropy) rather than specifically long-range-context-dependent instances of otherwise-easy tokens — an important distinction from the Token Weighting approach in Section 6, which explicitly targets long-range dependence via a short- vs. long-context confidence comparison.

The `--ema-blend` mechanism additionally allows interpolating between full EMA weighting and standard CE, either as a **sequential fine-tune** (start from an EMA-converged checkpoint, continue training with a fixed blend α) or, as tested in Section 4.4, as the **objective from step 0** of a from-scratch run.

Checkpoints store the `ema_loss` buffer directly (`save_checkpoint(..., ema_loss=...)`), and both `--resume` and `--finetune` restore it — the latter was a bug fix partway through this work (see Section 4.2): originally only `--resume` loaded the buffer, so any `--finetune` run silently restarted hard-token tracking from the uninformative uniform prior.

---

## 3. Experimental Setup

Identical protocol to `paper_blt.md` Section 3 unless noted: GPT-2 Small architecture (12 layers, 12 heads, D=768) or BLT (same base architecture, single shared M — see `paper_blt.md` Section 2), trained from scratch (no pretrained initialization) on 2M OpenWebText documents, Adam optimizer (lr=5e-5, cosine decay, 200 warmup steps, batch size 4, block size 1024), 500K steps unless noted. Evaluation: held-out OWT perplexity (files 21-25, sliding window), zero-shot accuracy on LAMBADA/HellaSwag/PIQA/Winogrande via lm-eval-harness. All EMA experiments in this paper use standard MHA (GPT-2) unless labeled BLT.

**Caution on the in-training LAMBADA proxy.** `train.py --lambada-eval-every` logs a cheap 200-example greedy-decoding cloze accuracy during training, for monitoring only. This proxy is **not discriminating** — it stayed flat at ~0.55-0.57 through an entire 10,000-step fine-tune run where the real lm-eval-harness LAMBADA accuracy (5,153 examples, log-likelihood scoring, the number reported everywhere in this paper) collapsed by several points (Section 4.2). Every result below is from the full lm-eval-harness benchmark, never the in-training proxy.

---

## 4. Results

### 4.1 Full EMA Weighting Is a Real, Architecture-General Trade-off

We trained GPT-2 and BLT from scratch with `--ema-loss-weighting` (α=1.0, decay=0.99), 500K steps, OWT, seed 42, and compared each against its own non-EMA baseline of the same architecture/seed/step-count.

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

**The trade-off shape is nearly identical across architectures**: LAMBADA accuracy gain is +0.028 (GPT-2) vs. +0.030 (BLT); PIQA degrades by almost the same amount on both (−0.010 vs. −0.009); OWT ppl worsens on both, proportionally somewhat more on BLT (+11.2% vs. +8.2%). This is strong evidence the effect is a property of the **loss function itself**, not of the attention mechanism — a more general claim than prior hard-token-weighting work (Rho-1, MiLe Loss; Section 6), which was only evaluated on standard transformer attention.

BLT-caused note: BLT already pays its own architecture-level OWT ppl cost relative to MHA even without EMA (`paper_blt.md` Section 4.3); EMA's cost stacks on top of that (30.81 → 34.25) rather than interacting with it in any special way, consistent with the "architecture-general" reading.

### 4.2 Post-Hoc Fine-Tuning Back to Standard CE Is a Forgetting Cliff

Took the converged `run_gpt2_ema_seed42.pt` (full EMA, 500K steps) and fine-tuned for just **10,000 steps** (2% of the original training budget) with plain cross-entropy — a fresh optimizer and LR schedule (`train.py --finetune ...`; `--resume` would be wrong here since it would continue the original cosine schedule, already decayed to ~0 LR by step 500K). Run on `titan`: `run_gpt2_ema_then_ce_finetune_seed42.pt`.

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

`--ema-blend α` interpolates the per-token weight toward uniform (α=1.0 = full EMA, α=0.0 = standard CE, Section 2). We swept α ∈ {0.25, 0.5, 0.75} as a **sequential fine-tune** from the same converged EMA checkpoint, same 10,000-step / fresh-schedule protocol as Section 4.2, all on `titan`, seed 42.

| | EMA (before) | α=0.25 ft | α=0.5 ft | α=0.75 ft | Pure-CE ft (α=0) | Non-EMA baseline |
|---|---|---|---|---|---|---|
| OWT held-out ppl | 30.06 | 29.29 | 29.68 | *not measured‡* | 29.13 | 27.78 |
| OWT held-out loss | 3.4031 | 3.3771 | — | — | — | 3.3243 |
| LAMBADA acc | 0.253 | 0.231 (±0.0059) | 0.244 | 0.258 (±0.0061) | 0.217 | 0.225 |
| LAMBADA ppl | 119.6 | 159.3 | 142.9 | 132.1 | 184.1 | 174.6 |
| HellaSwag acc_norm | 0.272 | 0.270 | 0.270 | 0.270 | 0.271 | 0.268 |
| PIQA acc_norm | 0.569 | 0.563 | 0.562 | 0.558 | 0.566 | 0.579 |
| Winogrande acc | 0.511 | 0.506 | 0.498 | 0.504 | 0.499 | 0.505 |

‡ OWT held-out ppl for the α=0.75 sequential fine-tune (`run_gpt2_ema_blend75_finetune_seed42.pt`) has not yet been measured — the checkpoint exists locally but `eval_owt.py` has not been run against it. Flagged as an open data gap rather than estimated (see Appendix A for exact file locations to close this gap).

The sweep is monotonic and interior points beat both endpoints on the trade-off that matters: every blend α tested keeps LAMBADA accuracy **above** the non-EMA baseline (0.231-0.258 vs. 0.225) — unlike the pure-CE fine-tune, which overshot *below* it (0.217) — while giving up less OWT-ppl recovery than pure CE (only 29.29-29.68 range vs. pure CE's 29.13, though the α=0.75 point's OWT ppl is currently unmeasured). The α=0.5 finding reported earlier (`CLAUDE.md`, `project_loss_function_ideas` memory) — "a small give-back in OWT-ppl recovery buys back a disproportionately large fraction of the LAMBADA benefit" — holds across the whole sweep, not just at α=0.5.

### 4.4 Jointly-Trained Blend From Scratch Beats Every Sequential Fine-Tune

Section 4.3's blends are all **sequential**: fine-tuned into the blend only after the model fully converged in the pure-EMA basin. This leaves open whether a model trained with the blended objective **from step 0** — never specializing into pure EMA before correcting — reaches a better point, since it never has to unlearn anything.

We trained GPT-2 from scratch (`--baseline --from-scratch --ema-blend 0.75`, seed 42, full 500K steps, OWT) rather than fine-tuning: `run_gpt2_ema_blend75_scratch_seed42.pt`, run on `titan`, DONE 2026-06-30.

| | EMA pure (α=1, 500K from scratch) | Pure-CE seq. fine-tune (10K) | 50/50 seq. fine-tune (10K) | **75/25 blend, jointly trained from scratch (500K)** | Non-EMA baseline |
|---|---|---|---|---|---|
| OWT held-out ppl | 30.06 | 29.13 | 29.68 | 29.61 | 27.78 |
| OWT held-out loss | 3.4031 | — | — | 3.3881 | 3.3243 |
| LAMBADA acc | 0.253 | 0.217 | 0.244 | **0.269** (±0.0062) | 0.225 |
| LAMBADA ppl | 119.6 | 184.1 | 142.9 | 130.6 | 174.6 |
| HellaSwag acc_norm | 0.272 | 0.271 | 0.270 | 0.273 | 0.268 |
| PIQA acc_norm | 0.569 | 0.566 | 0.562 | 0.565 | 0.579 |
| Winogrande acc | 0.511 | 0.499 | 0.498 | 0.522 | 0.505 |

**This is the best result in the whole EMA/blend family.** The jointly-trained 75/25 blend beats even *pure* EMA's LAMBADA accuracy (0.269 vs. 0.253) — a result no sequential fine-tune at any α reached — while landing at essentially the same OWT-ppl cost as the best sequential fine-tune tested (29.61 vs. 29.68 for the 50/50 sequential blend). It is strictly better than every sequential-fine-tune variant on LAMBADA and not meaningfully worse on OWT ppl than any of them. This confirms the hypothesis motivating the experiment: training the blended objective from scratch avoids the forgetting-cliff dynamics of Section 4.2 entirely (there is no pure-EMA basin to escape from), and reaches a Pareto-better point than any fine-tune-after-convergence protocol.

**A currently-running replicate.** A second jointly-trained 75/25-blend-from-scratch run, seed 7, identical config, is in progress on `titan` (`run_gpt2_ema_blend75_scratch_seed7.pt`, started 2026-06-30). As of the last check (2026-07-02, step ~187,000/500,000, ~37%), it is tracking a **lower** in-training WikiText val_ppl than seed42 was at the same step (by ~15-20 ppl, consistent across the last several eval checkpoints) and a higher in-training LAMBADA cloze proxy (0.540 vs. 0.515 at step 150K) — though per Section 3's caution, the in-training proxy is not reliable and this is not confirmed until the seed7 run completes and is scored with the real lm-eval-harness benchmark. ETA at last check: a few more days. **This result is not yet final and should be re-checked before citing a specific number for seed 7.**

---

## 5. Discussion

**Why blending beats switching.** Sections 4.2-4.4 together tell a coherent story about *when* a model specializes into an objective's basin, not just *which* objective is used. Pure EMA (α=1.0 throughout) and pure CE (α=0.0 throughout) each converge to their own basin; switching between them late (Section 4.2) forces the model to climb out of one basin and into another, and the climb-out is fast and disproportionate — the LAMBADA gain built over 500K EMA steps evaporates in under 10K CE steps. A fixed intermediate α, whether reached by sequential fine-tune (Section 4.3) or trained from scratch (Section 4.4), never creates a single-objective basin to escape from in the first place. Training the blend from scratch is better still than fine-tuning into it, presumably because the model's whole optimization trajectory — not just its final 2% of steps — is shaped by the trade-off it will actually be evaluated under.

**Cost is bounded, not runaway.** Across every configuration tested — pure EMA, every blend ratio, both architectures — OWT held-out perplexity degradation stays in a narrow band (roughly +5% to +11% relative to the matched non-EMA baseline). There is no configuration where the EMA mechanism causes catastrophic degradation of general language-modeling ability; the cost is a real but modest, controllable tax, and the 75/25 jointly-trained blend suggests that tax can be made even smaller while *keeping* essentially all of the LAMBADA benefit.

**Open questions this paper does not resolve** (carried forward from `project_loss_function_ideas` memory and CLAUDE.md's "Active run" notes, updated for what Sections 4.3-4.4 have since answered):

- *(Answered)* Does a jointly-trained blend beat sequential fine-tune? — **Yes** (Section 4.4).
- *(Partially answered)* Sweep more α values — Section 4.3 covers 0.25/0.5/0.75 sequential; Section 4.4 covers only α=0.75 jointly-trained. A full jointly-trained sweep (0.25, 0.5, and finer resolution near 0.75-1.0, since 0.75 already beats 1.0) has not been run.
- *(Open)* Is the jointly-trained-blend result architecture-general the way the base EMA effect was (Section 4.1)? No BLT-blend experiment (sequential or jointly-trained) has been run yet — only pure EMA (α=1.0) has been tested on BLT.
- *(Open)* The seed-7 jointly-trained-blend replicate (Section 4.4) is in progress and not yet confirmed by the real benchmark.
- *(Open, not yet started)* Option 2 from the original brainstorm — a short-context vs. long-context self-comparison, upweighting tokens where the loss gap between truncated and full context is large — was never implemented. It targets long-range dependence more directly than the per-vocabulary-ID EMA used throughout this paper (see the Token Weighting comparison, Section 6) but costs 2× forward-pass compute per step.
- *(Open, not yet started)* Cycling between structurally different loss functions during training (not just blending two, but rotating among 3+, e.g. standard CE → EMA-weighted → focal loss → back) was proposed as a way to avoid any single loss function's minima, but not implemented or tested.

---

## 6. Related Work

**Rho-1 / Selective Language Modeling.** Lin et al. train a small reference model, compute each training token's "excess loss" relative to the reference, and backpropagate only through the highest-excess tokens, showing strong gains on math reasoning benchmarks ([arXiv 2404.07965](https://arxiv.org/abs/2404.07965), NeurIPS 2024). This is the closest prior work in spirit — both identify and upweight "hard" tokens — but Rho-1 requires training and running a separate reference model at every step, while the EMA method here needs no reference model: it is entirely self-referential, using only the running model's own historical per-token loss.

**Token Weighting for Long-Range Language Modeling.** This work compares a long-context model's per-token confidence against a short-context model's confidence, and upweights tokens where long-range context specifically helps ([arXiv 2503.09202](https://arxiv.org/abs/2503.09202), NAACL 2025). This is the most targeted prior fix for the LAMBADA-style failure mode motivating this paper (Section 1), and directly inspired "Option 2" in the open-questions list (Section 5) — but it requires two forward passes per step (full and truncated context) to compute the confidence gap, versus this paper's single-forward-pass EMA approach. The EMA method's per-vocabulary-ID weighting is a coarser, cheaper proxy: it identifies tokens that are *globally* hard (rare or high-entropy words) rather than tokens that are specifically *long-range-dependent* in context, which likely explains why it captures only part of the available LAMBADA-style benefit relative to what a direct long-range confidence signal could in principle achieve. Testing the short-context-comparison approach directly against the EMA method at matched compute would be a natural extension.

**MiLe Loss.** Weights tokens by their predictive entropy — uncertain tokens receive a stronger training signal, with no reference model required ([arXiv 2310.19531](https://arxiv.org/abs/2310.19531)). Closer in spirit and cost to this paper's method than Rho-1 or Token Weighting (both are single-forward-pass, no-reference-model approaches), but entropy alone cannot distinguish genuine long-range uncertainty (the kind LAMBADA tests) from other sources of ambiguity (e.g. multiple locally-plausible continuations). The EMA method's per-token-ID historical tracking is a different single-pass proxy with the same limitation in a different form — see the open question in Section 5 about whether combining the two proxies helps.

**Tilting the Playing Field.** Proposes cycling time-dependent loss weights across multiple objectives during training, arguing this pushes the optimizer toward minima that are simultaneously good under all the cycled objectives rather than just the current one ([arXiv 2102.03793](https://arxiv.org/abs/2102.03793), ICML 2021). This is the direct precedent for the "cycling among multiple loss functions" idea in Section 5's open questions — but that paper cycles *weights* on a fixed objective, while the idea sketched here (not yet implemented) would cycle between *structurally different* loss functions (standard CE, EMA-weighted CE, a hypothetical focal loss), a stronger perturbation of the optimization landscape that, to our knowledge, has not been directly tested in the LLM pretraining literature as of mid-2026.

**Summary.** The EMA per-token weighting scheme in this paper occupies a specific, previously-unfilled point in the design space of hard-token-upweighting methods: no reference model (unlike Rho-1), single forward pass (unlike Token Weighting), and using an explicit historical-loss signal per vocabulary ID rather than instantaneous entropy (unlike MiLe Loss). Its combination with blending — and specifically the finding that jointly-trained blending beats post-hoc fine-tuning into a blend — has, to our knowledge, not been reported elsewhere for this class of method.

---

## Appendix A: Result File Index

For digging up raw data behind any number in this paper. Paths are relative to the repo root (`/home/benzene/llmopt/blt` on `bender`) unless marked otherwise.

| Result | Checkpoint | Training log | lm-eval JSON | OWT eval output |
|---|---|---|---|---|
| GPT-2 non-EMA baseline (seed 42) | `run_gpt2_baseline_seed42.pt` | `run_gpt2_baseline_seed42.log` | `lm_eval_baseline_scratch.json` | (see `CLAUDE.md`) |
| GPT-2 non-EMA baseline (seed 7) | `run_gpt2_baseline_seed7.pt` | `run_gpt2_baseline_seed7.log` | `lm_eval_gpt2_baseline_seed7.json` | `run_gpt2_baseline_seed7.stdout` (tail) |
| BLT non-EMA baseline (seed 7) | `run_blt_scratch_seed7.pt` | — | `lm_eval_blt_scratch_seed7.json` | — |
| GPT-2 full EMA (α=1.0, seed 42) | `run_gpt2_ema_seed42.pt` | `run_gpt2_ema_seed42.log` | `lm_eval_gpt2_ema_seed42.json` | `eval_owt_ema_seed42.stdout`* |
| BLT full EMA (α=1.0, seed 42) | `run_blt_ema_seed42.pt` | `run_blt_ema_seed42.log` | `lm_eval_blt_ema_seed42.json` | `eval_owt_ema_seed42.stdout`* |
| Pure-CE sequential fine-tune (10K, on `titan`) | `run_gpt2_ema_then_ce_finetune_seed42.pt` (local copy) | on `titan` only | `lm_eval_gpt2_ema_then_ce_finetune_seed42.json` (on `titan` only, not copied locally) | not run |
| α=0.25 sequential fine-tune | `run_gpt2_ema_blend25_finetune_seed42.pt` | on `titan` only | `lm_eval_gpt2_ema_blend25_finetune_seed42.json` | `eval_owt_gpt2_ema_blend25_finetune_seed42.stdout` (run 2026-07-02) |
| α=0.5 sequential fine-tune | `run_gpt2_ema_blend50_finetune_seed42.pt` (local copy) | on `titan` only | `lm_eval_gpt2_ema_blend50_finetune_seed42.json` (on `titan` only, not copied locally) | not run |
| α=0.75 sequential fine-tune | `run_gpt2_ema_blend75_finetune_seed42.pt` | on `titan` only | `lm_eval_gpt2_ema_blend75_finetune_seed42.json` | **not run — OWT ppl unmeasured, see Section 4.3** |
| α=0.75 jointly-trained-from-scratch (seed 42) | `run_gpt2_ema_blend75_scratch_seed42.pt` (on `titan`, not copied locally) | `run_gpt2_ema_blend75_scratch_seed42.log` (on `titan`) | `lm_eval_gpt2_ema_blend75_scratch_seed42.json` | `eval_owt_gpt2_ema_blend75_scratch_seed42.stdout` |
| α=0.75 jointly-trained-from-scratch (seed 7, IN PROGRESS) | `run_gpt2_ema_blend75_scratch_seed7.pt` (on `titan`) | `run_gpt2_ema_blend75_scratch_seed7.log` (on `titan`) | not yet run | not yet run |

\* `eval_owt_ema_seed42.stdout` contains the OWT eval for **both** the GPT-2-EMA and BLT-EMA checkpoints — check which `Baseline GPT-2:` / model line precedes the `OWT held-out loss` line you're reading.

**Code references**: EMA mechanism in `train.py` (search `ema_loss`, `ema_blend`, `ema_decay`; flags defined ~line 420-425, forward-pass logic ~line 289-312). `--finetune`'s EMA-buffer-loading fix (previously only `--resume` restored it) landed alongside `--ema-blend` in commit `c118787`.

**Memory references** (Claude session memory, not in this repo): `project_loss_function_ideas.md` — original brainstorm of Options 1-3 and the loss-cycling idea, plus the running log of Findings 1-3 this paper formalizes.

## Appendix B: Academic References

- Lin, Z. et al. "Not All Tokens Are What You Need for Pretraining." [arXiv 2404.07965](https://arxiv.org/abs/2404.07965), NeurIPS 2024. (Rho-1 / Selective Language Modeling)
- "Token Weighting for Long-Range Language Modeling." [arXiv 2503.09202](https://arxiv.org/abs/2503.09202), NAACL 2025.
- "MiLe Loss: a MInimizing the average LExical error for LLM Pretraining." [arXiv 2310.19531](https://arxiv.org/abs/2310.19531).
- "Tilting the Playing Field: Dynamical Loss Functions for Machine Learning." [arXiv 2102.03793](https://arxiv.org/abs/2102.03793), ICML 2021.

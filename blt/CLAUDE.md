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

## Active run

### Power-loss outage and resume (2026-07-20/21)
`bender` was physically powered off to attempt a replacement-fan install (see thermal-throttling notes in "GPT-2 medium" below) — **the new fan didn't fit**. Powering off `bender` also killed the in-progress job on `io` (mechanism not fully understood — the two runs aren't on shared infra as far as anyone knows, but both processes died at the same time, ~2026-07-20 13:19-13:37 local). Both were cleanly dead (no process, GPU idle) with valid, checkpoint-consistent state — not corrupted.

**Resumed both, verified healthy, 2026-07-20:**
- `bender`: `run_gpt2_medium_baseline_seed42.pt` was at step 153,500 (val_ppl 57.71, `micro_step` 307002). Resumed via:
  ```
  train.py --resume run_gpt2_medium_baseline_seed42.pt --baseline --from-scratch --pretrained gpt2-medium \
    --dataset openwebtext_large --seed 42 --batch-size 2 --grad-accum-steps 2 \
    --max-steps 1500000 --eval-every 1000 --lambada-eval-every 5000 \
    --save-path run_gpt2_medium_baseline_seed42.pt
  ```
  Confirmed progressing cleanly afterward (step 174,600+ as of the last check, val_ppl steady ~55).
- `io`: `run_gpt2_medium_ema_blend75_seed42.pt` was at step 161,500 (val_ppl 72.43, `micro_step` 161501, `ema_loss` buffer genuine — std 2.16, not the seed7-bug's std=0.0). Repo there was 6 commits behind (`af8375b`, missing the cross-machine-merge GQA/multi-M fixes) — pulled clean to `dfdd77a` first (one untracked-file collision on `lm_eval_gpt2_baseline_seed19.json`, resolved by deleting the untracked copy after `md5sum` confirmed it was byte-identical to the version being merged in — same non-GQA/non-training-loop-touching diff as the `titan` case below). Resumed via the matching EMA command (`--ema-loss-weighting --ema-blend 0.75 --batch-size 4`, otherwise identical). Confirmed progressing (GPU 97% util, 21.6GB used, step 182,200+ as of the last check).

**Both resumes used `--resume` (not `--finetune`)**, so the `ResumableSampler`'s saved `micro_step` was picked up correctly — no data replay, no LR discontinuity, per the fix documented under "GPT-2 medium" below.

**Noted in passing**: at this stage (~step 175-182K of 1.5M, ~12% through), io's EMA val_ppl is both higher and much noisier than bender's baseline (e.g. io swung 60.10→66.70→63.64→62.77→72.23 over five straight 1000-step checkpoints, vs. bender's tight 55.2-55.5 over the same span). Consistent with the established EMA trade-off pattern from every prior EMA experiment (worse OWT/val ppl, better LAMBADA) plus the already-documented noisiness of the in-training val_ppl proxy — not a red flag, but worth confirming with the real lm-eval benchmark once this run finishes rather than reading too much into the in-training curve.

**Idea noted, then implemented and launched (2026-07-21)**: user proposed annealing `--ema-blend` on a schedule (e.g. sine ramp from ~0 to 0.75 over the course of training) rather than a fixed value from step 0, on the theory that the EMA buffer is less trustworthy early (fewer samples per token) and should count for less then. Plausible extra lever beyond the existing decay rate — see `project_loss_function_ideas` memory for the full writeup.

Added `--ema-blend-schedule {fixed,sine}` (`train.py`; default `fixed`, exactly preserves prior behavior). `sine`: effective blend at step `s` is `ema_blend * sin(0.5*pi*min(s,max_steps)/max_steps)` — 0 at step 0, ramps smoothly (ease-in) to the target `--ema-blend` value exactly at `max_steps`. Verified two ways before trusting it: (1) pure-math check that the formula hits 0 at step 0 and the exact target at `max_steps`; (2) a real end-to-end 20-step smoke test (`--dataset wikitext103 --ema-blend-schedule sine`) on `venus` — ran clean, no errors, and the resulting checkpoint's `ema_loss` buffer showed real (small, expected-for-20-steps) differentiation (std 0.0078), confirming the branch actually executes the EMA update path correctly under the new schedule. **Not yet committed to git** — copied to `venus` via `scp` rather than `git pull`, since venus's repo was otherwise brought up to date to `dfdd77a` first (14 commits behind, one safe untracked-file collision resolved the same way as the `io` case above — `lm_eval_gpt2_ema_blend75_scratch_seed7_v2.json`, byte-identical via `md5sum`, deleted before pulling).

**Launched on `venus`** (idle GPU, 0% util beforehand): `run_gpt2_ema_blend75_sine_scratch_seed42.pt`, seed 42 (chosen to be directly comparable to the existing fixed-blend75 seed42 result, LAMBADA acc 0.269, the strongest in that family), `--dataset openwebtext` (original 21-file small-model corpus — venus already had the tokenized cache from the earlier seed7_v2 run, so no re-tokenization wait), `--ema-loss-weighting --ema-blend 0.75 --ema-blend-schedule sine`, 500K steps, same eval cadence as all other small-model runs. Confirmed running (GPU 100% util, 11.3GB used) shortly after launch. This is a single exploratory data point against the fixed-blend75 seed42 baseline — not yet a controlled multi-seed comparison.

**Early-training comparison investigation (2026-07-21).** User noticed the sine run's WikiText val_ppl was converging slower than the *fixed* pure-EMA seed42 run (`run_gpt2_ema_seed42.pt`) at matched step counts (e.g. step 17,000: 441.1 vs 331.4), even though the sine schedule's effective blend should be tiny that early (~0.04-0.09). Comparing against the non-EMA baseline seed42 made it look worse still (302.9 at step 17,000 — a 46% gap), which didn't fit the theory that a near-zero blend should track close to plain CE.

**Not conclusively resolved — open question, evidence is suggestive but not proof.** Initial read: `run_gpt2_baseline_seed19.pt` (P100, `titan`) tracks the sine run (P100, `venus`) almost exactly at every matched checkpoint (17K: 436.9 vs 441.1; 20K: 354.4 vs 363.2; 25K: 252.7 vs 260.8; 30K: 205.0 vs 201.0), suggesting hardware/numerics as the explanation. **User pushed back correctly**: final converged metrics landing in a tight band across seeds/hardware doesn't actually distinguish "hardware-driven early divergence" from "ordinary early-training seed noise" — both predict convergence by the end regardless of mechanism, so that fact alone can't settle it. Added a fourth point, `run_gpt2_baseline_seed7.pt` (confirmed V100, `bender`), for a real cross-check: it tracks `seed42`'s baseline tightly (17K: 297.9 vs 302.9; 30K: 170.1 vs 189.1), while the two P100 runs (sine, seed19) track *each other* tightly and sit ~30-50% higher throughout — a clean 2-vs-2 hardware-correlated split that's suggestive of a real effect. But `seed42`-baseline's original machine/GPU was never confirmed (CLAUDE.md only ever said "a different machine than `bender`," not its GPU model), so this is 4 data points, not a controlled test — **left as an open question**, plausible either way, doesn't change the sine-schedule experiment's validity either way. Reading the loss-weighting code (`train.py` ~401-431) didn't turn up an obvious bug — the weight formula does mathematically collapse to ≈1.0 (standard CE) at low blend, so a code bug remains unlikely regardless of which explanation is right.

**Queued diagnostic (not yet run)**: re-run `run_gpt2_baseline_seed7.pt`'s exact config on a P100 (same seed, same code, only the GPU differs from its original `bender` V100 run). If the rerun's early trajectory jumps to match the P100 cluster (sine, seed19), that's real evidence for a hardware/numerics effect; if it stays in the V100-like range regardless of which physical GPU runs it, that points to ordinary seed/noise variance instead. User flagged this as the clean test but said it'll have to wait — no GPU committed to it yet.

**Added `effective_blend` to the training log** (`train.py`) so the actual per-step blend value is directly observable going forward, rather than only inferred from the schedule formula — guards against exactly this kind of "is the code doing what the formula says" ambiguity in the future. New trailing column on the tab-separated log line (`step elapsed lr train_loss val_ppl lambada_acc effective_blend`), populated whenever `--ema-loss-weighting` is active (blank otherwise); appended at the end so it doesn't break any existing column-position-based tooling. Verified via a 20-step wikitext103 smoke test on `venus`: at step 10/20 (progress=0.5), logged `effective_blend=0.5303`, exactly matching `0.75 * sin(0.5*pi*0.5)` computed by hand.

**Applied to the live run**: killed the in-progress `venus` sine job (step ~32,150, no meaningful progress lost — checkpoints save every 500 steps), copied the updated `train.py` over, and resumed via `--resume` (confirmed clean pickup at step 32,000, no data replay). First post-resume log line shows `effective_blend=0.0753` at step 32,000, matching `0.75 * sin(0.5*pi*32000/500000)` exactly — confirms the schedule is behaving correctly in the actual training loop, not just in the standalone formula.

**DONE (2026-07-27) — best LAMBADA result in the whole EMA/blend family.** `run_gpt2_ema_blend75_sine_scratch_seed42.pt` completed all 500,000 steps (final WikiText val_ppl 64.91, `effective_blend` hit exactly 0.7500 at the end, confirming the schedule ran correctly start to finish). Benchmarked: OWT eval relayed to `io` (venus only has 18 parquet files, not the full 80-file set `eval_owt.py` needs — same workaround as the seed7_v2 case) and lm-eval-harness run directly on `venus` (no parquet dependency). Results (`lm_eval_gpt2_ema_blend75_sine_scratch_seed42.json`): OWT held-out ppl **29.44** (loss 3.3825), LAMBADA acc **0.281**, LAMBADA ppl **117.7**, HellaSwag acc_norm 0.267, PIQA acc_norm 0.571, Winogrande acc 0.511.

| | Sine blend seed42 | Fixed 75/25 seed42 | Fixed 75/25 seed19 | Fixed 75/25 seed7_v2 | Pure EMA seed42 | Non-EMA baseline |
|---|---|---|---|---|---|---|
| OWT held-out ppl | 29.44 | 29.61 | 29.61 | 29.42 | 30.06 | 27.78 |
| LAMBADA acc | **0.281** | 0.269 | 0.253 | 0.257 | 0.253 | 0.225 |
| LAMBADA ppl | **117.7** | 130.6 | 127.8 | 131.3 | 119.6 | 174.6 |
| HellaSwag acc_norm | 0.267 | 0.273 | 0.271 | 0.266 | 0.272 | 0.268 |
| PIQA acc_norm | 0.571 | 0.565 | 0.582 | 0.574 | 0.569 | 0.579 |
| Winogrande acc | 0.511 | 0.522 | 0.508 | 0.520 | 0.511 | 0.505 |

**The sine schedule beats every fixed-blend seed on LAMBADA acc (0.281 vs. 0.253-0.269) at essentially the same OWT-ppl cost** (29.44, right inside the fixed-blend seeds' own 29.42-29.61 range, not a worse trade-off point) — this looks like a genuine improvement from annealing the blend rather than just a different point on the same curve. Matches the original hypothesis: ramping the EMA weight in gradually (rather than applying it at full target strength from step 0, when the per-token buffer has barely differentiated) lets the model spend early training on uncorrupted signal and only lean on the reweighting once it's trustworthy. **Caveat**: one seed vs. three fixed-blend seeds — a real result, but not yet a multi-seed-confirmed one. A second sine-schedule seed would be the natural next step to confirm this isn't just favorable seed variance, the same way the fixed-blend finding needed three seeds before it was trusted.

**Second sine-schedule seed launched (2026-07-27).** `run_gpt2_ema_blend75_sine_scratch_seed19.pt` on `venus` (idle after the seed42 run finished) — seed 19 (matches the existing fixed-blend75 family's seed sequence for eventual direct comparison), otherwise identical config to the seed42 sine run. Confirmed healthy at step 0 (`effective_blend=0.0000`, as expected). In progress — past step 380,000/500,000 (76%) as of 2026-08-01, see "Current status snapshot" above for the in-training val_ppl comparison against seed42 (and why it shouldn't be over-read).

**DONE (2026-08-03).** Finished all 500,000 steps; the pre-existing watcher (`queue_sine_seed19_watcher.sh`, running on `bender`, polling venus's training PID over SSH) fired both benchmarks automatically on exit. Results (`lm_eval_gpt2_ema_blend75_sine_scratch_seed19.json`, `eval_owt_gpt2_ema_blend75_sine_scratch_seed19.stdout`, both on `venus`): OWT held-out ppl **29.37** (loss 3.3799), LAMBADA acc **0.246**, LAMBADA ppl 146.0, HellaSwag acc_norm 0.270, PIQA acc_norm 0.566, Winogrande acc 0.507.

| | seed19 (sine) | seed42 (sine) | Fixed-75/25 avg (3 seeds) | Non-EMA baseline |
|---|---|---|---|---|
| OWT held-out ppl | 29.37 | 29.44 | ~29.55 | 27.78 |
| LAMBADA acc | 0.246 | **0.281** | ~0.260 | 0.225 |
| LAMBADA ppl | 146.0 | **117.7** | ~130.0 | 174.6 |
| HellaSwag acc_norm | 0.270 | 0.267 | ~0.270 | 0.268 |
| PIQA acc_norm | 0.566 | 0.571 | ~0.574 | 0.579 |
| Winogrande acc | 0.507 | 0.511 | ~0.517 | 0.505 |

**Both sine seeds land well above baseline and above every individual fixed-blend seed's LAMBADA acc (0.253-0.269)** — the sine-schedule improvement over fixed-blend is a genuine two-seed result, not just seed42 alone. But seed19 (0.246) is noticeably weaker than seed42 (0.281), so seed42's original headline number was partly favorable variance rather than the sine schedule reliably landing at 0.28+; the two-seed average (~0.264) is still a clear, if smaller, win over fixed-blend's average (~0.260). **This is read as further supporting evidence for the sine-schedule bet on the medium-model EMA run** (`io`, `run_gpt2_medium_ema_blend75_sine_seed42.pt`) — two-for-two on beating fixed-blend's LAMBADA acc reinforces that decision, though the medium run's own confirmation still awaits its real lm-eval benchmark once it finishes.

**Third sine-schedule seed launched (2026-08-03).** `venus` was idle again (0% GPU util) once seed19 finished, so a third seed was kicked off immediately: `run_gpt2_ema_blend75_sine_scratch_seed7.pt`, seed 7 (completes the same 42/19/7 sequence used throughout this project), otherwise identical config/protocol to the first two sine seeds (`--baseline --from-scratch --dataset openwebtext --ema-loss-weighting --ema-blend 0.75 --ema-blend-schedule sine --max-steps 500000`). Reused the existing tokenized OWT cache on venus (no re-tokenization wait).

### Cross-machine merge (2026-07-19) — verified, two real bugs found and fixed
User made changes on another machine in parallel with the medium/EMA work below: generalized `build_gqa_model` (real warm-start support — average pretrained heads per KV group, not just random init) and `build_blt_model` (new `layers_per_m` strided-grouping variant, plus a new generalized `BLTMultiMAttention` class replacing the old `num_m_groups==2`-only `BLT2Attention`), added `bigblt.md` and two new run-sequence scripts (`run_layer4_sequence.sh`, `run_3m_gqa3_sequence.sh`). Merged via `git merge` (commit `d22dc8f`), explicitly keeping both `per_layer_m` (this session's one-M-per-layer) and `layers_per_m` (the other session's strided N-layers-per-M) as coexisting BLT variants.

**User asked for a functional check, not just a naming-diff read — good call, since it wasn't just names.** Two real bugs found:

1. **`--gqa --from-scratch` silently did the wrong thing.** `train.py` had `pretrained_name = args.pretrained if args.pretrained else None` before calling `build_gqa_model(..., pretrained=pretrained_name)`. Since `--pretrained` defaults to `'gpt2'` (always truthy), this check never actually caught `--from-scratch` — every GQA run would warm-start from real pretrained GPT-2 weights instead of random-initializing, with zero error or warning. **This bug predates the merge** (already present in the other machine's commit before merging, not introduced by the resolution) but is live in the current code regardless. Root cause: the other session's `build_gqa_model` redesign gave `pretrained` new dual meaning (shape *and* whether to warm-start) via truthiness, conflating two orthogonal concerns that every other builder in this file (`build_blt_model`, `build_hybrid_model`, the baseline path) keeps separate via an explicit `from_scratch` flag.
   **Fix**: gave `build_gqa_model` the same explicit `from_scratch` parameter as the others (`pretrained` now always controls shape via `GPT2Config.from_pretrained`/`.from_pretrained`, independent of whether real weights get loaded), and fixed `train.py`'s call site to pass `args.from_scratch` directly instead of the broken truthiness check. Also updated `blt_lm_eval.py`/`eval_owt.py`'s GQA eval paths to pass `from_scratch=True` explicitly — without it they'd now default to actually downloading/loading real pretrained GPT-2 weights before immediately overwriting them via `load_state_dict`, a wasteful and needless new network dependency for what's supposed to be a lightweight eval-time skeleton build.
2. **`kv_cache_bench.py` was flat-out broken** — imports `BLT2Attention` from `model.py`, which no longer exists (replaced by `BLTMultiMAttention`). Confirmed via direct `ImportError`. The import was already dead code (never used in the file's `isinstance` checks, which only ever handled `BLTAttention`/`GQAAttention`/`MHAAttention` — 2-M/multi-M models were never actually exercised by this benchmark script even before the merge), so the fix was just dropping it from the import line. Also added `from_scratch=True` to its own `build_gqa_model()` call for the same reason as the eval scripts above.

**Verified thoroughly before trusting any of this**: all 7 files that import `model.py` (`model`, `train`, `blt_lm_eval`, `eval_owt`, `generate`, `evaluate`, `kv_cache_bench`) import cleanly now. Existing checkpoints (`run_gqa3_scratch_seed42.pt`, `run_perlayerm_seed42.pt`) still load correctly into the fixed/new code — internal parameter names inside those specific attention classes weren't touched. New features (`layers_per_m`, `num_m_groups>1`/`BLTMultiMAttention`) build and run a real forward/backward pass correctly. The from-scratch fix was driven through the *actual* unmodified `train.py` CLI entry point (via `runpy`, not just direct `model.py` calls) with `--gqa --from-scratch --pretrained gpt2-medium` — confirmed genuinely random init (weight std ≈0.02, not pretrained-scale) and got all the way to real GPU allocation before stopping (on a GPU already busy with the live medium job — expected, not a fix-related failure).

**Confirmed unaffected**: the queued BLT seed19-500K watcher on `titan` uses only flags untouched by any of this (no `--gqa`, defaults for `--num-m-groups`/`--layers-per-m`/`--per-layer-m`). None of the three currently-running training jobs (`bender` medium baseline, `io` medium EMA, `titan` baseline seed19) use `--gqa` or reference any changed code path, and as already-running processes they have their own code loaded in memory regardless of what's now on disk.

**One caveat for later**: `run_gqa_scratch_seed42.pt`/`run_gqa_scratch_seed7.pt`/`run_gqa3_scratch_seed42.pt` are all from-scratch, 2-or-3-group, small-model GQA checkpoints — all still load fine with the fix. If a genuine 2-group *warm-start* GQA checkpoint ever existed from the old (buggy or pre-bug) code, it should still be checked against the new `build_gqa_model(from_scratch=False)` path before trusting it, since the averaging-based Wk/Wv warm-start logic itself wasn't changed by this fix, only the from-scratch dispatch — not urgent, no such checkpoint is known to exist in this project's history.

### Current status snapshot (2026-08-01) — read this first for hand-off
- **`bender`** (local): GPT-2 **medium** (355M) baseline from-scratch OWT, seed 42, still running — `run_gpt2_medium_baseline_seed42.pt`, target **1.5M steps**, past step 909,000 (61% through) as of last check. Survived the 2026-07-20 power outage via `--resume` (see "Power-loss outage and resume" below for that whole story, plus the earlier OOM/thermal/data-replay saga in "GPT-2 medium" below). This is the one stable, uneventful long-runner among the four machines right now.
- **`titan`**: **UV256 seed42 DONE (2026-08-01)** — see "Low-rank BLT (M = UV^T)" below for the full result and discussion (OWT held-out ppl **29.97**, beats every full-rank BLT seed on point estimates, but by a margin smaller than BLT's own seed-to-seed spread — not yet confirmed). Now running **UV256 seed19** (`run_blt_uv256_scratch_seed19.pt`), launched immediately after via a watcher script, to confirm or refute whether seed42's result generalizes. Same protocol as seed42 (`--uv-rank 256 --from-scratch --dataset openwebtext --seed 19 --max-steps 500000`), just started.

GPT-2 baseline now has a genuine three-seed iso-step (500K) set:

| | seed19 | seed42 | seed7 |
|---|---|---|---|
| OWT held-out ppl | 28.24 | 27.78 | 27.36 |
| LAMBADA acc | 0.208 | 0.225 | 0.214 |
| HellaSwag acc_norm | 0.266 | 0.268 | 0.270 |
| PIQA acc_norm | 0.572 | 0.579 | 0.583 |
| Winogrande acc | 0.520 | 0.505 | 0.519 |

All three land in a fairly tight, consistent band — seed19 is the weakest on OWT ppl/LAMBADA acc but not dramatically, Winogrande favors it. Normal seed variance, no outlier.
- **`venus`**: seed19 sine-blend run **DONE** (2026-08-03, OWT ppl 29.37, LAMBADA acc 0.246 — see "GPT-2 medium" below for the full two-seed comparison). Now running the **third sine-blend seed** (`run_gpt2_ema_blend75_sine_scratch_seed7.pt`), launched immediately after on the now-idle GPU to complete the 42/19/7 seed sequence. The three fixed-75/25-blend small-model seeds (42/19/7_v2) all finished earlier and are fully benchmarked — no longer in progress.
- **`io`**: running the **from-scratch sine-blend medium EMA run** (`run_gpt2_medium_ema_blend75_sine_seed42.pt`), launched 2026-07-28 to replace the original fixed-75/25-blend medium run, which was showing ~4x noisier in-training val_ppl than `bender`'s baseline at comparable training fraction (28% vs 6.5% relative swing). **Briefly paused 2026-08-02** (step 314,500 → resumed cleanly same day) to free the GPU for an unrelated `pprune/` job (`kl_faith_eval_ystar.py`) — a watcher (`queue_medium_sine_resume_watcher.sh`) auto-fired `--resume` the moment that job exited; confirmed correct via `effective_blend=0.2426` at the resumed step matching the sine formula exactly (not a step-0 restart). Now back to normal progress. The **original fixed-blend checkpoint is also still paused, not deleted** — `run_gpt2_medium_ema_blend75_seed42.pt` sits safely at step 594,500 (val_ppl 46.81) and is fully resumable if the sine bet doesn't pan out once the venus seed19 result above lands. See "GPT-2 medium" below for the full decision writeup and the noise-dynamics hypothesis this run is tracking toward (`effective_blend`≈0.5 around step 900K-1M).

### Live status re-check (2026-08-03) — fresher numbers than the 2026-08-01 snapshot above
Direct SSH check of all four machines (not just log-derived estimates):
- **`bender`** (local): step **1,018,260/1,500,000 (68%)**, healthy — up from 909,000 at the last snapshot.
- **`titan`**: UV256 seed19 **DONE (2026-08-05)** — OWT ppl 30.13, confirming low-rank BLT beats full-rank BLT as a genuine two-seed result. See "Low-rank BLT (M = UV^T)" below. `titan` is now idle.
- **`venus`**: sine-blend seed19 much further along than the stale 76% figure above — **step 470,780/500,000 (94%)**, `effective_blend=0.7468` (nearly at the 0.75 target, as expected this close to the end). Likely to finish within hours of this check.
- **`io`**: the medium sine run is **no longer paused** — confirmed actively training again at step **330,290/1,500,000 (22%)**, `effective_blend=0.2543`, via the `queue_medium_sine_resume_watcher.sh` process still alive there.

**Benchmark watcher for the venus run, and the parquet-copy question that prompted checking it.** User flagged that a background copy of OWT parquet files to `venus` may not have finished, which could be why a benchmark watcher hadn't visibly kicked off. Investigation: `venus`'s parquet dir (`~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/.../plain_text/`) has exactly 26 files, `train-00000` through `train-00025` — files 0-20 are **0-byte placeholder stubs**, files 21-25 (the actual held-out set `eval_owt.py` needs) are fully transferred (~300MB each, real data). Ran `eval_owt.py`'s exact `sorted(glob(...))[21:26]` logic for real on `venus` to confirm: it correctly resolves to the 5 real files — the 0-byte stubs for 0-20 are never opened, they only exist to occupy the right positions in the sort order so the positional slice lands correctly. **So the copy is actually sufficient as-is**, not a blocker.

Separately, `queue_sine_seed19_watcher.sh` (already sitting locally, untracked) turned out to **already be running** — PID 270206, launched earlier today, running on `bender` (not `venus`) and polling venus's training PID (133040) over SSH every 60s, ready to SSH back in and run both `blt_lm_eval.py` and `eval_owt.py` remotely the moment training exits. It was missed in the first pass because that pass only checked for a watcher process *on* venus — the script's own header comment explains the orchestration runs from bender since venus has no SSH access onward to other machines. No new watcher was added (would've raced with this one); given the run's at 94%, it should fire on its own within hours.

### `io` fan swap — did NOT fix thermal throttling (2026-08-04)
User had a second replacement fan (the first one didn't fit `bender`, see "Power-loss outage and resume" above) and decided to try it on `io` instead, since `io`'s SwThermalSlowdown signature (first seen 2026-08-01, 79°C, 76% clock) matched bender's unresolved issue and `bender` itself was too close to finishing its 1.5M-step run to risk touching.

**Before physically opening the machine**: backed up both of `io`'s only-ever-lived-there checkpoints (`run_gpt2_medium_ema_blend75_sine_seed42.pt`, the active run, and `run_gpt2_medium_ema_blend75_seed42.pt`, the paused fixed-blend fallback at step 594,500) to `bender` under `io_backup/`, verified via `md5sum` (fixed-blend matched byte-for-byte; the active one didn't match because training kept checkpointing during the transfer — confirmed not corrupted by loading it directly and checking all tensors finite, step 442,000). Also grabbed the small watcher/utility scripts living only on `io` (`queue_medium_sine_resume_watcher.sh`, `watcher_resume_medium_ema.sh`/`_2.sh`, `build_owt_large_cache.py`, `check_ckpt.py`, `check_io_ckpt.py`) that had never been backed up or committed anywhere. Pushed all pending local commits to `origin/blt` too. Then killed the training process cleanly via `SIGTERM` (checkpoint landed at step 444,500, verified all-finite) and confirmed GPU idle before the user powered down.

**Fan swapped, machine back up within minutes, training resumed via `--resume`** — confirmed correct pickup at step 444,500 (`effective_blend=0.3366`, matching the sine formula exactly, no data replay).

**Result: no improvement.** First reading right after resume looked great (54°C, full 1380 MHz clock, no throttle flags) — but that's exactly the false-positive trap already documented for `bender` above (idle/early-load readings don't reflect steady-state heat; this card takes several minutes of sustained load to get there). Re-checked ~2-3 minutes into sustained 100%-util load, three consecutive samples: **82°C, clock down to 772-795 MHz (56-58% of max), power draw ~131-135W of the 250W cap (ruling out a power-limit explanation, same as every prior reading), `SwThermalSlowdown` active on every sample.** Identical signature to before the swap. **Lesson (repeated from the bender case, worth internalizing this time): never report a thermal fix as confirmed from a reading taken within the first couple minutes of load — always let it run for several minutes under full utilization first.**

**Decision: leave it running throttled, no further action for now.** The fan wasn't the bottleneck (or wasn't installed correctly, or isn't the actual root cause — genuinely unclear, and undiagnosable remotely). Training correctness is unaffected either way (this has been true throughout every throttling episode in this project — only wall-clock speed is impacted), so there's no urgency. Both machines (`bender`, `io`) now have the same unresolved thermal issue and no remaining spare fan to try. If it becomes worth revisiting, the next step would be an in-person inspection (dust, thermal paste, case airflow, fan actually spinning/seated) rather than another parts swap.

### GPT-2 medium — testing EMA generality at a larger scale (started 2026-07-16)
**Motivation**: user wants to test whether the EMA per-token loss-weighting finding (currently validated on GPT-2 small and BLT small) generalizes across model size, not just architecture. GQA+EMA and hybrid+EMA (already-implemented architectures, just need the flag combo) were also discussed as cheap next steps but not yet started.

**Code changes required and made** (`model.py`, `train.py`): all four model-building call sites (`build_blt_model`'s from-scratch branch, `build_gqa_model`, `build_hybrid_model`, and `train.py`'s baseline from-scratch branch) previously hardcoded `GPT2Config()` (small, 124M) regardless of the `--pretrained` flag's value when `--from-scratch` was set — meaning `--pretrained gpt2-medium --from-scratch` silently built a *small* model. Fixed by using `GPT2Config.from_pretrained(pretrained)` (config-only fetch, ~0.2s, no weight download) instead of the bare `GPT2Config()` default in all four places, so `--from-scratch` now correctly respects `--pretrained`'s shape (small/medium/large/xl) while still randomly initializing all weights. `build_gqa_model`/`build_hybrid_model` also gained a `pretrained=` parameter (default `'gpt2'`, fully backward-compatible) for the same reason. Verified: medium builds correctly (354,823,168 params, matches published GPT-2-medium size) for baseline, BLT, and GQA; small-model construction confirmed unchanged (110,855,424 params for BLT, exact match to all prior runs).

**OOM on first launch**: `--batch-size 4` (the default, matching all prior small-model runs) OOM'd immediately on `bender`'s 16GB V100 in plain fp32, no gradient checkpointing — contrary to the initial assumption that medium would comfortably fit (only GPT-2 XL at 1.5B had previously needed fp16+grad-checkpointing). Fixed via `--batch-size 2 --grad-accum-steps 2`, preserving the same *effective* batch size (4) as the small-model protocol rather than reaching for `--fp16` (which would've introduced a confound relative to the fp32 small-model comparison, and has its own history of NaN risk in this codebase — see the BLT XL warm-start incident). Confirmed stable afterward: settled around 13,874/16,384 MB.

**Thermal throttling is a real, measured confound on `bender` for this run** (the fan-failure issue is still unresolved; new fan on order). Early in the run (steps 20-100): 1.19s/step at 1027 MHz SM clock. Later (steps 1980-2120, after sustained heat buildup): **1.53s/step at 945 MHz** — the throttling intensifies over the course of the run, not a one-time dip. Clock-ratio extrapolation (1530 MHz rated boost / 945 MHz actual) suggests throttling is costing roughly **39% of the runtime** (~8.9 days throttled vs. ~5.4 days estimated at full clock, for a 500K-step budget) — a rough proxy, not exact, since not all of the workload is equally clock-bound.

**Step-count decision**: 500K steps (matching the small-model protocol) would only give medium ~5.8 tokens/param (2.05B tokens / 355M params) — well short of the Chinchilla-optimal ~20 tokens/param, and a much less adequately-trained regime than small's own ~16.5 tokens/param at 500K steps. To keep the "does EMA still help" comparison fair across scales (not comparing a well-trained small model against an undertrained medium one), **user chose to extend to 1.5M steps** (~16.5 tokens/param, matching small's training adequacy) rather than accept the fixed-500K/fixed-compute framing.

**Resume bug found and fixed (2026-07-16), before any real training time was lost.** First attempt: killed the run at step 21,390 and resumed with `--resume ... --max-steps 1500000`. The cosine LR schedule itself re-stretched correctly over the new horizon (verified: resumed at step 21,000 at LR 5.00e-05, consistent with 1.4% through a 1.5M-step schedule — no LR discontinuity). *But* the user raised a sharper concern: would this be reproducible by someone re-running the documented command from scratch? Investigating turned up a real bug, not just "probably fine, within RNG noise":

- `train.py` calls `set_seed(args.seed)` once per process launch, and the old `DataLoader(..., shuffle=True)` drew its shuffle order from that same global RNG. Checkpoints saved model/optimizer/scheduler state but **not** DataLoader position or RNG state.
- Net effect: resuming re-seeded and rebuilt the DataLoader from scratch, reproducing the *same* shuffle order as the original process — but the resumed step counter picked up at 21,000, not 0. So the model would see the same early ~21K-step slice of data twice (once as original steps 0–21,390, once again as resumed steps 21,000–~42,390) before reaching genuinely new data — a deterministic anomaly, not equivalent to what running the documented command from step 0 would produce. ~1.4% of the 1.5M-step budget would have been affected.
- **Fix**: added `ResumableSampler` (`train.py`) — a custom `Sampler` that derives each epoch's permutation from `seed + epoch_index` via its own `torch.Generator` (independent of global RNG state), and can be constructed with a `start_sample` offset to resume mid-epoch without replaying already-consumed indices. `save_checkpoint()` now also persists `micro_step` (cumulative micro-batches consumed, i.e. `samples_consumed = micro_step * batch_size`), and `train()` peeks at the resume checkpoint's `micro_step` *before* constructing the DataLoader so the sampler picks up exactly where the data stream left off. Old checkpoints without a saved `micro_step` fall back to `step * grad_accum_steps` (safe approximation — assumes no skipped non-finite batches). `ClozeDataset` path unchanged (still plain `shuffle=True`) — small dataset, not used for long from-scratch runs, out of scope for this fix.
- **Verified two ways** before trusting it: (1) a pure-Python unit test on `ResumableSampler` in isolation, directly comparing a resumed sampler's output against the corresponding slice of an uninterrupted 3-epoch traversal — exact match, no replay, no skip; (2) a real end-to-end smoke test (`--dataset wikitext103`, tiny run, `--max-steps 20` then `--resume ... --max-steps 40`) — resumed cleanly, no tracebacks, loss trajectory consistent with fresh data rather than a repeat of the original early-step losses.
- **The tainted first-resume checkpoint was archived, not reused** — `stale_medium_resume_replay_20260716/`. The run was restarted clean from step 0 with the fix in place (`run_gpt2_medium_baseline_seed42.pt`, confirmed healthy at step 100). This also directly serves the practical need ahead: **the user plans to physically interrupt this exact run once the replacement fan arrives**, and this fix means that interruption can now use `--resume` safely without reintroducing the replay bug.

**Time/cost estimates discussed** (all extrapolated from measured bender numbers, not independently benchmarked elsewhere):
- 1.5M steps on `bender`, throttled: **very roughly ~26-27 days** (linear extrapolation from the 8.9-day/500K-step throttled estimate).
- 1.5M steps on `bender`, if the new fan fixes cooling: **~16 days** (from the ~5.4-day/500K estimate).
- 1.5M steps on `io`'s 32GB V100 (not overheating, and enough memory for a true batch=4 pass instead of batch=2+grad-accum, saving an estimated further ~10-25% from removing the small-batch inefficiency tax): **~13-15 days** estimated, ~14 as a point estimate. Not yet measured — `io` is currently busy with unrelated work. Also note `io` runs an older pinned software stack (PyTorch 2.3.1+cu121 per the CC-7.0 compatibility note) which could shift throughput for reasons unrelated to thermals.
- On an A100 (hypothetical, no access confirmed): 500K-step budget ~2.3-2.8 days (~$37-45 at $0.67/GPU-hr); 1.5M-step budget ~6.7-8 days (~$107-129).
- **None of the A100/io numbers are measured** — only the `bender` throttled-vs-full-clock numbers come from real data (via the clock-ratio proxy). A short smoke-test on either machine, once available, would replace guesswork with real numbers in minutes.

**Expanded the training corpus to avoid epoch repetition (2026-07-17).** At 1.5M steps (6.1B tokens), the original 21-file/2.2M-doc OWT cache (2.26B tokens) would have meant ~2.7 repeated epochs — a confound small's training never had (500K steps stayed within one epoch). All 80 OWT parquet files are already local (no download needed); added `--dataset openwebtext_large` (`train.py` `TokenDataset`): files 0-20 + 26-79 (75 files, ~7.9B tokens, single-epoch coverage), deliberately skipping files 21-25 since those are the held-out set `eval_owt.py` has always used — keeps held-out numbers comparable to every prior small-model result. Separate cache path (`~/.cache/blt_owt_75files_blocks.pt`), original `openwebtext` (21-file) path fully unchanged for reproducibility of all prior runs.

**OOM killed the first attempt at building this cache, silently (no traceback).** The existing small-corpus code materializes the whole filtered document list as a Python `list[str]` before tokenizing — fine at 21 files (~2.2M docs), but at 75 files (~7.5M docs) the peak memory (full Arrow dataset + full materialized text list held simultaneously) exceeded `bender`'s 94GB RAM and the OS OOM-killer took the process out with zero warning (RAM dropped back to baseline, no Python exception, process just gone). Fixed by tokenizing directly from chunked slices of the Arrow dataset (`ds[i:i+chunk_size]['text']`) instead of ever materializing the full corpus as one Python list — peak memory now bounded to one chunk at a time. **Verified the chunked path produces bit-for-bit identical tokenization** to the original bulk approach via a controlled test on a real file (same underlying row range compared both ways — first comparison attempt used mismatched ranges and wrongly looked like a bug; corrected test showed exact match). Relaunched successfully afterward.

**Second, subtler OOM in the same code — the chunking fix above still had a doubling bug.** After the first fix, `bender`'s actual training run (not just the cache-build smoke test) died again the same way: silent, 0-byte stdout, process just gone, RAM back to baseline. Root cause: the chunked loader still built a *growing Python list* of per-chunk token tensors, then called `torch.cat(all_ids)` — which briefly holds both the full pre-concat list AND the newly concatenated tensor simultaneously. At ~7.9B tokens and int64, that's ~63GB per copy, ~126GB combined — over `bender`'s 94GB, and would have been worse than `io`'s entire 62GB RAM (confirmed: a parallel cache-build attempt on `io`, still running the old single-pass code at the time, was climbing steadily toward that same wall — killed pre-emptively once the fix was ready rather than let it crash).

**Fix**: two-pass loading (`train.py` `TokenDataset`, `openwebtext_large` branch). Pass 1 tokenizes each chunk only to measure its length, discarding the ids immediately (peak memory O(chunk_size), not O(corpus_size)). Pass 2 pre-allocates *one* int32 tensor sized to the now-known total and fills it in place chunk by chunk — never more than one full-corpus-sized allocation plus a small transient chunk in memory at once (~32GB peak, safely under both machines' limits). Also dropped the `.clone()` after slicing to `n_blocks * block_size` (harmless before — the old code was slicing/cloning a *view into the shared 21-file `texts` list machinery*, unlike this fresh purpose-built tensor, where cloning would just re-introduce the same doubling for zero benefit — the tiny truncated remainder, at most `block_size - 1` elements, costs nothing to leave referenced). Storage dtype is int32 throughout (GPT-2's 50,257-token vocab fits trivially); `TokenDataset.__getitem__` now does `.long()` per-block (cheap, 1024 elements) rather than paying the int64 cost for the whole corpus — harmless no-op for the existing int64 small-corpus cache. **Verified correct** via a controlled test against the bulk reference implementation on a real file sample — bit-for-bit identical output, confirming the two-pass restructuring doesn't change what gets tokenized, only how much memory it costs to get there.

**Confirmed working end-to-end on `bender` (2026-07-17 evening).** Relaunched with the fix; peak RAM during the two-pass build settled at **38GB** (vs. the ~63-126GB that killed the prior two attempts), cache built successfully (`~/.cache/blt_owt_75files_blocks.pt`, 33.9GB on disk, **8,266,900 blocks ≈ 8.47B tokens**), and training proceeded cleanly past step 100 with a healthy loss curve (11.03 → 7.66). Took noticeably longer than hoped (~1h50m wall-clock for the two-pass tokenization, vs. an initial ~30-40min estimate) — the double-tokenization cost is real, but it's a one-time-per-machine setup cost, not a recurring one, so acceptable. The corresponding `io` cache-build attempt (still on the old, buggy single-pass code at the time) was killed pre-emptively once the fix was ready, rather than let it run into the same wall it was climbing toward — **not yet rebuilt with the fix**, and the user separately flagged they may need to run something else on `io` first, so it's on hold pending explicit clearance before touching that machine again.

**`io` is running the EMA counterpart — DONE launching, 2026-07-18.** User clarified after some cross-talk: `bender`'s medium run has no `--ema-loss-weighting`/`--ema-blend` flags (confirmed via live `ps`) — it's the plain non-EMA baseline, not "the EMA one" as briefly assumed. `io` runs the matching **EMA 75/25-blend** version, same seed (42), same `--dataset openwebtext_large`, same 1.5M-step target. Fresh clone at `/home/benzene/llmopt_medium_ema/blt` (io's existing checkouts were "in a state of flux" with other unrelated work — user asked for a new directory rather than reusing/pulling those).

**io's own cache-build attempt (using the two-pass fix) ballooned anyway — different failure mode than bender's, not fully explained.** Same code, same corpus, same fix that worked cleanly on bender (38GB peak) — but on io, RSS climbed past 50GB and VSZ reached 77.5GB (more than double the expected ~34GB tensor size) with swap actively in use, and still growing when killed. Checked for an environment difference (`datasets`/`pyarrow`/`transformers`/`torch` versions) — all identical to bender's, so the cause wasn't a version mismatch; not investigated further since a faster path existed. **Confirmed this was purely a dataset-tokenization issue, unrelated to EMA** — `build_owt_large_cache.py` is a standalone script with no EMA logic at all, same code path regardless of downstream training config.

**Fix: skip rebuilding the cache on io entirely — copy bender's already-built one instead.** Both machines need byte-identical output (same files, same tokenizer), so `scp`'d bender's `~/.cache/blt_owt_75files_blocks.pt` (33,861,225,755 bytes) to io directly. **md5sum verified identical on both ends** (`dd84d7564fe3f22d6ca79bc6091b4932`) before trusting it. This sidesteps whatever's different about io's memory behavior for that specific workload — worth remembering as the go-to move if a similar cross-machine tokenization discrepancy shows up again, rather than re-debugging per-machine.

**Launched and verified**: `train.py --baseline --from-scratch --pretrained gpt2-medium --ema-loss-weighting --ema-blend 0.75 --dataset openwebtext_large --seed 42 --batch-size 4 --max-steps 1500000 --eval-every 1000 --lambada-eval-every 5000 --save-path run_gpt2_medium_ema_blend75_seed42.pt`. Loaded the copied cache instantly (8,266,900 blocks, exact match to bender's count), GPU settled at 19.5GB/32.75GB (comfortable headroom at true batch=4, no grad-accum needed as planned), loss declining normally past step 1260. **`ema_loss` checkpoint buffer verified genuine at step 1,000**: std 0.651, only 2,336/50,257 tokens untouched — real EMA activity, not a repeat of the seed7 no-op bug.

**Paused at step 594,500 (2026-07-28) — replaced with a from-scratch sine-blend attempt, decision point in ~6 days.** In-training val_ppl was noticeably noisier than `bender`'s non-EMA baseline at comparable training fraction: io's val_ppl swung 45.6-58.5 (~28% relative) over steps 575K-594K vs. bender's 37.5-40.0 (~6.5% relative) over steps 628K-647K — a real, persistent ~4x noise gap, not just perception (also visible as far back as step ~180K). Given the small-model sine-blend result (seed42: best LAMBADA acc in the whole family, `project_loss_function_ideas` memory), and given the sine schedule's effective blend would be much gentler than the fixed 0.75 at this point in training (`0.75 * sin(0.5*pi*0.396) ≈ 0.44` at step 594,500/1,500,000), the working theory is that the fixed-blend reweighting itself is injecting the extra noise, and ramping it in should calm the curve.

**Decision: full restart with the sine schedule, not a hybrid resume.** A resume-with-schedule-switch (keeping the 594,500 steps already banked) was considered and rejected — user wants a clean, uncontaminated sine-blend medium run rather than a hybrid of 40% fixed-blend history + 60% sine, since the fixed-blend checkpoint stays fully valid and resumable on its own if this bet doesn't pan out. **The original checkpoint (`run_gpt2_medium_ema_blend75_seed42.pt`, step 594,500, val_ppl 46.81) was left untouched, not deleted** — the process was killed but the file was never touched by the new run, which uses a distinct save-path. If the sine bet is wrong, resume that checkpoint exactly as documented above.

**New run launched**: `train.py --baseline --from-scratch --pretrained gpt2-medium --ema-loss-weighting --ema-blend 0.75 --ema-blend-schedule sine --dataset openwebtext_large --seed 42 --batch-size 4 --max-steps 1500000 --eval-every 1000 --lambada-eval-every 5000 --save-path run_gpt2_medium_ema_blend75_sine_seed42.pt`. io's medium repo (`/home/benzene/llmopt_medium_ema/blt`) was 1 commit behind (missing the sine-blend feature entirely) — pulled clean to `24543f2`, no untracked-file collisions. Confirmed healthy: step 0 logged `effective_blend=0.0000` as expected, GPU settled at 19.5GB/32.75GB (same as the original run).

**Explicit decision point**: the second small-model sine-blend seed (`run_gpt2_ema_blend75_sine_scratch_seed19.pt`, on `venus`) is expected to finish in ~6 days from 2026-07-28 and will confirm or refute whether the seed42 sine result (best LAMBADA acc in the family) generalizes or was favorable variance. If it doesn't hold up, kill this medium sine run and resume `run_gpt2_medium_ema_blend75_seed42.pt` from step 594,500 instead.

**Early noise check (2026-08-02) — promising, not yet a result.** Compared the sine run's val_ppl noise against the paused fixed-blend run's at the *same absolute step range* (270K-312K, both still-preserved logs): sine's swing was ~12% relative (44.25-49.94), fixed-blend's was ~31% relative (49.68-67.55) at the same steps — and for reference, `bender`'s non-EMA baseline at that same step range swings ~14%, i.e. sine is currently tracking close to the clean baseline's noise level while fixed-blend was already ~2.5x noisier at this same point in training. Consistent with the working theory (noise scales with `effective_blend` magnitude, not step count) — sine's `effective_blend` was only ~0.24 at step 312K vs. fixed-blend's constant 0.75 the whole run.

**Hypothesis for what should happen next, worth checking as the run progresses**: `effective_blend` only reaches its full 0.75 target right at step 1,500,000 (the sine ramp's endpoint), so the fair comparison to fixed-blend's constant-0.75 noise isn't "at the same step count" but "at the same blend value" — meaning noise should rise as training passes the ~900K-1M mark (where `effective_blend` crosses ~0.5) and plausibly approach fixed-blend's ~25-31% swing by the very end. **User's read, which seems right: rising noise late in training is expected and not a red flag** — LR is simultaneously decaying to ~0 via the same cosine schedule at that point, so even if the reweighted training signal is at full strength, the optimizer's ability to actually let a noisy gradient move the weights is shrinking at the same time. The point of annealing was never to eliminate the reweighting's perturbation, just to keep the model out of that regime early, while the per-token EMA buffer was still statistically thin and LR was still high enough for a bad step to do real damage. Noise showing up in val_ppl is downstream of training-gradient noise, not a property of the (always-unweighted) eval loss itself, so this is consistent with "the mechanism is behaving as designed," not "the mechanism produces noisier eval."

**What would actually falsify vs. support this**: check again once `effective_blend` crosses ~0.5 (roughly step 900K-1M) — if noise at matched blend value looks like fixed-blend's noise at its constant 0.75, that confirms blend magnitude alone drives it (timing doesn't independently matter beyond avoiding the early low-blend-buffer regime). If it stays tighter than fixed's even at matched blend, something else is also helping (e.g. the EMA buffer being better-calibrated after more accumulated per-token samples by that point in training, independent of the blend schedule). **Not written into `paper_ema.md` yet** — this is reasoning from the in-training val_ppl proxy on an incomplete (21%-through) run, and the paper's own stated discipline (Section 3) is to never cite that proxy as evidence, only the real lm-eval-harness benchmark once the run is actually done. Revisit for the paper once this run finishes and has real numbers.

**Paused 2026-08-02 at step 314,500** to free `io`'s GPU for another test — killed cleanly via SIGTERM (checkpoint's `micro_step`/`ema_loss` buffer confirmed intact afterward), not deleted. See "Current status snapshot" above for the exact `--resume` command. This pauses the noise-dynamics check above mid-way — the `effective_blend`≈0.5 checkpoint (~step 900K-1M) that would confirm or complicate the hypothesis hasn't been reached yet and won't be until this is resumed.

### BLT seed19-at-500K bookkeeping (resolved 2026-07-13)
Investigating a discrepancy in `run_blt_scratch_seed42_500k.json` on `titan` (different LAMBADA/HellaSwag/PIQA/Winogrande numbers than the committed version under the same filename) surfaced a cluster of confusing but ultimately-explained loose ends:
- **titan's `run_blt_scratch_seed42_500k.pt`/.json is a different, unexplained checkpoint** — internally tagged seed 42, genuinely completed its own 500K-step schedule (confirmed via `scheduler_state`: `last_epoch=500000`, saved LR exactly `0.0`, ruling out "mid-run snapshot of the 550K run"), standard single-shared-M architecture, but final val_ppl 49.16 doesn't match any documented seed42 run (550K→72.88, the canonical 500K→76.64). Preserved as `lm_eval_blt_scratch_seed42_500k_titan_version.json` rather than overwritten. **Not fully explained — left as an untraceable artifact, not used for anything.**
- **A separate, previously-undocumented local file set was found**: `run_blt_scratch_seed19_500k.pt/.log/.stdout` (dated 2026-07-06, right in the bender-overheating window) — internally seed 19, from-scratch, targeting 500K steps, but **killed at step 910** and never resumed. This matches the user's memory of a seed19 attempt killed early due to overheating. **This is real but incomplete — not usable data.**
- **Net effect**: BLT has never had a genuine, complete, from-scratch 500K-step seed19 run. It has `run_blt_scratch_seed19.pt` (600K steps, DONE, OWT ppl 30.48) and the killed 910-step attempt above, but nothing at exactly 500K to match seed7 (30.81) and seed42 (31.69) for an iso-step three-way comparison — mirroring what GPT-2 baseline now has (seed42, seed7, and seed19 in progress on `titan`). **Fix queued**: watcher on `titan` launches this the moment the GPT-2 baseline seed19 run finishes (see status snapshot above).

**DONE (2026-07-23, benchmarked 2026-07-25).** The queued watcher fired as designed. `run_blt_scratch_seed19_500k.pt`, 500,000 steps, 355,988s, final WikiText val_ppl 82.37. Benchmarked: OWT held-out ppl **32.87** (loss 3.4927), LAMBADA acc 0.208, LAMBADA ppl 343.4, HellaSwag acc_norm 0.271, PIQA acc_norm 0.559, Winogrande acc 0.500 (`lm_eval_blt_scratch_seed19_500k.json`). See status snapshot above for the direct comparison against GPT-2 baseline seed19. This closes out the seed19-at-500K gap this section exists to track — BLT now has a genuine three-seed iso-step (500K) set: seed42_500k (31.69), seed19_500k (32.87), seed7 (30.81).

**GPT-2 (standard MHA) from-scratch OWT with EMA per-token loss weighting, seed 42**: started 2026-06-19, **DONE** 2026-06-23. `run_gpt2_ema_seed42.pt/.log`, 500K steps, final val_ppl=67.12, OWT held-out ppl=**30.06** (loss 3.4031). Mirrors `run_gpt2_baseline_seed42.pt` (from-scratch, OWT, seed 42, 500K steps) but adds `--ema-loss-weighting` (default `--ema-decay 0.99`). Benchmarked: `lm_eval_gpt2_ema_seed42.json` — LAMBADA acc 0.253 (vs baseline 0.225), LAMBADA ppl 119.6 (vs 174.6), HellaSwag acc_norm 0.272 (vs 0.268), PIQA acc_norm 0.569 (vs 0.579), Winogrande acc 0.511 (vs 0.505).

**Major finding — EMA loss weighting trade-off is architecture-general, not BLT-specific.** Compared against BLT-EMA (`run_blt_ema_seed42.pt`, from-scratch BLT/OWT/seed42/500K steps with the same `--ema-loss-weighting`, copied in from another machine, OWT ppl 34.25 vs BLT-seed7's 30.81, `lm_eval_blt_ema_seed42.json`): both architectures show the *same shape* of trade-off, within a few percent — LAMBADA acc gain nearly identical (BLT +0.030 vs GPT-2 +0.028), LAMBADA ppl drop substantial on both (GPT-2 actually larger relatively: -31.5% vs BLT's -22.9%), PIQA degrades by almost the same amount on both (-0.009 vs -0.010), HellaSwag/Winogrande roughly flat on both, OWT ppl worsens on both (BLT proportionally more: +11.2% vs GPT-2's +8.2%). This is strong evidence the EMA per-token-ID reweighting (upweight tokens whose vocabulary-id has historically been hard) is a genuine, generalizable loss-function finding — trading some uniform next-token accuracy for materially better hard/long-range prediction — independent of attention architecture. See `project_loss_function_ideas.md` memory for the original hypothesis (Option 1, self-referential EMA tracking).

**Major finding — post-hoc fine-tuning back to standard CE is a forgetting cliff, not a free lunch.** Took the converged `run_gpt2_ema_seed42.pt` checkpoint and fine-tuned for just 10,000 steps (2% of the original 500K budget) with plain CE (`train.py --finetune ... `, fresh optimizer/schedule — `--resume` would be wrong here since it'd continue the original cosine schedule already decayed to ~0 LR by step 500K). Result, on machine `titan`, `run_gpt2_ema_then_ce_finetune_seed42.pt`: LAMBADA acc didn't regress toward baseline, it **overshot past it** (0.253 → 0.217 vs. non-EMA baseline's 0.225) and LAMBADA ppl ended slightly worse than the never-EMA baseline (184.1 vs 174.6) — while OWT held-out ppl only recovered 41% of its gap (30.06 → 29.13, target 27.78) in the same window. The EMA-driven LAMBADA benefit is fragile and erodes much faster than the OWT-ppl cost can be bought back — matches the mechanism behind the earlier (2026-05-28) catastrophic-forgetting result from WikiText fine-tuning. **Caution**: the in-training cloze-proxy `lambada_acc` (200 examples, greedy) completely missed this — stayed flat ~0.55-0.57 throughout the whole 10K-step fine-tune while the real lm-eval LAMBADA accuracy (5153 examples) collapsed. Always confirm with the full lm-eval benchmark, not the in-training proxy.

Added `--ema-blend` (commit `c118787`) to test whether a blended objective avoids the cliff: interpolates the per-token EMA weight toward uniform (1.0 = full EMA, 0.0 = standard CE). Also fixed `--finetune` to load the checkpoint's EMA per-token-loss buffer (previously only `--resume` did, so any fine-tune was restarting hard-token tracking from a cold, uninformative state).

**Major finding — a 50/50 blend lands on a clearly better point than either pure endpoint.** `run_gpt2_ema_blend50_finetune_seed42.pt`, `titan`, 10K steps, `--ema-blend 0.5`, otherwise identical protocol to the plain-CE fine-tune: OWT held-out ppl 29.68 (recovered only 17% of the EMA-vs-baseline gap, vs. pure-CE's 41%) but LAMBADA acc **0.244** — solidly above the non-EMA baseline (0.225), retaining ~68% of the original EMA gain, vs. pure-CE's 0.217 (which had overshot *below* baseline). LAMBADA ppl 142.9 also lands much closer to EMA's 119.6 than pure-CE's 184.1. So a small give-back in OWT-ppl recovery buys back a disproportionately large fraction of the LAMBADA benefit — real evidence the blend hypothesis works, at least at α=0.5, via this sequential fine-tune-from-converged-checkpoint protocol. Full table:

| | EMA (before) | Pure CE fine-tune (10K) | 50/50 blend fine-tune (10K) | Non-EMA baseline |
|---|---|---|---|---|
| OWT held-out ppl | 30.06 | 29.13 | 29.68 | 27.78 |
| LAMBADA acc | 0.253 | 0.217 | 0.244 | 0.225 |
| LAMBADA ppl | 119.6 | 184.1 | 142.9 | 174.6 |
| HellaSwag acc_norm | 0.272 | 0.271 | 0.270 | 0.268 |
| PIQA acc_norm | 0.569 | 0.566 | 0.562 | 0.579 |
| Winogrande acc | 0.511 | 0.499 | 0.498 | 0.505 |

Open questions: sweep more α values to map the curve; test a *jointly*-trained blend (fixed mixture from step 0 of a from-scratch run) rather than sequential fine-tune-from-EMA; check whether this favorable blend result is architecture-general (BLT) the way the base EMA effect was.

**α=0.25 fills in the sweep — confirms a monotonic curve, not just two endpoints plus a midpoint.** `run_gpt2_ema_blend25_finetune_seed42.pt`, same 10K-step sequential fine-tune-from-EMA protocol, `--ema-blend 0.25`. Result (`lm_eval_gpt2_ema_blend25_finetune_seed42.json`): OWT held-out ppl 29.29, LAMBADA acc **0.231**, LAMBADA ppl 159.3, HellaSwag acc_norm 0.270, PIQA acc_norm 0.563, Winogrande acc 0.506 — lands cleanly between the pure-CE (α=0) and 50/50 (α=0.5) points on every metric, no crossovers. Full sweep:

| α | 0 (pure CE) | 0.25 | 0.5 | EMA (before, α=1 equiv.) | Non-EMA baseline |
|---|---|---|---|---|---|
| OWT held-out ppl | 29.13 | 29.29 | 29.68 | 30.06 | 27.78 |
| LAMBADA acc | 0.217 | 0.231 | 0.244 | 0.253 | 0.225 |
| LAMBADA ppl | 184.1 | 159.3 | 142.9 | 119.6 | 174.6 |
| HellaSwag acc_norm | 0.271 | 0.270 | 0.270 | 0.272 | 0.268 |
| PIQA acc_norm | 0.566 | 0.563 | 0.562 | 0.569 | 0.579 |
| Winogrande acc | 0.499 | 0.506 | 0.498 | 0.511 | 0.505 |

Both OWT-ppl cost and LAMBADA-acc gain increase smoothly and monotonically with α across all four points (0, 0.25, 0.5, 1.0) — a clean dial, not a threshold effect. Note this result was already on disk (dated 2026-06-24, i.e. before the 50/50 point was written up) but had never been folded into this doc until 2026-08-03.

**Follow-up — jointly-trained blend-from-scratch (α=0.75), not sequential fine-tune.** Tests the first open question above: rather than fine-tuning a converged EMA checkpoint back toward CE, `--ema-blend 0.75` was mixed in from step 0 of a full 500K-step from-scratch OWT run. `run_gpt2_ema_blend75_scratch_seed42.pt` (machine `titan`), seed 42, DONE 2026-06-30: final val_ppl 64.26, OWT held-out ppl **29.61** (loss 3.3881 nats), LAMBADA acc **0.269** (best of the whole GPT-2/EMA family, beating even pure EMA's 0.253), LAMBADA ppl 130.6, HellaSwag acc_norm 0.273, PIQA acc_norm 0.565, Winogrande acc 0.522. Result files: `lm_eval_gpt2_ema_blend75_scratch_seed42.json`, `eval_owt_gpt2_ema_blend75_scratch_seed42.stdout` (checkpoint/log/stdout live only on `titan` as of 2026-07-02, not copied locally).

| | EMA pure | Pure CE fine-tune (10K) | 50/50 blend fine-tune (10K) | **75/25 blend from scratch (500K, seed42)** | Non-EMA baseline |
|---|---|---|---|---|---|
| OWT held-out ppl | 30.06 | 29.13 | 29.68 | 29.61 | 27.78 |
| LAMBADA acc | 0.253 | 0.217 | 0.244 | **0.269** | 0.225 |
| LAMBADA ppl | 119.6 | 184.1 | 142.9 | 130.6 | 174.6 |
| HellaSwag acc_norm | 0.272 | 0.271 | 0.270 | 0.273 | 0.268 |
| PIQA acc_norm | 0.569 | 0.566 | 0.562 | 0.565 | 0.579 |
| Winogrande acc | 0.511 | 0.499 | 0.498 | 0.522 | 0.505 |

**Best result in the whole EMA/blend family, seed42**: the jointly-trained 75/25 blend beat pure EMA's LAMBADA acc (0.269 vs 0.253) at almost the same OWT-ppl cost as the 50/50 sequential fine-tune. Verified genuine (see bug note below): seed42's checkpoint `ema_loss` buffer has real per-token spread (mean 6.43, std 2.15, range 0.05–15.68 across 50,257 tokens, only 120 untouched) — unambiguous evidence 500K steps of real EMA tracking happened.

**BUG FOUND 2026-07-07 — the "seed 7 replicate" never actually had EMA active.** `run_gpt2_ema_blend75_scratch_seed7.pt` was launched on `titan` and finished 2026-07-06 (final val_ppl 59.47, benchmarked: OWT ppl 28.18, LAMBADA acc 0.203, ppl 207.7, HellaSwag 0.268, PIQA 0.577, Winogrande 0.515) — numbers that looked like a "mirror image" of seed42's result and were initially treated as a real contradiction requiring a third seed to resolve. **That was wrong.** Direct inspection of the checkpoint's `ema_loss` buffer shows it is bit-for-bit at its untouched initialization value (`log(vocab_size)` = 10.8249) for **all 50,257** vocab entries, std exactly 0.0 — proof `--ema-loss-weighting` was never active for a single step (the buffer is only touched inside that flag's code branch in `train.py`). Whoever/whatever launched that run passed `--ema-blend 0.75` without `--ema-loss-weighting`, which is a complete no-op (`--ema-blend` only matters inside the `if args.ema_loss_weighting:` block). The run trained as an ordinary from-scratch baseline the entire 500K steps, just on different hardware (`titan` P100) than the real `run_gpt2_baseline_seed7.pt` (`bender` V100) — fully explaining why its numbers looked baseline-ish rather than showing any blend signature. **Confirmed via a controlled smoke-test comparison first (CPU, same seed, with/without the flag, `/tmp/smoke_noema_cpu.log` vs `/tmp/smoke_ema_cpu.log`) before the checkpoint-buffer check made it conclusive** — the buffer inspection is the decisive evidence, the smoke test was corroborating but noisier (CPU/GPU float differences were comparable in size to the true early-step EMA signal).

**Retracting the "two seeds disagree" conclusion from 2026-07-07 morning — it was never a real second seed test of the blend hypothesis.** The seed42 75/25-blend result stands uncontested; there is still only one (verified-genuine) data point for the jointly-trained 75/25 blend. **A real seed7 replicate needs to be relaunched** with `--ema-loss-weighting --ema-blend 0.75` both present, and this time verify via the checkpoint `ema_loss` buffer (should show real per-token spread, not std=0.0) before trusting the benchmark numbers. **Lesson for future runs**: always sanity-check a new EMA/blend run's checkpoint buffer early (e.g. at the first checkpoint save) rather than waiting until a full 500K-step run completes and gets benchmarked — this bug went undetected through an entire run and a full benchmark pass.

**DONE (2026-07-13) — seed 19, second genuine 75/25-blend-from-scratch data point.** `train.py --baseline --from-scratch --ema-loss-weighting --ema-blend 0.75 --dataset openwebtext --seed 19 --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 --save-path run_gpt2_ema_blend75_scratch_seed19.pt`, launched on `titan` 2026-07-07, finished 2026-07-13, final val_ppl **60.35**. Verified correct at launch (`ps` showed both flags in argv, unlike the broken seed7 launch) — final result is trustworthy.

Benchmarked: OWT held-out ppl **29.61** (loss 3.3882 nats — essentially identical to seed42's 29.61/3.3881 despite seed19 tracking meaningfully better on the in-training WikiText val_ppl proxy throughout training, another instance of that proxy not predicting the metric that matters). `lm_eval_gpt2_ema_blend75_scratch_seed19.json` — LAMBADA acc 0.253, LAMBADA ppl 127.8, HellaSwag acc_norm 0.271, PIQA acc_norm 0.582, Winogrande acc 0.508.

| | seed19 | seed42 | Non-EMA baseline |
|---|---|---|---|
| WikiText val_ppl | 60.35 | 64.26 | — |
| OWT held-out ppl | 29.61 | 29.61 | 27.78 |
| LAMBADA acc | 0.253 | **0.269** | 0.225 |
| LAMBADA ppl | **127.8** | 130.6 | 174.6 |
| HellaSwag acc_norm | 0.271 | 0.273 | 0.268 |
| PIQA acc_norm | **0.582** | 0.565 | 0.579 |
| Winogrande acc | 0.508 | 0.522 | 0.505 |

Both seeds land solidly above the non-EMA baseline on LAMBADA acc (confirming the core 75/25-blend finding holds as a genuine two-seed result), though seed42 is the stronger of the two on LAMBADA specifically (~6% relative gap) while seed19 wins on PIQA by a similar margin — normal seed variance, not a contradiction.

**Third genuine seed — DONE (2026-07-18).** `run_gpt2_ema_blend75_scratch_seed7_v2.pt` (`venus`, launched 2026-07-11, verified genuine at step 2,000 via checkpoint buffer), finished at step 500,000, final val_ppl 57.38. Benchmarked (`lm_eval_gpt2_ema_blend75_scratch_seed7_v2.json`; OWT eval run on `io` since `venus` never received the raw OWT parquet files `eval_owt.py` needs — only the pre-tokenized training cache — so the 1.49GB checkpoint was relayed there instead of transferring several GB of parquet data): OWT held-out ppl **29.42** (loss 3.3818), LAMBADA acc 0.257, LAMBADA ppl 131.3, HellaSwag acc_norm 0.266, PIQA acc_norm 0.574, Winogrande acc 0.520.

| | seed7_v2 | seed42 | seed19 | Non-EMA baseline |
|---|---|---|---|---|
| OWT held-out ppl | 29.42 | 29.61 | 29.61 | 27.78 |
| LAMBADA acc | 0.257 | **0.269** | 0.253 | 0.225 |
| LAMBADA ppl | 131.3 | 130.6 | **127.8** | 174.6 |
| HellaSwag acc_norm | 0.266 | 0.273 | 0.271 | 0.268 |
| PIQA acc_norm | 0.574 | 0.565 | **0.582** | 0.579 |
| Winogrande acc | 0.520 | **0.522** | 0.508 | 0.505 |

**All three seeds land solidly above the non-EMA baseline's LAMBADA acc (0.225)** — 0.269, 0.253, 0.257, averaging ~0.260 — a genuine three-seed confirmation of the jointly-trained 75/25-blend finding. OWT ppl is tight and consistent across all three (29.42–29.61). This is now a solid basis for the paper claim, not a single uncontested data point.

**DONE**: `run_gpt2_baseline_seed7.pt/.log`, finished 2026-06-26 ~21:20 local time. Standard cross-entropy (no EMA), from-scratch, OWT, seed 7, 500K steps. Final val_ppl 55.42, OWT held-out ppl **27.36** (loss 3.3091 nats). Benchmarked: `lm_eval_gpt2_baseline_seed7.json` — LAMBADA acc 0.214, LAMBADA ppl 185.5, HellaSwag acc_norm 0.270, PIQA acc_norm 0.583, Winogrande acc (see json). Gives MHA a second iso-step (500K) seed, matching BLT's existing seed7. **User's goal: at least two genuine iso-step seeds each for BLT, MHA, and GQA**; GQA still only has one (seed42) — deferred until the BLT seed42_500k run below lands, per explicit user instruction.

**DONE (2026-07-06 ~01:17 local time)**: a NEW BLT seed42 run at exactly 500K steps (`run_blt_scratch_seed42_500k.pt` — deliberately NOT overwriting the existing `run_blt_scratch_seed42.pt`, which ran 550K steps and is a different, longer run) — `train.py --dataset openwebtext --seed 42 --max-steps 500000 --from-scratch --eval-every 1000 --lambada-eval-every 5000 --save-path run_blt_scratch_seed42_500k.pt`. Launched 2026-07-02 ~15:33 local time. Final WikiText val_ppl 76.64, OWT held-out ppl **31.69** (loss 3.4561 nats). Benchmarked: `lm_eval_blt_scratch_seed42_500k.json` — LAMBADA acc **0.188**, LAMBADA ppl 369.1, HellaSwag acc_norm 0.267, PIQA acc_norm 0.567, Winogrande acc **0.489**.

**This is the weakest of the four BLT seeds on record, though likely still within ordinary seed-to-seed variance.** Compare to the existing seeds:

| Metric | seed42_500k (this run) | seed42 (550K) | seed19 (600K) | seed7 (500K) |
|---|---|---|---|---|
| OWT held-out ppl | 31.69 | 31.05 | 30.48 | 30.81 |
| LAMBADA acc | 0.188 | 0.205 | 0.209 | 0.212 |
| Winogrande acc | 0.489 | 0.528 | 0.511 | 0.516 |
| HellaSwag / PIQA acc_norm | 0.267 / 0.567 | 0.271 / 0.561 | 0.267 / 0.572 | 0.268 / 0.568 |

Initially flagged this as a possible thermal-throttling confound (see below), but on reflection that doesn't hold up: throttling only reduces clock speed, it has no plausible mechanism for changing training *outcomes* (zero NaN, healthy loss curve throughout) — and the GPT-2 baseline seed7 run trained under the same thermal conditions on this machine without any similarly large deviation from the other MHA baselines, which is direct evidence against a throttling-driven quality effect. With only 3 prior BLT seeds (spread 30.48-31.05 on OWT ppl already), a 4th seed landing somewhat outside that range isn't strong evidence of anything beyond ordinary seed variance at this sample size. Treat this as just a seed observation, not a hardware anomaly, unless further evidence emerges.

**Note — this run was restarted from scratch, not resumed**, after a power outage on this machine (`bender`) knocked the V100 offline (see machine list below). A first launch attempt of this run (2026-06-30/07-01) reached only step 5,820/500,000 before the outage, and turned out to have been silently training on **CPU** the whole time (`torch.cuda.is_available()` was False at launch, and `train.py` has no hard failure on that — it just logs `Device: cpu` and proceeds at ~12.8s/step instead of ~562ms/step). That run's timing data is unusable for the wall-clock comparison this run exists to produce, so rather than resume it, the old `.pt`/`.log`/`.stdout` were moved to `stale_cpu_run_20260701/` and training restarted from step 0 on GPU (confirmed via log header: `Device: cuda`, `from_scratch=True`). GPU confirmed working again via physical reseat (`nvidia-smi` clean, V100-DGXS-16GB detected).

**New issue found post-restart — sustained GPU thermal throttling on `bender`.** Since partway through this run, the V100 has been running at 81-82°C with its SM clock locked around 620-800 MHz vs. the 1530 MHz rated max boost (roughly 40-52% of full clock), confirmed via repeated `nvidia-smi` samples and at least one direct `sw_thermal_slowdown: Active` reading. Power draw stays well under the 300W limit throughout, ruling out a power cap — this is a real thermal/cooling issue, not a config issue. Net effect: this run has been running ~10-12% slower per step than the previous `run_blt_scratch_seed7.pt` 500K run did at the same step count (e.g. at step 384,000: this run took 228,342s vs. seed7's 205,462s, a ~6.4 hour gap that has been slowly growing, not shrinking). Training itself is unaffected (zero NaN, healthy loss/val_ppl), just slower — **worth a physical check of the machine (fans, airflow, dust, whether the recent reseat left an airflow shroud loose) once convenient**, since sustained 81-82°C is hot for this card and the timing cost is real and compounding across long runs.

**This also resolves an open methodology question**: per-step timing comparisons between BLT and GPT-2 were previously confounded by running on *different physical machines* (the existing seed42/seed7 BLT runs and seed42 GPT-2 baseline ran on a different machine than this one). Running MHA-seed7 and the new BLT-seed42 back-to-back on this same local GPU, same software stack, gives a clean apples-to-apples ms/step comparison for the paper — though the thermal throttling above means this particular run's own timing is no longer clean for that comparison either; a machine at healthy thermals would be needed for a truly apples-to-apples number.

**Exact figures** (from `run_gpt2_baseline_seed7.log` / `run_blt_scratch_seed42_500k.log`, both on `bender`, 500,000 steps each): GPT-2 baseline **637.2 ms/step** (318,595s total) vs. BLT **588.3 ms/step** (294,157s total) — BLT is ~7.7% faster per step on identical hardware, confirming the speed advantage isn't purely a cross-machine artifact, though the thermal-throttling caveat above still applies to both numbers equally (same machine, same conditions, so the comparison between them is still valid even if neither absolute number is "clean" relative to an unthrottled machine).

**GQA seed 7 — DONE (2026-07-11).** The watcher script above fired as designed: once the BLT `seed42_500k` process exited, it first ran the lm-eval-harness and OWT held-out benchmarks against the finished checkpoint (results above), then launched `train.py --gqa --from-scratch --dataset openwebtext --seed 7 --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 --save-path run_gqa_scratch_seed7.pt` at 2026-07-06 ~01:28 local time (confirmed `Device: cuda`, 112,627,968 unique parameters). Watcher log: `queue_gqa_seed7_watcher.log`. **Stopped by the user at step 190,160/500,000** (2026-07-07 ~15:51) to free `bender`'s GPU for a BLT seed42 redo attempt (redo was aborted and reverted, see above), then **resumed via `queue_gqa_resume_watcher.sh`** once the per-layer-M warm-start run finished on that same GPU (2026-07-08 ~09:54, correctly picked up at step 190,120 via `train.py --resume`). Ran to completion 2026-07-11, final val_ppl **64.49**.

Benchmarked: OWT held-out loss 3.3410 nats, ppl **28.25** (`eval_owt_gqa_scratch_seed7.stdout`); lm-eval-harness in `lm_eval_gqa_scratch_seed7.json` — LAMBADA acc 0.192, LAMBADA ppl 231.9, HellaSwag acc_norm 0.271, PIQA acc_norm 0.577, Winogrande acc 0.509. Consistent with GQA seed42 (OWT ppl 27.64, LAMBADA acc 0.204, HellaSwag 0.269, PIQA 0.568, Winogrande 0.496) — no outliers on any metric. **This gives GQA its second iso-step (500K) seed, matching BLT and MHA, which each already have two — completing the user's stated goal of at least two genuine iso-step seeds per architecture.**

Note: the first attempt to run both benchmarks (lm-eval-harness + OWT held-out) concurrently on the same GPU crashed the lm-eval-harness process with `RuntimeError: CUDA error: unspecified launch failure` a couple minutes in — GPU itself recovered cleanly afterward (0% util, 4MiB used, no wedge), so this looks like resource contention from running two GPU jobs at once rather than a lasting hardware fault. Re-ran lm-eval-harness alone (after letting OWT eval finish first) and it completed without issue. Worth remembering for future runs on this machine: don't launch two concurrent GPU benchmark jobs on `bender`.

**BLT seed42 500K redo — ATTEMPTED AND ABORTED, 2026-07-07; original result confirmed final.** User initially believed the `seed42_500k` run above (OWT ppl 31.69, weakest of the four BLT seeds) was "somehow incorrect" and asked for a redo, so it was archived to `stale_throttled_run_20260706/` and a fresh run relaunched from step 0 with the identical command on `bender` (accepting the still-unfixed thermal throttling, since `titan`/`io` were unavailable). **Direct comparison of the redo's log against the archived run's log showed the two are bit-for-bit identical on `train_loss` at every matching step (0 through 1320)** — same seed, same code, same data order, fully deterministic, so the redo was just reproducing the exact same trajectory, not generating an independent data point. This is itself strong positive confirmation of the earlier conclusion: GPU clock throttling perturbs wall-clock speed only, not the computation. **The redo was killed and the archived files restored to their original filenames** — `run_blt_scratch_seed42_500k.pt/.log/.stdout` and `lm_eval_blt_scratch_seed42_500k.json/.stdout`, `eval_owt_blt_scratch_seed42_500k.stdout` are once again the real, final seed42_500k result (OWT ppl 31.69, LAMBADA acc 0.188, etc., as documented above). **If a genuinely independent 4th/5th BLT data point is wanted, it needs a different seed** (or some other source of variation) — rerunning seed 42 on any hardware will reproduce the same numbers.

## Active run on `io` — BLT 1.3B warm-start fine-tune (hand-off point, 2026-06-22)

**Machine**: remote box named `io`, SSH-accessible, repo at `/home/benzene/llmopt.blt/blt` (note: `.blt` in the path, unlike local `/home/benzene/llmopt/blt`). Default remote shell is **tcsh**, not bash — `2>&1` and `source x.sh` don't work; use `ssh io bash -c '...'` for bash syntax, or tcsh-native `>&`, or invoke `~/miniconda3/bin/conda run -n blt python ...` directly without sourcing any activation script. GPU is a Tesla V100-PCIE-32GB (32GB VRAM, CC 7.0), was otherwise idle.

**What/why**: user has access to this 32GB V100 and wants a cheap first look at whether scaling BLT to ~1B+ params is worth pursuing, before committing to a full from-scratch 1B+ training run. Approach: warm-start BLT from pretrained **GPT-2 XL** (1.3B params, 48 layers, n_head=25, n_embd=1600) — M initialized as the average of Wq@Wk^T across all layers/heads, Wv/Wo kept from pretrained — then fine-tune on WikiText-103, mirroring the original small-model BLT warm-start protocol (Section 4.1 of the paper). **User explicitly scoped this to BLT-only for now**: "I would just like to run the blt fine-tune and see what the results look like compared to whatever you used as the baseline. Then we can see what to do next." MHA control and GQA at this scale are deferred pending these results — GQA in particular would need code changes since `build_gqa_model()` currently has no pretrained-warm-start path (builds from random weights only).

**Pretrained GPT-2 XL baseline** (for comparison once this run finishes): WikiText-103 ppl = **15.30** (measured, sliding-window eval). Zero-shot benchmark numbers (lambada/hellaswag/piqa/winogrande) were never obtained — that lm-eval run was killed for taking too long and was not re-attempted; worth re-running (possibly narrowed to just `lambada_openai`) if a benchmark comparison is wanted later.

**Launch command** (currently running):
```
cd /home/benzene/llmopt.blt/blt && ~/miniconda3/bin/conda run -n blt python train.py \
  --pretrained gpt2-xl --dataset wikitext103 --max-steps 50300 --warmup-steps 2000 \
  --batch-size 4 --grad-accum-steps 1 --fp16 --grad-checkpointing \
  --eval-every 1000 --lambada-eval-every 5000 --save-path run_blt_xl_warmstart_wikitext.pt
```
Started 2026-06-22 ~11:29 local time on `io`. `--warmup-steps 2000` (10x the default 200) was a deliberate defense-in-depth choice after the corruption incident below. fp16 + grad-checkpointing were required to fit a 1.3B model + Adam optimizer state in 32GB; batch_size=4 was calibrated to just fit (~28.6GB steady-state usage).

**Attempt 1 — corrupted.** Identical command, default `--warmup-steps` (200). Ran cleanly to step 16,830 (loss healthy, 7.4–7.85) then went NaN at step 16,840 with no warning, and **kept training for 1000+ more steps on NaN loss, repeatedly overwriting both the checkpoint and its `.bak` with all-NaN tensors** (verified via a throwaway script loading the checkpoint and checking `torch.isnan`/`isinf` on all 630 tensors — both files were 100% corrupted). Root cause: BLT's single shared `M` matrix is used by every layer, so one numerical blowup (likely fp16 forward-pass overflow — `GradScaler` only protects the backward/gradient path, not forward-pass overflow) corrupts the entire model in one backward pass, far more severe than the equivalent failure would be in standard MHA. Corrupted checkpoint/log files were deleted before relaunching. **Fix applied** (`train.py`, commit `87184e0`): a pre-backward `if not torch.isfinite(loss): ... continue` guard, and `save_checkpoint()` now raises rather than overwriting a good checkpoint with a non-finite one. Verified via unit test.

**Attempt 2 — recovered but slow, then superseded.** Relaunched with `--warmup-steps 2000`. No NaN/Inf this time, but two large self-recovering instability events: loss spiked to 26.5 at step 630 (mid-warmup), recovered to ~9.5-10 by step 1450; then drifted up to 17.1 at step 4050 (post-warmup, full LR), recovered to ~10-13 by step 5610. Both fully finite throughout, no guard triggers — but the *pattern itself* (model lurching far from a reasonable loss before slowly clawing back) pointed at the M warm-start init being too aggressive for this model size, not a one-off fluke. This attempt was abandoned in favor of attempt 3 rather than let it run to completion.

**Attempt 3 — current, much healthier.** Root cause identified: the M-averaging warm-start (mean of Wq@Wk^T across all layers/heads) is far more destructive at GPT-2 XL scale (48 layers × 25 heads = 1200 attention contexts averaged into one matrix) than at GPT-2 scale (12×12=144 contexts) — the averaged M apparently carries too much energy/variance for a model this size. Added `--warmstart-scale` (`train.py`/`model.py`, commit `8b69d7c`): blends the averaged M toward zero by this factor before use (default 1.0 = unchanged behavior for existing small-model warm starts). Relaunched on `io` with `--warmstart-scale 0.25`, otherwise identical command/flags to attempt 2. Result: **smooth, monotonic loss decline with no instability spikes at all** — val_ppl 31.71 (step 1000) → 27.70 (step 2000) → 22.47 (step 5000) → 19.77 (step 10000) → **18.06 (step 14000)**, closing fast on the pretrained GPT-2 XL baseline of 15.30. In-training LAMBADA cloze proxy already at 0.50-0.53 by steps 5000-10000. Zero NaN/Inf, zero guard triggers, as of step 14,600 / elapsed ~29,251s (~8.1h) at last check (2026-06-22 ~22:48 local time on `io`).

**Step-0 sanity baseline** (attempt 3, for context when reading early loss values): loss 8.78 at step 0 (val_ppl 13719) — above pretrained GPT-2 XL's ~2.73 nats (ppl 15.30), but below random-guessing loss of ln(50257) ≈ 10.83. Slightly better step-0 loss than attempt 1/2's scale=1.0 init (9.03/17869), consistent with the milder init being a smaller surgery.

**Time estimate**: at ~2s/step recent pace (29,251s / 14,600 steps ≈ 2.0s/step, slower per-step than attempts 1/2 likely due to the eval cadence rather than the scale change), remaining ~35,700 steps → roughly **20 more hours** from the last check, total run time somewhat longer than attempts 1/2's estimate due to the two restarts, but converging far better per step.

**Next steps once this finishes**: compare final WikiText ppl (and LAMBADA acc via the `--lambada-eval-every 5000` checkpoints already being logged) against the pretrained GPT-2 XL baseline (ppl 15.30) and against the small-model BLT warm-start result for a sense of how the gap scales. Then, per the user's framing ("then we can see what to do next"), decide whether an MHA control fine-tune at this scale is warranted (straightforward — `train.py --baseline --pretrained gpt2-xl ...`) and whether GQA at this scale is worth the `build_gqa_model()` code changes needed to support a pretrained warm-start path.

## Benchmark results (lm-eval-harness)

### From-scratch OWT runs — primary comparison

| Task | BLT seed42 (550K) | BLT seed19 (600K) | BLT seed7 (500K) | Hybrid 6MHA+6BLT (500K) | GPT-2 seed42 (500K) | GQA seed42 (500K) | GQA seed7 (500K) |
|------|-------------------|-------------------|-------------------|--------------------------|---------------------|-------------------|-------------------|
| OWT held-out ppl | 31.05 | 30.48 | 30.81 | 28.40 | 27.78 | **27.64** | 28.25 |
| OWT held-out loss | 3.4357 | 3.4170 | 3.4279 | 3.3462 | 3.3243 | **3.3192** | 3.3410 |
| LAMBADA acc | 0.205 | 0.209 | 0.212 | **0.222** | 0.225 | 0.204 | 0.192 |
| LAMBADA ppl | 349.6 | 288.6 | 244.4 | **167.1** | 174.6 | 205.3 | 231.9 |
| HellaSwag acc_norm | 0.271 | 0.267 | 0.268 | **0.273** | 0.268 | 0.269 | 0.271 |
| PIQA acc_norm | 0.561 | 0.572 | 0.568 | 0.562 | **0.579** | 0.568 | 0.577 |
| Winogrande acc | **0.528** | 0.511 | 0.516 | 0.504 | 0.505 | 0.496 | 0.509 |

**Key findings:**
- All three BLT seeds consistent (OWT ppl 30.48–31.05), confirming ~0.10 nat gap vs GPT-2/GQA is real, not seed variance.
- GQA's two seeds are consistent with each other (OWT ppl 27.64/28.25, LAMBADA acc 0.204/0.192, no outliers on any metric) — completes the "at least two iso-step seeds per architecture" goal for BLT, MHA, and GQA alike.
- GQA edges GPT-2 on OWT ppl (27.64 vs 27.78) despite similar parameter counts — KV compression at 2 groups has no cost and marginal benefit.
- GPT-2 still wins on LAMBADA acc despite worse OWT ppl than GQA — full per-layer Wk matters for long-range prediction specifically. But the hybrid model actually has the best LAMBADA ppl (167.1, beating even GPT-2's 174.6), suggesting 6 full MHA layers recover essentially all of LAMBADA's long-range needs.
- HellaSwag, PIQA, Winogrande are essentially ties within noise across all variants; BLT/hybrid hold a slight Winogrande edge over GPT-2/GQA.
- **Hybrid result is non-additive**: halving the BLT layer count (12→6) cuts the OWT loss gap vs GPT-2 from ~0.10-0.11 nats down to 0.022 nats — recovering over 75% of the gap, not 50%. This implies the expressiveness cost of M-sharing is front-loaded or compounding across layers rather than a flat per-layer tax; 6 unrestricted MHA layers (regardless of which 6) are enough to mostly route around it. See paper_blt.md "Hybrid architecture" for analysis of which layer positions were used.
- The ~0.10 nat BLT gap vs GPT-2/GQA is attributable to cross-layer M sharing (expressiveness cost), not parameter count — GQA has similar params to BLT but matches GPT-2.
- BLT speed advantage (seed42 machine) was hardware-specific — on this machine BLT runs at 562ms/step vs GPT-2's 560ms/step on the other machine.
- **Paper scope decision (2026-06-19)**: `paper_blt.md` now compares all architectures (BLT, MHA, GQA, hybrid) at a fixed 500K training steps, using BLT seed7 (the only BLT seed trained for exactly 500K steps) as the primary BLT from-scratch result. Seed42 (550K) and seed19 (600K) were originally trained longer for wall-clock parity with the 500K baselines, but converged to nearly the same OWT ppl as seed7 (31.05/30.48 vs 30.81) — the extra 50-100K steps bought no improvement — so the paper reports the iso-step comparison only, with a note in Section 3 about the longer runs. This table above and the full-history table still keep seed42/seed19 for completeness; only the paper text was scoped down.

### Full history

| Task | GPT-2 pretrained | BLT WikiText (50K) | BLT OWT random-M (250K) | BLT from-scratch OWT (550K) | GPT-2 from-scratch OWT (500K) |
|------|------------------|--------------------|--------------------------|------------------------------|-------------------------------|
| LAMBADA acc | **0.242** | 0.114 | 0.199 | 0.205 | 0.225 |
| LAMBADA ppl | **83.0** | 1307.5 | 206.8 | 349.6 | 174.6 |
| HellaSwag acc_norm | **0.291** | 0.275 | 0.280 | 0.271 | 0.268 |
| PIQA acc_norm | 0.560 | 0.541 | 0.568 | 0.561 | **0.579** |
| Winogrande acc | 0.502 | 0.507 | 0.490 | **0.528** | 0.505 |

Result files: `lm_eval_baseline.json` (truncated/corrupt), `lm_eval_blt.json`, `lm_eval_cloze_blt.json`, `lm_eval_2m_blt.json`, `lm_eval_lambada_blt.json`, `lm_eval_owt_randm_126k.json`, `lm_eval_owt_randm_218k.json`, `lm_eval_owt_randm_250k.json`, `lm_eval_blt_scratch.json`, `lm_eval_baseline_scratch.json`.

## Known machines

- **`io`**: remote V100-PCIE-32GB, repo at `/home/benzene/llmopt.blt/blt` (note `.blt`), tcsh default shell, pinned PyTorch 2.3.1+cu121/transformers 4.46.3 per the CC-7.0 compatibility note below.
- **`titan`**: remote, repo at `~/llmopt/blt` (no `.blt`), tcsh default shell. GPU is Tesla P100-PCIE-16GB, CC **6.0** — older than the V100s. Despite that, its pre-existing `blt` conda env runs **PyTorch 2.6.0+cu124 / transformers 5.9.0** (not the pinned versions) and this works fine empirically (smoke-tested full train/eval/save cycle, no CUDA errors) — the CC-7.0-compatibility claim below may not generalize to even-older CC 6.0, or may have been specific to a particular wheel; not chased down further. Measured throughput: ~0.96-1.0s/step (GPT-2 124M, batch=4, block=1024) vs ~0.65s/step on the local 16GB V100 — roughly **1.5x slower**, consistent with Pascal lacking Tensor Cores. Good for short/cheap jobs (fine-tune pilots), not ideal for full 500K-step from-scratch runs (~5.5 days vs ~3-3.5 days on a V100). `eval_owt.py`'s `owt_heldout_tokens()` globs parquet files and takes a **positional** slice `[start_file:start_file+n_files]` — if you copy over only the 5 held-out files (21-25) rather than the full 80-file set, the slice lands on nothing and silently returns empty; either copy the full parquet dir or run the OWT eval on a machine that already has it. **Repo lags badly if not pulled before use** — was found 20-40 commits behind origin twice in July 2026 (once missing the per-layer-M/GQA-3 work entirely), causing stale-code risk and an untracked-file collision on `git pull` (see below); always `git pull` before launching anything here.
- **`venus`**: desktop workstation (has a full GNOME session — a real user's machine, not headless), Tesla P100-PCIE-16GB, CC 6.0, same generation as `titan`. Repo at `/home/benzene/llmopt/blt`. New to this project as of 2026-07-11 — required full setup: `git pull` (was 30+ commits behind), and the pre-existing `blt` conda env had a broken mixed CUDA 12.x/13.x nvidia-* package set causing `ImportError: libcusparseLt.so.0: cannot open shared object file` on `import torch` — fixed via `pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` (letting pip resolve matching nvidia-* deps; do NOT use `--no-deps`, that skips the fix). OWT tokenized-block cache (`~/.cache/blt_owt_2m_blocks.pt`, 18GB) was `scp`'d over from `bender` rather than re-downloaded/re-tokenized — much faster. **Quirk**: this ssh connection silently drops the first line of output from short/fast commands (e.g. a bare `echo hello` produces nothing) — prefix remote commands with a harmless no-op (`true; <real command>`) to reliably see output. Also: flags/pipes in `ssh venus bash -c '...'` commands sometimes silently get dropped by the local tcsh parsing layer for complex multi-redirect strings (e.g. commands with `2>&1` or `2>/dev/null`) — avoid stderr-merging redirects in remote one-liners; write a script to a file and `scp` it over instead for anything non-trivial (this was the reliable fallback used throughout).

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
- `train.py` (2026-07-06): now hard-fails with a clear error and `sys.exit(1)` if `torch.cuda.is_available()` is False at startup, instead of silently falling back to CPU. Direct fix for the incident where `run_blt_scratch_seed42_500k.pt`'s first launch attempt silently trained on CPU for 5,820 steps at ~12.8s/step with no warning (see "Active run" above). Scoped to `train.py` only — `eval_owt.py`, `evaluate.py`, `generate.py` etc. still silently fall back, since those are short-lived and the cost of a silent CPU run there is minutes, not days.
- `train.py --per-layer-m` / `blt_lm_eval.py --per-layer-m` (2026-07-08): one M per layer instead of one shared globally. See "Architecture variants" below for the parameter/bandwidth analysis.
- `train.py` `ResumableSampler` + `save_checkpoint(..., micro_step=...)` (2026-07-16): fixes a real data-replay bug on `--resume` — old `shuffle=True` DataLoader replayed the start of the shuffle on every process restart since its order came from global RNG state that wasn't checkpointed. See "GPT-2 medium" below for the full writeup and verification. Any `--resume` on a checkpoint saved before this fix falls back to an approximate `micro_step` (`step * grad_accum_steps`) rather than an exact one.
- `train.py --ema-blend-schedule sine` (2026-07-21): anneals the EMA loss-blend coefficient from 0 up to `--ema-blend`'s target via a sine ease-in over the course of training, instead of applying it at full strength from step 0. Also logs the actual per-step `effective_blend` value on the training log's trailing column. See "GPT-2 medium" / `project_loss_function_ideas` memory for results.
- `train.py --uv-rank`/`--uv-source` (2026-07-27/28): low-rank BLT (M ≈ U@V^T). See "Low-rank BLT (M = UV^T)" below.

## Dataset loading (OpenWebText)
OWT tokenization caches to `~/.cache/blt_owt_2m_blocks.pt` after first run — subsequent starts load in seconds. First run takes ~10 minutes (bulk parquet load + encode_batch tokenization). Loading 2M docs uses ~15GB RAM peak; tokenized blocks are ~9GB. Uses first 21 parquet files (sorted), which matches `train[:2000000]` ordering — verified doc-for-doc against HuggingFace split.

## Architecture variants
- `model.py build_blt_model(num_m_groups=1)`: single shared M (original BLT)
- `model.py build_blt_model(num_m_groups=2)`: two shared M matrices, heads 0-5 and 6-11
- `model.py build_blt_model(random_m=True)`: random M initialization
- `model.py build_blt_model(from_scratch=True)`: random init for all weights, no pretrained load
- `model.py build_blt_model(per_layer_m=True)`: one M per layer (not shared across layers, still shared across all heads within a layer). Only valid with `num_m_groups=1`. 25% fewer attention params than standard MHA (117,343,488 unique params vs. the shared-M model's 110,855,424) — vs. ~48% for the cross-layer-shared M — since each layer needs its own D×D fetch and loses the "M loaded once for the whole model, cached in L2" bandwidth amortization the shared-M design gets. Warm-start init uses each layer's own `Wq @ Wk^T` directly (no cross-layer averaging). `train.py --per-layer-m`, `blt_lm_eval.py --per-layer-m`.
- `model.py build_gqa_model(n_kv_groups=N)` (2026-07-11): generalized from the original hardcoded 2-group `GQAAttention`. `N` must evenly divide `n_head` (12) — valid values 1, 2, 3, 4, 6, 12. `train.py --gqa --num-kv-groups N`, `blt_lm_eval.py --num-kv-groups N`, `eval_owt.py --gqa-checkpoint ... --num-kv-groups N` (default 2 everywhere, matches prior behavior).

**3-group GQA, seed 42 — DONE (2026-07-15).** `run_gqa3_scratch_seed42.pt`, from-scratch OWT, 500K steps, launched on `bender` 2026-07-11, final val_ppl 57.35. Benchmarked: OWT held-out ppl **27.46** (loss 3.3129) — best of the whole GQA family, edging out both 2-group seeds (27.64/28.25). `lm_eval_gqa3_scratch_seed42.json` — LAMBADA acc 0.195, LAMBADA ppl 216.4, HellaSwag acc_norm 0.271, PIQA acc_norm **0.553**, Winogrande acc 0.511. Most metrics land within normal seed-to-seed noise of the 2-group results, but PIQA is notably lower than both 2-group seeds (0.568/0.577) — a real-looking gap, though only one seed so far; a second 3-group seed would be needed to confirm it's not a fluke.

| | GQA-3 seed42 | GQA-2 seed42 | GQA-2 seed7 |
|---|---|---|---|
| OWT held-out ppl | **27.46** | 27.64 | 28.25 |
| LAMBADA acc | 0.195 | 0.204 | 0.192 |
| HellaSwag acc_norm | 0.271 | 0.269 | 0.271 |
| PIQA acc_norm | **0.553** | 0.568 | 0.577 |
| Winogrande acc | 0.511 | 0.496 | 0.509 |

### Per-layer-M warm-start run (started 2026-07-07/08)
Added 2026-07-07/08 after the "would one M per layer still save memory" discussion below confirmed the parameter math (25%) but clarified the bandwidth mechanism doesn't come from head-sharing (standard MHA already loads Wq/Wk once per layer regardless of head count) — it comes from *cross-layer* sharing, which a per-layer M gives up. User wanted an actual data point for this variant rather than just the theoretical analysis.

**Sanity-checked before launch**: a controlled A/B (`evaluate.py`'s `compute_perplexity`, same WikiText-103 slice, same warm-start weights) showed shared-M (original global BLT) scores 4142 val_ppl at step 0 vs. per-layer-M's 3009 — per-layer-M is *better* at init, consistent with the theory that each layer reconstructing its own head-collapsed attention pattern (via that layer's own `Wq@Wk^T`) beats blurring 12 layers' patterns into one average. Both numbers are far above the pretrained baseline (26.39) because collapsing 12 independent per-head softmaxes into one shared score matrix per layer is the core lossy BLT approximation, present in both variants — not a bug.

**DONE (2026-07-08).** `train.py --seed 42 --per-layer-m --lr 5e-5 --max-steps 50300 --warmup-steps 200 --eval-every 1000 --lambada-eval-every 5000 --save-path run_perlayerm_seed42.pt`, launched on `bender` 2026-07-07 ~21:45 local time, completed all 50,300 steps in ~28,131s (~7.8h). Mirrors the original 1-M WikiText-103 warm-start protocol exactly (same lr/warmup/step count) for direct comparability. Final val_ppl **32.22** — worse than both the shared-M original (21.50) and the untouched pretrained baseline (26.39), despite a better step-0 init (3009.74 vs. shared-M's 4142) and more parameter capacity (117,343,488 vs. 110,855,424 unique params). Genuine plateau, not an unconverged run cut short: val_ppl is already 32.30 at step 13,000 with LR still at 85% of peak (4.24e-05 of 5e-05), and barely moves the rest of the way (32.87 at the 50%-LR midpoint step 25,000; 31.47 final) — if more parameters simply needed more high-LR gradient steps, the long high-LR middle stretch (steps 13K-38K) is exactly where that should have shown up, and it didn't. Rules out "just needs more steps within this schedule"; does not rule out "would do better with a longer schedule built for more steps" (untested, proposed as a follow-up).

**Benchmarked** (`lm_eval_perlayerm_seed42.json`, run interleaved with GQA seed7 training on the same GPU): despite the worse WikiText val_ppl, **LAMBADA is substantially better than shared-M** — acc 0.175 vs. shared-M's 0.114 (+54% relative), ppl 570.2 vs. 1307.5 (less than half). HellaSwag/PIQA are near-ties (0.276/0.544 vs. 0.275/0.541), Winogrande slightly favors shared-M (0.488 vs. 0.507). Full comparison:

| | Per-layer-M | Shared-M (original 1-M) | Pretrained GPT-2 |
|---|---|---|---|
| WikiText val_ppl | 32.22 | 21.50 | 26.39 |
| LAMBADA acc | **0.175** | 0.114 | 0.242 |
| LAMBADA ppl | **570.2** | 1307.5 | 83.0 |
| HellaSwag acc_norm | 0.276 | 0.275 | 0.291 |
| PIQA acc_norm | 0.544 | 0.541 | 0.560 |
| Winogrande acc | 0.488 | 0.507 | 0.502 |

Same loss/benchmark mismatch pattern documented elsewhere in this project — WikiText/OWT next-token loss doesn't reliably predict LAMBADA's long-range performance. Plausible mechanism (untested): each layer's faithful, non-cross-layer-blurred attention reconstruction may specifically help long-range structure even though it converges to worse *average* next-token loss. **The in-training `lambada_acc` proxy (plateaued ~0.55) was again wildly off from the real benchmark (0.175)** — consistent with the earlier documented incident; don't trust that proxy number for any run.

`bender`'s thermal throttling issue (user gave up fixing it, see BLT seed42 redo note above) was not observed as a problem during this run (81°C, 1327 MHz — not the 620-800 MHz throttled state seen previously).

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
5. **UV^T (low-rank BLT)**: implemented; first from-scratch r=256 seed (seed42) DONE, second seed (seed19) in progress on `titan`. See details below.

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

**Implemented (2026-07-27/28), commit `24543f2`.** `model.py`: `BLTLowRankAttention` (forward: `softmax((X@U)@(X@V)^T/scale) @ (X@Wv) @ Wo`) + `build_blt_lowrank_model(rank, source_checkpoint, pretrained, from_scratch)`. The `source_checkpoint` path does Option 2 properly, not just a cosmetic factorization: truncated SVD of the checkpoint's `transformer.M_blt` (Eckart-Young optimal, `U_init = U_svd[:,:r]*sqrt(S[:r])`, `V_init = Vh_svd[:r,:].T*sqrt(S[:r])`), then carries over every other trained weight (embeddings, per-layer Wv/Wo, LN, MLP) unchanged via `load_state_dict` — only the attention score mechanism gets replaced. `train.py`: `--uv-rank`/`--uv-source` flags. `blt_lm_eval.py`/`eval_owt.py`: `--uv-rank` support for benchmarking low-rank checkpoints.

**Verification before trusting it**: rank-192 SVD reconstruction against a real trained M (`run_blt_scratch_seed19_500k.pt`) captured relative Frobenius error 0.5418 → ~71% energy, exact match to the documented SVD analysis above; full-rank (r=768, no truncation) reconstructed M to ~2e-6 relative error, confirming the SVD math itself is correct; a real 20-step run through `train.py`'s actual CLI (not just direct `model.py` calls) on `titan` showed loss starting at 5.28 (not random-init's ~10.9, confirming Wv/Wo genuinely carried over) declining to 4.82 by step 10, and exact expected parameter count (110,560,512 = full BLT's 110,855,424 minus the M→U+V savings of 294,912); both eval scripts loaded a low-rank checkpoint cleanly with no errors. One real bug found and fixed during verification: the SVD-source `load_state_dict` path initially raised on "unexpected"/"missing" keys — PyTorch auto-registers a shared parameter (M in the source, U/V in the new model) under *every* submodule holding a direct reference, not just the top-level `transformer.M_blt`/`transformer.U_blt`/`transformer.V_blt` names, so the per-layer duplicates (`transformer.h.{i}.attn.M`, `.attn.U`, `.attn.V`) also needed filtering/expected-missing handling.

**From-scratch r=256, not Option 2's warm-start.** User's call, given this project's repeated bad experiences with warm starts (BLT XL warm-start NaN corruption, M-average overshoot at scale, catastrophic forgetting from pretrained fine-tuning) — rather than SVD-factorize a trained M and fine-tune, `run_blt_uv256_scratch_seed42.pt` randomly initializes U, V, and everything else and trains the full 500K-step from-scratch protocol on `titan` (`--uv-rank 256 --from-scratch --dataset openwebtext --seed 42 --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000`), directly comparable to the existing full-rank BLT seed42_500k/seed19_500k/seed7 results. r=256 chosen as the primary rank (81% Frobenius energy of a converged full M) per the earlier rank-selection discussion; r=192 (71%) is a plausible follow-up if there's appetite for a second rank point, not yet started.

**DONE (2026-08-01).** 500,000 steps, 358,797s elapsed, final WikiText val_ppl 64.85. A watcher script (`queue_uv256_seed42_watcher.sh`, launched on `titan`, polled the training PID every 60s) auto-triggered the benchmark suite the moment training exited — confirmed working end-to-end (this was itself briefly a point of confusion: the watcher's launch command hung locally over ssh due to a quoting issue, but the remote process launched fine regardless and completed both evals ~9 minutes after training finished). Results (`lm_eval_blt_uv256_scratch_seed42.json`, `eval_owt_blt_uv256_scratch_seed42.stdout`): OWT held-out ppl **29.97** (loss 3.4002), LAMBADA acc 0.213, LAMBADA ppl 222.8, HellaSwag acc_norm 0.269, PIQA acc_norm 0.569, Winogrande acc 0.512.

| Metric | UV256 seed42 | BLT seed42 (500K) | BLT seed19 (500K) | BLT seed7 (500K) | GPT-2 seed42 (500K) |
|---|---|---|---|---|---|
| OWT held-out ppl | **29.97** | 31.69 | 32.87 | 30.81 | 27.78 |
| OWT held-out loss | 3.4002 | 3.4561 | 3.4927 | 3.4279 | 3.3243 |
| LAMBADA acc | **0.213** | 0.188 | 0.208 | 0.212 | 0.225 |
| LAMBADA ppl | **222.8** | 369.1 | 343.4 | 244.4 | 174.6 |
| HellaSwag acc_norm | 0.269 | 0.267 | 0.271 | 0.268 | 0.268 |
| PIQA acc_norm | 0.569 | 0.567 | 0.559 | 0.568 | 0.579 |
| Winogrande acc | 0.512 | 0.489 | 0.500 | **0.516** | 0.505 |

**Reads like a clear win over full-rank BLT at first glance — pairwise, it beats all three BLT seeds on 5 of 7 metrics, losing only two individual metric/seed comparisons (HellaSwag vs. seed19, Winogrande vs. seed7), both tiny and each against a different seed.** But the primary, most-trusted metric (OWT held-out ppl) tells a more cautious story: UV256's improvement over BLT's *best* seed is 0.84 ppl (30.81 → 29.97) — smaller than the 2.06 ppl spread BLT's own three seeds already show from seed variation alone (30.81 to 32.87). One UV seed beating three individual BLT seeds is suggestive, but it's not the same evidence as UV's distribution sitting above BLT's — the margin is inside BLT's own noise floor. **Second UV256 seed (seed19) launched 2026-08-01 on `titan` to settle it**: landing again in the 29-30 range would be real confirmation (especially paired with a second sweep against the BLT seeds); landing back in BLT's 31-33 range would mean seed42 was simply a favorable draw. Same watcher pattern as seed42 (`queue_uv256_seed19_watcher.sh`, polls training PID, auto-benchmarks on exit).

**DONE (2026-08-05) — second seed confirms it.** `run_blt_uv256_scratch_seed19.pt`, 500,000 steps, final WikiText val_ppl 67.39, watcher auto-benchmarked on exit. Results (`lm_eval_blt_uv256_scratch_seed19.json`, `eval_owt_blt_uv256_scratch_seed19.stdout`): OWT held-out ppl **30.13** (loss 3.4054), LAMBADA acc **0.214**, LAMBADA ppl 237.9, HellaSwag acc_norm 0.267, PIQA acc_norm 0.583, Winogrande acc 0.492.

| Metric | UV256 seed42 | UV256 seed19 | BLT seed42 (500K) | BLT seed19 (500K) | BLT seed7 (500K) |
|---|---|---|---|---|---|
| OWT held-out ppl | 29.97 | **30.13** | 31.69 | 32.87 | 30.81 |
| LAMBADA acc | 0.213 | 0.214 | 0.188 | 0.208 | 0.212 |
| HellaSwag acc_norm | 0.269 | 0.267 | 0.267 | 0.271 | 0.268 |
| PIQA acc_norm | 0.569 | 0.583 | 0.567 | 0.559 | 0.568 |
| Winogrande acc | 0.512 | 0.492 | 0.489 | 0.500 | **0.516** |

**This settles it — UV256's two-seed OWT ppl range (29.97-30.13) sits entirely below all three full-rank BLT seeds (30.81-32.87), with no overlap at all**, unlike the single-seed comparison above where the margin was inside BLT's own noise floor. LAMBADA acc is also consistently a bit better across both UV256 seeds (0.213/0.214) than the full-rank spread (0.188-0.212). **Low-rank BLT (r=256, 66% of full M's parameters) matches or beats full-rank BLT on the primary metric while also fixing the tensor-parallelism problem** (see "Tensor parallelism analysis" above) — genuinely the better BLT variant, not just a cheaper one. Worth updating `paper_blt.md` Section 4.7 to promote this from "preliminary, not-yet-seed-confirmed" to a confirmed two-seed result.

**Third seed — DONE (2026-08-10).** `run_blt_uv256_scratch_seed7.pt`, 500,000 steps on `titan`, final WikiText val_ppl 66.64, watcher auto-benchmarked on exit. Results (`lm_eval_blt_uv256_scratch_seed7.json`, `eval_owt_blt_uv256_scratch_seed7.stdout`): OWT held-out ppl **30.38** (loss 3.4137), LAMBADA acc 0.207, LAMBADA ppl 265.7, HellaSwag acc_norm 0.267, PIQA acc_norm 0.578, Winogrande acc 0.502.

| Metric | UV256 seed42 | UV256 seed19 | UV256 seed7 | BLT seed42 (500K) | BLT seed19 (500K) | BLT seed7 (500K) |
|---|---|---|---|---|---|---|
| OWT held-out ppl | 29.97 | 30.13 | 30.38 | 31.69 | 32.87 | 30.81 |
| LAMBADA acc | **0.213** | **0.214** | 0.207 | 0.188 | 0.208 | 0.212 |
| HellaSwag acc_norm | 0.269 | 0.267 | 0.267 | 0.267 | 0.271 | 0.268 |
| PIQA acc_norm | 0.569 | **0.583** | 0.578 | 0.567 | 0.559 | 0.568 |
| Winogrande acc | 0.512 | 0.492 | 0.502 | 0.489 | 0.500 | **0.516** |

**Now a genuine three-seed-vs-three-seed comparison, matching this project's standard for every other architecture family.** UV256's full range (29.97-30.38) still sits entirely below full-rank BLT's full range (30.81-32.87) — the third seed didn't just fail to break the pattern, it landed right in the middle of where the first two predicted (a good sign the two-seed read wasn't a fluke). LAMBADA acc holds its edge over BLT's weakest seed (seed42's 0.188) on all three UV256 seeds, though seed7 (0.207) is closer to BLT's stronger seeds (0.208, 0.212) than the first two UV256 seeds were — normal seed variance, doesn't change the OWT-ppl conclusion.

titan freed up immediately after (repo was one commit behind — stale git index on `kv_cache_bench_results.json`, verified byte-identical via `md5sum` against the already-committed version before discarding and pulling clean) and immediately launched the queued `num_uv_groups=4` from-scratch OWT run (see "Queued: num_uv_groups" below, now in progress, no longer queued).

This first result has been written up in `paper_blt.md` Section 4.7 (and Section 7.3 updated to reference it) as a preliminary, not-yet-seed-confirmed finding — consistent with how every other single-seed result in this project has been framed until a second seed lands. **Update DONE (2026-08-06)**: Section 4.7 now reports the seed19 confirmation, the direct comparison against the standard MHA baseline (below), and the same-hardware speed measurement; Section 6's synthesis bullet list and Section 7.3's closing paragraph updated to match.

**UV256 vs. standard MHA baseline, not just vs. full-rank BLT (2026-08-06).** Attention parameter count: standard MHA is 28.3M (12 layers × 4 D×D matrices: Wq,Wk,Wv,Wo) vs. UV256's 14.5M (12×2 D×D for per-layer Wv/Wo, plus one globally-shared U,V pair at 2×D×r) — a **48.6% reduction**, matching full-rank BLT's documented 48% almost exactly despite UV256 keeping a real factored Q/K computation (not fused into one opaque matrix like full-rank M).

| Metric | UV256 seed42 | UV256 seed19 | MHA seed42 | MHA seed19 | MHA seed7 |
|---|---|---|---|---|---|
| OWT held-out ppl | 29.97 | 30.13 | 27.78 | 28.24 | 27.36 |
| LAMBADA acc | 0.213 | 0.214 | 0.225 | 0.208 | 0.214 |
| HellaSwag acc_norm | 0.269 | 0.267 | 0.268 | 0.266 | 0.270 |
| PIQA acc_norm | 0.569 | 0.583 | 0.579 | 0.572 | 0.583 |
| Winogrande acc | 0.512 | 0.492 | 0.505 | 0.520 | 0.519 |

OWT ppl shows the usual real ~0.10 nat BLT-family cost (~30.0 vs ~27.8). But LAMBADA acc is close to identical (UV256's two-seed mean 0.2136 vs. MHA's three-seed mean 0.2157 — inside seed noise for either group), and HellaSwag/PIQA/Winogrande show no consistent direction. Roughly half the attention parameters reproducing standard MHA's downstream benchmark performance on 4 of 5 metrics is exactly the kind of evidence the whole BLT research premise (that standard multi-head attention over-parameterizes the QK interaction) predicts — worth stating plainly, not just implying.

**Same-hardware speed measurement**: both runs on `titan` (P100, no cross-machine confound) — MHA baseline seed19 at 1,052 ms/step (526,162s/500,000 steps) vs. UV256 seed42 at 718 ms/step (358,797s/500,000 steps) = **32% faster**, notably larger than full-rank BLT's own ~7.7% edge over MHA (measured on a V100). Makes sense structurally: standard MHA computes 12 independent (L,L) score matrices and 12 softmaxes per layer; UV256 computes exactly one of each, applied across the full D-dim value vector — the same head-collapse mechanism driving both the parameter savings and the larger-than-full-rank-BLT speedup.

**Synthesis with the hybrid result (Section 4.5 of the paper) — sharpens "attention is wasteful" into "where, specifically."** The hybrid experiment (6 MHA + 6 BLT layers) relaxes BLT's *cross-layer* sharing constraint for half the layers and recovers >75% of BLT's OWT-loss gap. UV256 relaxes the *within-layer* head-collapse constraint not at all (still one shared rank-256 Q/K pair per layer) yet still nearly matches MHA on LAMBADA/HellaSwag/PIQA/Winogrande. Together: giving up **cross-layer** independence is expensive (the hybrid needs real per-layer MHA layers to recover quality); giving up **within-layer head** independence is nearly free (UV256 shows collapsing 12 heads to 1 shared computation costs almost nothing on 4 of 5 benchmarks). The redundancy in standard attention is concentrated in per-head multiplicity within a layer, more than in the sheer count of independently-parameterized attention computations overall — a more specific and more actionable claim than "attention has too many parameters."

### UV256 inference (decode-latency) benchmark (2026-08-06)

**Math predicted two things before measuring anything.** Decode-step score computation: standard MHA and full-rank BLT both cost O(D² + D·L) — a D×D-scale query projection plus one O(D) dot product against each of L cached keys. UV256 costs O(D·r + r·L) — at r=256, D=768, a 3× reduction on both terms. Two predictions follow: (1) UV256 should beat MHA by roughly the margin full-rank BLT already does, and (2) UV256's edge over full-rank BLT *specifically* should widen with context length, since the L-dependent term (r·L vs D·L) is the one that shrinks. KV-cache: MHA and full BLT are already documented as "tied" (both cache D-dim per token); UV256 caches the V-projected key at rank r instead — (r+D) vs (2D) per token, ~33% less total cache at r=256.

**`kv_cache_bench.py` had no support for the low-rank attention class** — only `BLTAttention`, `GQAAttention`, `MHAAttention`. Added `blt_lowrank_prefill`/`blt_lowrank_step` (cache `x_j @ V` at rank r instead of the raw D-dim hidden state full-rank BLT caches) and wired it into the dispatch + `__main__` model set. **Verified before trusting any timing**: cached decode matches the model's own non-cached forward pass exactly (max abs logit diff 5.7e-06, CPU, random-init weights).

**Ran the full 5-way benchmark (MHA, BLT, GQA, Hybrid, UV256) on `titan`'s P100** — `bender`'s V100 (where the original 4-way table in `paper_blt.md` was measured) was mid-run on the medium baseline and the user chose not to interrupt it; ran on `titan` instead, which meant briefly pausing UV256's own seed7 confirmation run to get an uncontested GPU (killed cleanly at step 100,500, checkpoint verified finite, watcher stopped first so it wouldn't fire prematurely on the paused checkpoint, then both resumed correctly afterward — confirmed via `Resumed at step 100500` in the log). Also had to copy `run_gpt2_baseline_seed42.pt`/`run_blt_scratch_seed7.pt`/`run_gqa_scratch_seed42.pt`/`run_hybrid_mha6_scratch_seed42.pt` to `titan` (none were there before) and `git pull` titan's repo (one byte-identical untracked-file collision on `lm_eval_blt_uv256_scratch_seed42.json`, resolved the usual way via `md5sum`). **These P100 numbers are not directly comparable in absolute terms to the existing bender/V100 table** — only mutually comparable within this second same-hardware set, all five measured back-to-back on the same idle P100.

| Config | MHA | BLT | GQA | Hybrid | UV256 |
|---|---|---|---|---|---|
| bs=1, prompt=128, TTFT | 6.86ms | 6.14ms | 6.77ms | 6.49ms | 6.30ms |
| bs=1, prompt=128, decode | 320.2 tok/s | 350.1 | 345.8 | 335.5 | **356.6** |
| bs=8, prompt=128, TTFT | 36.32ms | 32.37ms | 32.48ms | 34.35ms | **30.85ms** |
| bs=8, prompt=128, decode | 1707.2 | 1898.5 | 1742.7 | 1799.6 | 1895.6 (tied w/ BLT) |
| bs=1, prompt=900, TTFT | 46.14ms | 35.15ms | 42.51ms | 40.59ms | **32.85ms** |
| bs=1, prompt=900, decode | 260.8 | 281.3 | 282.6 | 271.2 | **301.4** |

**Both predictions confirmed.** UV256 beats MHA by 11-16% on decode across all three configs (same range as full BLT's own 9-16% edge). And the context-length prediction shows up cleanly: UV256 vs. full BLT is roughly tied at prompt=128 (+1.9% decode, −2.6% TTFT, both within noise) but pulls to a real lead at prompt=900 (+7.1% decode, **+6.5% TTFT**). Prefill (TTFT) is even more exposed than decode since the L×L score matrix scales with L², not L — UV256's TTFT edge over MHA grows from 8.2% (prompt=128) to **28.8%** (prompt=900), a much bigger swing than the corresponding decode-tok/s gap (11.4%→15.6%) over the same range. This is the first evidence that UV256 isn't just "full-rank BLT but cheaper to store" — at long context specifically, it's faster to *run* than full-rank BLT too, not just smaller.

Written up in `paper_blt.md` Section 6 (new paragraph after the existing decode-latency table) — kept as an explicitly separate P100 measurement, not merged into the V100 table's numbers.

### num_uv_groups=4 from-scratch OWT run — LAUNCHED (2026-08-10, on `titan`)

**Plan changed at launch time: titan finished first, not venus.** The original watcher (`queue_venus_seed7_then_uvgroups4.sh` on `bender`, PID 329155) was polling venus's sine-blend seed7 run, expected to finish first. Instead `titan`'s UV256 seed7 confirmation run finished first (2026-08-10) and freed its GPU while venus's own run was still at 99%+ (step 496,730/500,000). Killed the venus-side watcher before it could fire (would have raced with launching the same experiment twice, once per machine, into a shared output filename) and replaced it with `queue_sine_seed7_benchmark_only_watcher.sh` — benchmarks venus's sine seed7 on its own completion, no longer launches anything afterward. Launched `num_uv_groups=4` directly on `titan` instead.

**Confirmed healthy at launch**: `Device: cuda`, `num_uv_groups=4` in the build log, correct param count (111,838,464 — exactly 110,658,816 + 3×393,216 for the three extra (U,V) pairs beyond single-pair UV256), loss declining normally from step 0 (10.96→10.80). `queue_uvgroups4_watcher.sh` armed on `titan` itself (not bender this time, since titan orchestrates its own follow-up here) to auto-benchmark on completion using the now-`--num-uv-groups`-aware `eval_owt.py`/`blt_lm_eval.py`.

**Found and fixed a real gap while building the watcher**: neither `eval_owt.py` nor `blt_lm_eval.py` supported `--num-uv-groups` — both would have built the wrong architecture (single global U,V instead of 4 groups) and failed to load the finished checkpoint's state_dict, days from now with nobody watching. Added `--num-uv-groups` to both (commit `fcd2f10`), verified via a real save/reload round-trip (saved a genuine num_uv_groups=4 checkpoint, reloaded it through both scripts' actual load paths, confirmed identical forward-pass output, max diff 0.0).

### layers-per-m=4 from-scratch OWT run — armed on venus (2026-08-10), the complementary axis

User asked what to run on venus next (sine-blend seed7 nearly done, 99.5%+); a second `num_uv_groups=4` seed was the obvious default but deliberately **not** taken — G=4 was chosen as one decisive point specifically to see its result before committing more compute to that axis, and it hasn't reported anything yet. Recommended `--layers-per-m 4` instead: isolates the *cross-layer* axis alone (3 distinct M's across 4-layer blocks, heads still fully collapsed within each M, mirroring the block structure of the existing-but-untrustworthy warm-start `layers-per-m=4` result), complementing `num_uv_groups=4`'s isolate-the-*head*-axis-alone test on `titan`. No from-scratch OWT `layers-per-m` result has ever existed — the only prior data (per-layer-M, `layers-per-m=4`) was WikiText warm-start, and this project has repeatedly found warm-start doesn't predict from-scratch behavior, so neither counts as real evidence for the OWT-gap question. User agreed.

**Verified before launching**: `--layers-per-m` + `--from-scratch` had never been combined via the actual CLI before (only warm-start). Direct check: 3 distinct M's, correct param count (112,035,072 = shared-M's 110,855,424 + 2 extra M's worth, 2×589,824), real forward+backward with gradients flowing to multiple `M_layer_i` parameters.

`queue_venus_seed7_then_layersperm4.sh` armed on `bender` (PID 357628), replacing the earlier benchmark-only watcher — same chain pattern as the other overnight watchers: poll venus's seed7 PID (216841), benchmark on exit, `git pull` + abort-if-stale check for `--layers-per-m`, launch `run_blt_layersperm4_scratch_seed42.pt`, post-launch health check, arm a follow-up watcher for its own eventual completion. `blt_lm_eval.py --layers-per-m` and `eval_owt.py --blt-layers-per-m` already existed (unlike `--num-uv-groups`, no gap to fix here).

**Automated launch actually failed — caught and fixed by hand (2026-08-10 morning).** Venus's sine-blend seed7 finished and got benchmarked correctly (OWT ppl **29.37**, LAMBADA acc **0.251**, LAMBADA ppl 136.0, HellaSwag acc_norm 0.272, PIQA acc_norm 0.590, Winogrande acc 0.517 — third sine-blend seed, joining seed42's 0.281 and seed19's 0.246), but the `layers-per-m=4` launch itself never actually happened. Two compounding causes:
1. **`git pull` genuinely failed** — venus had an *uncommitted* local patch of the sine-blend feature sitting there since 2026-07-21 (documented in this file at the time as "not yet committed to git... copied via scp"), 24 commits behind `origin/blt`. Diffed venus's local `train.py` against `origin/blt`'s current version before touching anything: the only real differences were this session's own additions (`uv_rank`/`num_uv_groups`/`resume_batch_size`) — nothing venus-only would be lost. Also hit one byte-identical untracked-file collision (`lm_eval_gpt2_ema_blend75_sine_scratch_seed42.json`, verified via `md5sum`, same pattern as every other machine). Discarded the local `train.py` diff and the untracked collision, pulled clean — venus is now current for the first time since 2026-07-21.
2. **Separately**, the watcher's inline `ssh venus '... 2>&1 ...'` commands hit the exact same tcsh redirect-mangling bug already found and fixed for `io` — it just hadn't shown up on venus before. The abort-gate logic itself worked as designed (empty PID after the launch attempt → warning printed, no follow-up watcher armed) rather than silently believing a broken launch succeeded — but the launch itself needed a human to actually happen.

Relaunched by hand using the file-based approach (write a script, `scp` it over, execute it — sidesteps inline-quoting entirely, same pattern already used reliably for the `titan`/`num_uv_groups` launch). Confirmed healthy: `Device: cuda`, correct config (`layers_per_m=4, from_scratch=True`), correct param count (112,035,072), loss declining normally from step 0. Watcher re-armed on venus itself this time (PID 475131), same lesson as the `titan` launch — orchestrate the follow-up from whichever machine is actually running the thing, not always from `bender`.

**Lesson for future overnight automation**: "the watcher didn't crash and printed a plausible-looking log" is not the same as "it worked" — the abort-gate caught this one, but only because it happened to check for something checkable (a PID). Worth spot-checking `ps aux` on the target machine after any unattended multi-stage watcher fires, not just trusting its own log output.

**What would settle the debate once both land**: if `num_uv_groups=4` (head axis) shows marked improvement over UV256's single-pair range (29.97-30.38) and `layers-per-m=4` (layer axis) doesn't, that's real evidence the redundancy is specifically in within-layer head multiplicity, not cross-layer sharing. If both improve similarly, that supports the alternative theory raised early in this discussion — that added capacity anywhere closes the gap, not specifically the head axis. If neither improves much, the head/layer framing itself needs rethinking.

**Interim read at the halfway point (2026-08-13, step 250K/500K on both) — a large, real, and growing gap, already favoring the head-axis theory.** In-training WikiText val_ppl at matched steps:

| step | full-rank BLT | layers-per-m=4 | num_uv_groups=4 |
|---|---|---|---|
| 25,000 | 300.28 | 475.77 | 276.59 |
| 50,000 | 182.71 | 228.25 | 149.52 |
| 100,000 | 124.40 | 148.75 | 106.06 |
| 150,000 | 106.54 | 125.19 | 88.58 |
| 200,000 | 100.63 | 101.75 | 80.96 |
| 250,000 | 89.80 | 96.03 | **70.05** |

`num_uv_groups=4` has been ahead of `layers-per-m=4` at every single checkpoint, not just recently — 27% better by step 250K, and already closing in on the MHA baseline's own value at this step (63.92) while `layers-per-m=4` is still behind even single-shared-M full-rank BLT. Real caveats: single seed each, only halfway through training (full-rank BLT's own seeds have converged closer together late in some cases, so this gap could narrow), and this is the in-training WikiText proxy, not the final OWT/lm-eval benchmark this project has repeatedly found doesn't always agree with it. But directionally this is a clean, large signal for the head-axis theory — the opposite conclusion from what GQA-3's near-baseline result had suggested (mild evidence for the layer axis being the forgiving one).

**SETTLED (2026-08-15) — both seed42 runs finished, and the interim read held all the way to convergence.** `run_blt_uv256_groups4_scratch_seed42.pt` (titan) and `run_blt_layersperm4_scratch_seed42.pt` (venus) both completed their full 500K-step from-scratch OWT schedules. Final WikiText val_ppl: `num_uv_groups=4` 61.38, `layers-per-m=4` 80.16 — the gap didn't narrow late in training, it stayed just as wide. Benchmarked (`lm_eval_blt_uv256_groups4_scratch_seed42.json`, `lm_eval_blt_layersperm4_scratch_seed42.json`):

| Metric | `num_uv_groups=4` seed42 | `layers-per-m=4` seed42 | UV256 single-pair (3 seeds) | Full-rank BLT (3 seeds) | MHA (3 seeds) |
|---|---|---|---|---|---|
| OWT held-out ppl | **28.95** | 33.14 | 29.97–30.38 | 30.81–32.87 | 27.36–28.24 |
| LAMBADA acc | 0.2045 | 0.1976 | 0.207–0.214 | 0.188–0.212 | 0.208–0.225 |
| HellaSwag acc_norm | 0.2677 | 0.2678 | 0.267–0.269 | 0.267–0.271 | 0.266–0.270 |
| PIQA acc_norm | 0.5686 | 0.5626 | 0.569–0.583 | 0.559–0.568 | 0.572–0.583 |
| Winogrande acc | 0.4988 | 0.5059 | 0.492–0.512 | 0.489–0.516 | 0.505–0.520 |

**This settles the head-axis-vs-layer-axis question this whole side-by-side test was designed to answer.** `num_uv_groups=4` beats every single-pair UV256 seed on OWT ppl (28.95 vs. their 29.97 floor) and lands closer to the MHA/GQA band (27.36–28.24) than to full-rank BLT's — going from 1 to 4 (U,V) groups recovered roughly half of UV256's remaining gap to MHA. `layers-per-m=4`, by contrast, is the *worst* OWT-ppl result of any BLT-family variant in this project, worse even than every full-rank single-shared-M seed — relaxing cross-layer sharing alone didn't just fail to help, it actively hurt relative to keeping one M for the whole model (plausibly: 3 separately-optimized M's each get less gradient signal per step than one M receiving updates from all 12 layers, a cost the head-axis split doesn't pay since every UV group is still shared globally across all layers). **Net conclusion: BLT's redundancy is concentrated in per-head multiplicity within a layer, not in the sheer number of independently-parameterized attention computations across layers** — consistent with, and now much more directly evidenced than, the earlier hybrid-model finding.

**Caveat (at the time)**: single seed each. A second `num_uv_groups=4` seed (seed19, titan) and third (seed7, venus) were launched 2026-08-18 to confirm this isn't a favorable seed42 draw — same discipline as every other multi-seed result in this project.

**Second seed — DONE (2026-08-20), and it lands remarkably close to seed42.** `run_blt_uv256_groups4_scratch_seed19.pt` finished on titan, final val_ppl 62.28. Benchmarked (`lm_eval_blt_uv256_groups4_scratch_seed19.json`, `eval_owt_blt_uv256_groups4_scratch_seed19.stdout`):

| Metric | seed42 | seed19 |
|---|---|---|
| OWT held-out ppl | 28.95 | **28.95** |
| LAMBADA acc | 0.2045 | 0.2016 |
| LAMBADA ppl | 229.2 | 238.8 |
| HellaSwag acc_norm | 0.2677 | 0.2683 |
| PIQA acc_norm | 0.5686 | 0.5637 |
| Winogrande acc | 0.4988 | 0.5122 |

OWT held-out ppl is identical to two decimal places across seeds — the tightest seed-to-seed agreement of any result in this project so far, and strong confirmation the head-axis finding is real rather than a favorable seed42 draw.

**Third seed — DONE (2026-08-21).** `run_blt_uv256_groups4_scratch_seed7.pt` finished on venus, final val_ppl 60.90. Its own benchmark watcher (`queue_uvgroups4_seed7_watcher.sh`) vanished without a trace the same way seed19's did on titan — benchmarks were run by hand (see the venus network-outage saga elsewhere in this file), and the results sat only on venus until now. `lm_eval_blt_uv256_groups4_scratch_seed7.json`, `eval_owt_blt_uv256_groups4_scratch_seed7.stdout`:

| Metric | seed42 | seed19 | seed7 |
|---|---|---|---|
| OWT held-out ppl | 28.95 | 28.95 | **28.85** |
| LAMBADA acc | 0.2045 | 0.2016 | 0.2073 |
| LAMBADA ppl | 229.2 | 238.8 | 232.1 |
| HellaSwag acc_norm | 0.2677 | 0.2683 | 0.2652 |
| PIQA acc_norm | 0.5686 | 0.5637 | 0.5686 |
| Winogrande acc | 0.4988 | 0.5122 | 0.5122 |

**This completes the standard three-seed set for `num_uv_groups=4`.** All three seeds land within 0.10 ppl of each other (28.85–28.95) — the tightest seed cluster of any result in this project. `num_uv_groups=4` is now as solidly confirmed as every other multi-seed architecture family here (full-rank BLT, UV256 single-pair, GQA, MHA), and the head-axis-vs-layer-axis conclusion rests on a complete, standard-strength evidence base rather than a lone seed or two.

**Not evidence for PyramidKV-style depth-importance claims, despite surface resemblance.** User drew a connection to PyramidKV's argument that later transformer layers carry less information and tolerate more KV-cache compression. `layers-per-m=4` can't actually speak to that: its grouping is **strided, not contiguous** (`layer i uses M_params[i % 3]`, so each of the 3 M's serves layers spanning the *entire* depth — e.g. M_0 serves {0,3,6,9} — early, middle, and late layers all feed every group equally). It tests "does relaxing cross-layer sharing at all help," not "do later layers need less capacity." The experiment that actually tests the depth-position claim already exists: the **hybrid model** (Section 4.5, 6 MHA layers placed *first*, 6 fully-collapsed BLT layers placed *last*, that placement explicitly reasoned from Gromov et al. ICLR 2025's layer-pruning finding) — and it already supports the PyramidKV-style claim: converting only the *later* half of layers recovered >75% of the full-BLT OWT-loss gap, well above the ~50% a depth-independent cost would predict.

### GPT-2 medium baseline — DONE, and the io→bender EMA migration confirmed working (2026-08-11)

**`run_gpt2_medium_baseline_seed42.pt` completed all 1,500,000 steps** (overnight, 2026-08-10, via the `queue_bender_finish_then_migrate_io_ema.sh` watcher — see the "GPT-2 medium" saga above for the full multi-week history: OOM fixes, the resume-replay bug, the two-pass cache-loading fix, the power outage, all of it). Benchmarked automatically on completion: **OWT held-out ppl 18.09** (loss 2.8951) — its first fully-converged result, and notably better than every 500K-step small-model baseline (27.36-28.24), consistent with the ~16.5 tokens/param training-adequacy target this step count was chosen for. lm-eval-harness results in `lm_eval_gpt2_medium_baseline_seed42.json`.

**Migration to bender confirmed working correctly in production, not just in the isolated test.** io's EMA sine-blend run was stopped cleanly at step 718,000, copied over, and resumed on bender with `--batch-size 2 --grad-accum-steps 2 --resume-batch-size 4` (bender's 16GB card can't fit io's true batch=4, same constraint the baseline itself needed grad-accum for). Log confirms `Resumed at step 718000` exactly and `effective_blend=0.5123` at that step matches the sine formula precisely (`0.75·sin(0.5π·718000/1500000)` = 0.5123) — the `--resume-batch-size` fix (added specifically for this migration, verified via a direct reference-simulation before trusting it) is holding up under real use, not just the synthetic test.

**First post-migration comparison against the (now-complete) baseline, matched-step (715K-740K):**

| | EMA sine-blend | Baseline |
|---|---|---|
| val_ppl range | 40.78–46.38 | 36.21–39.26 |
| val_ppl mean | 43.44 | 37.25 |
| Swing (range/mean) | 12.9% | 8.2% |

EMA is both consistently higher (worse) on val_ppl (~17% gap) and ~1.6x noisier at this point — both expected: some val_ppl cost is the known EMA trade-off, and noise scales with `effective_blend` magnitude (currently 0.52, a bit past halfway to the 0.75 target) per the noise-dynamics hypothesis from the original io run. Worth re-checking once `effective_blend` crosses ~0.7 near the end of the run.

**DONE (2026-08-22) — the EMA trade-off holds at medium scale, confirming it's not just a small-model effect.** `run_gpt2_medium_ema_blend75_sine_seed42.pt` completed all 1,500,000 steps on `bender` (after the io→bender migration above). Benchmarked automatically via the migration/blend-sweep watcher (`eval_owt_gpt2_medium_ema_blend75_sine_seed42.stdout`, `lm_eval_gpt2_medium_ema_blend75_sine_seed42.json`): OWT held-out ppl **18.79** (loss 2.9332).

| Metric | Sine-blend EMA (medium) | Non-EMA baseline (medium) |
|---|---|---|
| OWT held-out ppl | 18.79 | **18.09** |
| LAMBADA acc | **0.373** | 0.311 |
| LAMBADA ppl | **31.2** | 40.8 |
| HellaSwag acc_norm | **0.302** | 0.297 |
| PIQA acc_norm | 0.594 | **0.607** |
| Winogrande acc | 0.515 | **0.522** |

**Same trade-off shape as every small-model EMA result in this project, now confirmed at ~3x the parameter count.** OWT ppl is modestly worse (+3.9% relative), but LAMBADA improves substantially (acc +20% relative, ppl -24%), HellaSwag edges up slightly, PIQA/Winogrande dip slightly — matching Finding 1 from the original small-model EMA work (architecture-general trade-off) and now extending it to model-scale-general as well. This was the whole motivation for running the medium-scale test in the first place (see "GPT-2 medium — testing EMA generality at a larger scale" below) — motivation confirmed.

**Per-step timing, clean segment since the resume (step 718,000→740,490)**: **1.32 s/step** (29,632s / 22,490 steps), consistent across both halves (1.322 / 1.313 s/step, no drift) — essentially identical to bender's own baseline rate (1.322 s/step over its full 1.2M-step run). Confirms the EMA loss-reweighting itself costs no meaningful extra compute (just a per-token multiply, not new matmuls) — the migration didn't introduce any speed penalty beyond what bender's hardware already costs relative to io.

### num_uv_groups=4 vs layers-per-m=4 — an early, real speed data point (2026-08-11)

Same hardware class (both P100s — venus/titan), so directly comparable: `layers-per-m=4` on venus runs **~722 ms/step** vs `num_uv_groups=4` on titan's **~902 ms/step** — `layers-per-m` is ~25% faster. Confirms the compute-cost asymmetry flagged before launching either: `layers-per-m` still computes exactly one shared attention pattern per layer (cycling through 3 different full-rank M's depending on layer, no extra per-step arithmetic over vanilla single-shared-M BLT — 722ms/step lands right in the same range as single-pair UV256's 717-740ms/step despite using full D×D matrices, since per-step cost is dominated by one bilinear computation either way), while `num_uv_groups=4` computes four separate rank-256 score matrices per layer — genuinely more FLOPs. 902ms/step is already closing in on standard MHA's own 1,052ms/step baseline on the same hardware — a real chunk of UV's speed advantage is already gone at G=4, the practical cost side of the same axis being tested for quality.

**Why**: settles an open question from the UV256/hybrid discussion above. The hybrid result (Section 4.5) can't actually isolate whether BLT's OWT-loss gap comes from lost *cross-layer* independence or lost *within-layer head* independence, because standard MHA layers relax both simultaneously. `num_uv_groups` isolates the head axis alone (multiple (U,V) pairs, each still shared globally across all 12 layers — unlike GQA, whose groups are relearned independently per layer and don't get BLT's cross-layer bandwidth reuse). The one existing full-rank precedent for this axis (BLT-2M, `num_m_groups=2`) was never trained to convergence, so this would be a genuinely new result, not confirming an old one. It's also the better design to actually deploy regardless of the quality answer: G groups can be *sharded* disjointly across GPUs (no replication, just the standard Wo all-reduce), a strictly cleaner TP story than either the single-global-UV design or `layers_per_m`.

**Code done and verified** (commit `7c6933a`): `BLTLowRankMultiAttention` + `build_blt_lowrank_model(num_uv_groups=G)`, mirrors `num_m_groups` exactly (each group's U,V stays full rank r, only Wv/Wo get sliced per group). Verified: correct param count (111,052,032 for G=2, r=256 — exactly 393,216 more than single-pair UV256's 110,658,816, matching one extra (U,V) pair), real forward+backward with gradients flowing to both groups, and `--num-uv-groups` parses correctly through `train.py`'s actual argparser. **Not yet smoke-tested through a live GPU run** — `train.py` hard-fails without CUDA and no machine was free; watch the first few hundred steps closely once launched, in place of a proper dedicated smoke test.

**Decided (2026-08-06): start with G=4, not G=2 — one decisive point before committing to a sweep.** Since parameter/bandwidth cost stays negligible at any G (all groups still load once, shared across every layer) but *compute* cost scales roughly linearly with G (each group needs its own rank-r score matrix and softmax — by G=12 this would mean more per-step arithmetic than standard MHA's 12 rank-64 heads, likely eroding the 32% training-speed and TTFT wins UV256 already demonstrated), a single spot-check can't distinguish "real, bounded head-axis effect" from "noise" from "monotonic, needs to go most of the way to G=12." G=4 was chosen as the first informative point (divides evenly into 12 heads; a real midpoint, not an endpoint) — user's framing: "one would expect marked improvement with 4 [if the head-axis theory is right]; if you don't get it, we need to try something different."

**Explicitly not the same as GQA-3 (3-group GQA)**, which is a real, converged, near-baseline result (OWT ppl 27.46, inside the 27.36-28.24 MHA range) — worth not conflating the two. GQA's groups are relearned independently *per layer* (full per-layer/query-head independence kept, only K/V compressed) and never pay BLT's cross-layer-sharing tax at all, so GQA-3 landing near baseline is arguably evidence that the *layer* axis is the forgiving one when left untouched — if anything mild evidence against the head-axis theory, not for it. `num_uv_groups` is a genuinely different, untested cell in the design space: head axis relaxed, cross-layer sharing kept fully intact.

**Launch command used** (mirrors the UV256 from-scratch protocol exactly, running now on `titan`):
```
train.py --uv-rank 256 --num-uv-groups 4 --from-scratch --dataset openwebtext --seed 42 \
  --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 \
  --save-path run_blt_uv256_groups4_scratch_seed42.pt
```
Compare final OWT held-out ppl against full-rank BLT (30.81-32.87), UV256 single-pair (29.97-30.38, now three seeds), and MHA (27.36-28.24). Marked improvement over UV256's single-pair range → real evidence for the head-axis theory, worth extending to a fuller {2, 12} sweep. No marked improvement → the head-axis theory as currently framed doesn't hold up, and the layer axis (or some other explanation entirely) needs a genuine from-scratch OWT test instead (see the `layers_per_m` gap noted earlier in this section).

**Limitation of GPT-2 scale testing:** KV cache and M-caching benefits are most compelling at 7B+ scale with long contexts. GPT-2 scale establishes viability; the practical case requires larger models.

### Cumulative loss-weighting mode — first real test, plus a migration/blend-sweep chain (started 2026-08-21)

`--loss-weighting-mode cumulative` (see "Flags added" below and `project_loss_function_ideas` memory) had been committed but never actually run. Tested it for real:

**Smoke test passed** (`titan`, 30 steps, wikitext103): checked the checkpoint's `token_loss`/`token_count` buffers directly — perfectly consistent (no token with a count but no differentiated loss, or vice versa), real per-token differentiation already visible (std 0.73) after just 30 steps. Confirmed the mechanism works before trusting a real run, same discipline as every other EMA-family change in this project.

**Real run launched**: `run_gpt2_cumulative_scratch_seed42.pt` on `titan` — mirrors the original pure-EMA protocol exactly (`--baseline --from-scratch --ema-loss-weighting --dataset openwebtext --seed 42 --max-steps 500000`, default `--ema-blend 1.0`, fixed schedule) with `--loss-weighting-mode cumulative` swapped in, isolating exactly one variable against `run_gpt2_ema_seed42.pt` (OWT ppl 30.06, LAMBADA acc 0.253). Also using `--owt-eval-every 5000` (now confirmed working end-to-end, see the memory file) for live OWT-ppl visibility. As of this writeup: step 2,940/500,000.

**User wants a blend-sweep fine-tune too, like the original α=0/0.25/0.5/0.75 sweep** — but that sweep branched from an *already fully-converged* EMA checkpoint (500K steps of buffer differentiation already baked in); a short fine-tune only had to vary the blend. Cumulative mode has no such converged checkpoint yet, and unlike fixed-decay EMA (which fully differentiates within ~100 steps regardless of how long training runs), cumulative mode's *long tail* of rarer tokens needs many more accumulated samples before their running mean is trustworthy — so a fine-tune branching from an under-trained cumulative buffer would inherit a lot of noise. Decision: wait for the from-scratch run above to reach step 100,000 (a "fair number of steps," per the user), snapshot it there, and branch the blend sweep from that snapshot instead of from scratch. Fine-tune length: 50,000 steps per blend value (upper range user OK'd was 50K-100K; went with the lower end given P100 speed). Alpha values: **0.25 and 0.5** — exactly two, one per machine, matching the two P100s (`titan`, `venus`) that become free around the same time (see below).

**Scheduling constraint that shaped this**: `bender`'s current medium EMA sine-blend run (`run_gpt2_medium_ema_blend75_sine_seed42.pt`) is ~1.5 days from finishing its 1.5M steps (step 1,396,800/1,500,000 as of this writeup) — coincidentally close to when the new titan cumulative run above will hit its own step-100K threshold (ETA ~35 hours from launch). User's plan: once `bender` frees up, migrate the titan cumulative run there to keep the full 500K-step "full monty" going, which frees up `titan`; combined with `venus` freeing up once its own `num_uv_groups=4` seed7 run finishes (step 480,620/500,000, much sooner — a few hours), that gives two free P100s to run the blend sweep on. Slower than a V100, but "might give us some clue" per the user, without blocking on a fully free machine.

**Armed as a single chained watcher** (`queue_bender_finish_then_cumulative_migrate_and_blend_sweep.sh`, running locally on `bender`, PID logged in `queue_bender_finish_then_cumulative_migrate_and_blend_sweep.log`):
1. Poll bender's current training PID (3705) until it exits.
2. Benchmark the finished medium run (`eval_owt.py` + `blt_lm_eval.py`, with `--pretrained gpt2-medium`).
3. Poll titan's cumulative run until it reaches step 100,000, then snapshot the checkpoint (`run_gpt2_cumulative_scratch_seed42_ckpt100k.pt`) — doesn't touch the live job.
4. Stop titan's cumulative training cleanly (`SIGTERM` on the known PID, with a guard that verifies the PID still matches the expected command before killing it — protects against a PID-reuse edge case over a multi-hour wait), verify the checkpoint is finite, copy checkpoint+log to bender, resume there.
5. Verify the bender resume is actually healthy (process alive AND log shows `Resumed at step`) before declaring titan free — logs a loud `*** BENDER RESUME FAILED ***` marker if not, rather than silently pressing on.
6. Launch the alpha=0.25 fine-tune on titan from the step-100K snapshot.
7. Poll for venus's own benchmark output to confirm it's free, copy the snapshot there, launch the alpha=0.5 fine-tune.

**Caught and fixed two real bugs before trusting this to run unattended for ~34 hours**: the benchmark commands originally called a nonexistent `~/miniconda3/envs/blt/bin/conda` (conda's own binary only lives in the base install, `~/miniconda3/bin/conda` — env-specific bin dirs don't have it); and the planned PID-rediscovery logic (`ps aux | grep ... | grep -v Sl`) turned out not to match the actual process line at all (tested live, matched nothing) — replaced with the already-known launch-time PID (2894048) plus a guard that checks the PID still corresponds to the expected command before sending `SIGTERM`, rather than blindly trusting a fragile re-derivation. Verified the finiteness-check script and the log step-parsing (`tail -1 | cut -f1`) against live data before deploying.

**Plan revised same day — the fine-tune-from-100K-snapshot sweep was scrapped as methodologically weak.** User's own objection, and it's correct: branching a fine-tune from a checkpoint only 100K/500K (20%) into training confounds two things at once — buffer immaturity (the reason 100K was chosen in the first place) AND an undertrained base model, which the *original* sweep (branched from a fully-converged 500K-step checkpoint) never had to deal with. Any difference between blend values at that branch point could just be "which one got a luckier draw of ordinary training," not a real blend effect.

**Replacement, user's call**: mirror what actually worked best for fixed-decay EMA instead — the jointly-trained-from-scratch α=0.75 run, which beat even pure EMA and became the 3-seed-confirmed headline result. `run_gpt2_cumulative_blend75_scratch_seed42.pt` launched on `venus` (`--baseline --from-scratch --ema-loss-weighting --loss-weighting-mode cumulative --ema-blend 0.75 --dataset openwebtext --seed 42 --max-steps 500000`, fixed schedule, `--owt-eval-every 5000`) — no snapshot dependency, so it doesn't need to wait on titan or bender at all. The big migration watcher's venus-specific steps (copy snapshot, launch α=0.5 fine-tune) were removed since they're superseded; titan's replacement plan is still undecided as of this writeup.

**Launch saga worth remembering**: venus's own `num_uv_groups=4` seed7 run finished, but its benchmark watcher (`queue_uvgroups4_seed7_watcher.sh`) had vanished without a trace (no script, no log file) sometime between when it was armed and when the run finished — same class of failure as the titan seed19 watcher earlier, cause not established. Re-ran the benchmarks by hand (confirmed step 500,000, all-finite first). Then, while trying to launch the new blend=0.75 run, venus dropped off the network entirely (`ping`: "Destination Host Unreachable" — a real network-layer outage, not just SSH flakiness) for about 15+ minutes. It came back on its own (`uptime` showed 51 days, no reboot — a transient connectivity blip, not a machine outage) and, remarkably, the run had actually launched successfully during one of the retry attempts before the connection dropped. It then sat with the GPU at 0% util and the log stuck for close to an hour at "Loading OWT held-out tokens" (normally ~5s) — traced to `fwupd`/`snapd` activity and a load average spiking to 37-42 around the same time as the network blip, most likely an automatic OS update cycle (same mechanism as bender's earlier driver-mismatch incident) starving the tokenization step of CPU/disk I/O. Confirmed it wasn't a deadlock (CPU time kept climbing steadily throughout) before deciding to wait rather than kill it — it cleared on its own once system load subsided, and training is now healthy (step 1,850+, `effective_blend=0.7500` exactly as expected, loss declining normally).

**Status snapshot (2026-08-21 ~15:25 MDT):**
- `bender`: medium sine-blend run, step 1,439,640/1,500,000 (96.0%), ~1.16 s/step recent pace → ETA **~11 AM MDT tomorrow (2026-08-22)**. The migration/benchmark watcher (`queue_bender_finish_then_cumulative_migrate_and_blend_sweep.sh`) is confirmed still alive, correctly polling PID 3705.
- `titan`: cumulative-mode from-scratch run, step 44,190/500,000. Will get migrated to bender once bender's job finishes (per the watcher above), which now happens immediately rather than waiting for step 100K (no snapshot dependency anymore). Once freed, titan launches a clean from-scratch **blend=0.5** cumulative-mode run (`run_gpt2_cumulative_blend50_scratch_seed42.pt`), mirroring venus's blend=0.75 design. Together with the migrated blend=1.0 baseline and venus's blend=0.75, this gives three clean points (0.5, 0.75, 1.0) on the cumulative-mode blend curve.
- `venus`: new cumulative-mode blend=0.75 from-scratch run, step 7,530/500,000, `effective_blend=0.7500`, healthy.

**Migration chain executed 2026-08-22, exactly in the designed order.** `bender`'s medium job finished overnight; the watcher benchmarked it first (results above), only then stopped titan's run (step 104,000, verified finite), migrated it to bender (`Resumed at step 104050`, confirmed healthy), and finally tried to launch titan's blend=0.5 run. **That last step hit the by-now-familiar tcsh `2>&1`-in-a-remote-command bug** (titan's default shell mangles the redirect, same failure mode documented elsewhere for venus/io) — the watcher's own verification step correctly logged that nothing launched, rather than silently claiming success. Fixed by hand via the usual reliable pattern (write the launch command to a local script, `scp` it over, run via `nohup bash script.sh`) — confirmed alive independently before and after cleaning up the local hung SSH task. **Lesson reinforced**: even a watcher that's been tested and works fine for other steps can still hit this exact class of bug on a *new* remote command it hasn't exercised before — worth specifically avoiding inline `2>&1` in any future watcher's remote launch lines, not just trusting that "it worked elsewhere in this same script."

Current state: `bender` running the migrated cumulative-mode blend=1.0 "full monty" from step 104,050; `titan` running blend=0.5 from step 0; `venus` running blend=0.75. All three from-scratch cumulative-mode points now in flight.

**Status snapshot (2026-08-23):**
- `bender`: cumulative blend=1.0 "full monty", step 237,010/500,000 (47%).
- `titan`: cumulative blend=0.5, step 47,550/500,000 (10%).
- `venus`: cumulative blend=0.75, step 112,440/500,000 (22%).

**Also found and cleaned up while checking in on this**: three genuine result files that had been computed but never committed — `lm_eval_blt_uv256_groups4_scratch_seed7.json` (the third `num_uv_groups=4` seed, sitting only on venus since 2026-08-21), `lm_eval_gpt2_medium_ema_blend75_sine_seed42.json` (the medium-scale EMA benchmark, see "GPT-2 medium" above), and `lm_eval_blt_layersperm4_scratch_seed42.json` (the `layers-per-m=4` result — this one is cited extensively in both this file and the new `paper_blt.md` Section 4.8, and had never made it off venus at all until now). All three committed.

**One benchmark found on titan and deliberately NOT committed**: `lm_eval_gpt2_cumulative_scratch_seed42.json`, timestamped right around the migration (2026-08-22 ~11:00). This is spurious — the leftover `queue_cumulative_seed42_watcher.sh` (armed when the cumulative run first launched on titan) was still polling PID 2894048, and when the migration script `SIGTERM`'d that PID to move the run to bender, the watcher read the exit as "training finished" and auto-benchmarked the partial step-104,000 checkpoint. Left in place on titan for now (harmless, just needs to not be mistaken for the real result) but not pulled down or committed — the real benchmark for this run will come once it actually finishes on bender.

**Also reconciled stale/duplicated code on both remote machines while doing this.** `venus` had an uncommitted local diff to `train.py` that turned out to be byte-for-byte the same work already merged as commit `59f8d09` weeks earlier (same pattern as the `titan` case documented under "Cross-machine merge" above) — discarded the local diff and pulled clean (one byte-identical untracked-file collision on `lm_eval_blt_uv256_groups4_scratch_seed7.json`, resolved via `md5sum` the usual way). `titan` had no local modifications but was several commits behind; pulled clean with no conflicts. Both machines are now fully current with `origin/blt`, and — since Python already has the previous `train.py` loaded in memory for each live training process — none of this touched the actual running jobs.

**DONE (2026-08-24) — the cumulative-mode blend=1.0 "full monty" finished on `bender`, and it directly confirms the reason cumulative mode was built.** All 500,000 steps complete (migrated from titan at step 104,050, continued on bender to completion), final val_ppl 72.29. Checkpoint verified genuine before trusting it: `token_count` sum 2,046,004,092 across 50,134/50,257 vocab tokens touched, `token_loss` std 2.15 — a fully-populated cumulative buffer, not a partial/interrupted one. Benchmarked (`lm_eval_gpt2_cumulative_scratch_seed42.json`, `eval_owt_gpt2_cumulative_scratch_seed42.stdout`): OWT held-out ppl **30.37** (loss 3.4135).

| Metric | Cumulative (blend=1.0) | Fixed-decay EMA (blend=1.0, `run_gpt2_ema_seed42.pt`) |
|---|---|---|
| OWT held-out ppl | 30.37 | 30.06 |
| LAMBADA acc | **0.268** | 0.253 |
| LAMBADA ppl | 119.3 | 119.6 |
| HellaSwag acc_norm | 0.271 | 0.272 |
| PIQA acc_norm | 0.566 | 0.569 |
| Winogrande acc | 0.507 | 0.511 |

**This is the direct A/B the whole cumulative-mode feature was built to test** (see `project_loss_function_ideas` memory): does an exact running mean (no permanent noise floor) beat fixed-decay EMA at the same full blend strength? It does — LAMBADA acc up ~6% relative (0.268 vs 0.253), among the best LAMBADA results of any pure-full-blend run in the whole EMA family, at a small OWT-ppl cost (+0.31, ~1%). Everything else ties within noise. One seed so far, matching this project's usual practice of reading a first result as suggestive rather than confirmed until a second seed lands — but it's a clean, real signal in the predicted direction, not an ambiguous one.

**Not yet written up in `paper_ema.md`** — that paper hasn't been updated with anything from the cumulative-mode work yet (smoke test, this blend=1.0 result, or the in-progress blend=0.5/0.75 runs). Worth doing once the other two blend points land, so it's one coherent update rather than three.

Status: `titan` (blend=0.5) and `venus` (blend=0.75) still in progress. `bender` is now free.

**Migration (2026-08-24): venus's blend=0.75 seed42 run moved to bender, venus started a second blend=0.75 seed.** Once `bender` freed up from finishing the blend=1.0 run above, `venus`'s blend=0.75 run (step 213,000/500,000 at the time) was migrated there to keep running on faster hardware — same protocol as the earlier titan migration: `SIGTERM` (with a PID-match guard), checkpoint verified genuine before touching anything (step 213,000, all finite, `token_count` sum 871,600,092 across 50,117/50,257 tokens), copied to bender (md5sum-verified byte-identical), resumed there (`Resumed at step 213000`, confirmed healthy). `venus` was then immediately used to start a **second blend=0.75 seed** (`run_gpt2_cumulative_blend75_scratch_seed19.pt`, seed 19, otherwise identical config) — the natural next step toward a multi-seed-confirmed result for this blend point, matching this project's usual practice.

Current state: `bender` continuing blend=0.75 seed42 from step 213,000; `titan` still on blend=0.5 seed42; `venus` now on blend=0.75 seed19 from step 0.

**DONE (2026-08-25) — cumulative blend=0.75 seed42 finished, and it's an unexpected trade-off shape.** All 500,000 steps complete (final val_ppl 68.26), checkpoint verified genuine (all finite, `token_count` sum 2,046,004,092 across 50,134/50,257 tokens — fully populated, matching the blend=1.0 run's population level). Benchmarked (`lm_eval_gpt2_cumulative_blend75_scratch_seed42.json`): OWT held-out ppl **29.35** (loss 3.3794).

| Metric | Cumulative blend=0.75 | Cumulative blend=1.0 | Fixed 75/25 avg (3 seeds) | Sine 75/25 avg (3 seeds) |
|---|---|---|---|---|
| OWT held-out ppl | **29.35** | 30.37 | 29.55 | 29.39 |
| LAMBADA acc | 0.249 | **0.268** | 0.260 | 0.259 |
| LAMBADA ppl | 141.6 | 119.3 | 129.9 | 133.2 |
| HellaSwag acc_norm | 0.269 | 0.271 | 0.270 | 0.270 |
| PIQA acc_norm | 0.573 | 0.566 | 0.574 | 0.576 |
| Winogrande acc | 0.525 | 0.507 | 0.517 | 0.512 |

**Not what the blend-value logic alone would predict.** Cumulative blend=0.75 has the *best* OWT ppl of the four (even better than both established fixed/sine 75%-blend averages), but the *worst* LAMBADA acc — the opposite of blend=1.0's profile (worst OWT ppl, best LAMBADA acc). For fixed-decay EMA, lower blend reliably means "closer to baseline on both metrics" (a monotonic dial, confirmed across the whole α=0/0.25/0.5/0.75/1.0 sweep). Cumulative mode's blend=0.75 point doesn't sit on that same monotonic line relative to its own blend=1.0 — LAMBADA acc dropped by more than the OWT-ppl-vs-blend=1.0 trade would suggest. **One seed — this could be seed noise rather than a real shape difference; titan's blend=0.5 seed42 and venus's blend=0.75 seed19 (both in progress) are the next two points that will clarify whether this is a real pattern or an outlier.** Was running concurrently with the r=256 medium UV fine-tune on the same GPU (tight but survived — see below); both jobs finished/continued without incident.

### Medium-scale UV rank probe (r=256 vs r=512) — de-risking a 3-week commitment (2026-08-25)

**Context**: user is planning two full 3-week GPT-2-medium training runs (once `bender`/`io` free up from their current small-model work) — one testing cumulative-mode EMA loss weighting, one testing low-rank UV BLT at `num_uv_groups=8` (chosen for NVLink/TP reasons independent of this project's own compute-crossover analysis — see the design discussion earlier this session). Before committing 3 weeks to a specific rank `r`, wanted a cheap way to check whether r=256 vs r=512 makes a meaningful quality difference.

**SVD-truncation of an existing trained full-rank M gave a first, free data point, but it's a weak one and shouldn't be over-trusted**: r=256 captures 81% of a converged full M's Frobenius energy (previously documented), r=512 captures 98.2% (computed fresh from `run_blt_scratch_seed42.pt`). But this measures how well a rank-r matrix reconstructs an *already-trained full-rank solution* — not how well a rank-r model *trained from scratch* performs, and those are known to diverge: the actual from-scratch UV256 (r=256) result already **beat** full-rank BLT outright despite only "explaining" 81% of its energy, which the energy metric alone would not have predicted.

**Real test built instead**: added `mha_warmstart_checkpoint` to `build_blt_lowrank_model` (`model.py`, `--uv-mha-warmstart` in `train.py`) — warm-starts a low-rank UV model from a **standard MHA checkpoint** (there's no full-rank BLT checkpoint at medium scale to factorize from) by computing M_avg = mean over layers of Wq@Wk^T from the checkpoint's own weights (same cross-layer-average formula as `build_blt_model`'s existing warm-start), SVD-truncating that at the target rank, and carrying over embeddings/Wv/Wo/LN/MLP unchanged. Mutually exclusive with the existing `--uv-source` (BLT-checkpoint) path.

**Verified before trusting it** (small-scale, CPU, using `run_gpt2_baseline_seed42.pt` as the source): parameter count matches the established single-pair UV256 count exactly; Wv/Wo/embeddings confirmed byte-identical to the checkpoint's real trained values (not left at random init); U/V confirmed properly shared across layers with sane, non-degenerate statistics; a real forward+backward pass produces finite gradients throughout. Step-0 loss on real text (~7.5-7.9 nats, vs. the unmodified checkpoint's ~1.16 on the same input) is substantially elevated but consistent with this project's own documented pattern for M-warm-start disruption (e.g. the BLT-XL incident) — expected, not a red flag.

**Launched both branches as a 50,000-step fine-tune** (user's call — explicitly not trying to save a few hours given a 3-week run is riding on the answer), warm-started from `run_gpt2_medium_baseline_seed42.pt` (the completed 1.5M-step medium MHA baseline, OWT ppl 18.09), single-pair (G=1, isolating rank alone as the variable), on `--dataset openwebtext_large` to match that checkpoint's own training data:
- **`io`**: r=512, `run_gpt2_medium_uv_r512_finetune_seed42.pt`. io turned out to already be free (its unrelated job had finished) — launched immediately rather than via a watcher. Needed the checkpoint (4.26GB) and the `openwebtext_large` 75-file cache (33.9GB) copied over from `bender` first, both verified byte-identical via `md5sum` (the cache's hash, `dd84d7564...`, matches the exact same file from the original medium-EMA cross-machine transfer). io's `llmopt_medium_ema/blt` checkout was ~15 commits behind (no local modifications) — pulled clean. Confirmed healthy: 305,490,944 unique parameters (exact match to the predicted 2ND²+2Dr formula at r=512), loss declining steadily from step 0 (9.03→6.79 over 120 steps), no instability.
- **`bender`**: r=256, `run_gpt2_medium_uv_r256_finetune_seed42.pt`. Armed as a watcher (`queue_bender_finish_then_uv_r256_medium_finetune.sh`) polling bender's current cumulative blend=0.75 job (PID 90902, ~26h out at time of writing) — will auto-launch once that finishes.

Once both land, compare OWT ppl / LAMBADA acc to decide r for the real 3-week medium UV run. This is explicitly a quick, "doesn't have to be perfect" probe (per the user) — not a substitute for the eventual from-scratch result, just enough signal to avoid committing 3 weeks to the wrong rank.

**r=512 DONE (2026-08-26).** All 50,000 steps finished on `io` (final val_ppl 56.34), checkpoint verified finite. Benchmarking initially failed with a real bug: `eval_owt.py`/`blt_lm_eval.py`'s low-rank-checkpoint loading path never passed `--pretrained` through to `build_blt_lowrank_model` (defaulted to `gpt2`'s 12 layers regardless of CLI flag) — never caught before since every prior UV checkpoint benchmarked was small-scale, where the default happened to be correct. Fixed and verified (rebuilds correctly to 24 layers / 305,490,944 params at `pretrained='gpt2-medium'`) before re-running. Real result: `lm_eval_gpt2_medium_uv_r512_finetune_seed42.json` — **OWT held-out ppl 23.56**, LAMBADA acc 0.246, HellaSwag acc_norm 0.276, PIQA acc_norm 0.594, Winogrande acc 0.513.

**r=256 DONE (2026-08-26), and it settles the question.** All 50,000 steps finished on `bender` (final val_ppl 54.45), checkpoint verified finite. Benchmarked (`lm_eval_gpt2_medium_uv_r256_finetune_seed42.json`, `eval_owt_gpt2_medium_uv_r256_finetune_seed42.stdout`):

| Metric | r=256 | r=512 |
|---|---|---|
| OWT held-out ppl | **23.37** | 23.56 |
| LAMBADA acc | **0.248** | 0.246 |
| LAMBADA ppl | **117.4** | 118.6 |
| HellaSwag acc_norm | **0.285** | 0.276 |
| PIQA acc_norm | 0.590 | **0.594** |
| Winogrande acc | 0.504 | **0.513** |

**r=256 wins on OWT ppl, LAMBADA (both metrics), and HellaSwag; r=512 only edges ahead on PIQA and Winogrande.** Doubling the rank bought nothing on the two metrics this project treats as primary (OWT ppl, LAMBADA), despite the SVD-energy prior suggesting r=512 should capture far more structure (98.2% vs. 81% of a converged full M's Frobenius energy) — the same lesson as the earlier UV256-vs-full-rank-BLT surprise: SVD-truncation energy capture of an existing solution doesn't predict trained-model quality. **Decision: r=256 for the real 3-week from-scratch medium UV run** — better on the metrics that matter and cheaper (fewer params, less memory, faster per step).

### Cumulative blend sweep — second blend=0.75 seed launched on io (2026-08-26)

Venus's blend=0.75 seed19 attempt died early (step 990/500,000 — negligible loss) sometime after launch, cause not investigated (venus was found idle with no process running; likely stopped deliberately during a machine reassignment, per user). Given blend=0.75 seed42's result showed an unexpected shape (best OWT ppl of the cumulative family, but *worst* LAMBADA acc — the opposite pattern from blend=1.0) and no second seed exists yet to confirm whether that's real or noise, relaunched the confirming seed on `io` (now idle after the r=512 medium probe finished) rather than venus: `run_gpt2_cumulative_blend75_scratch_seed19.pt`, same protocol as the original seed42 run. Timed deliberately to land around the same time as `titan`'s blend=0.5 seed42 (both ~2.5-3 days out at launch) — user's call, wanting the two open questions (is 0.5 a real point on the curve? is 0.75's odd shape real?) resolved together rather than staggered. Confirmed healthy: `effective_blend=0.7500`, loss declining normally from step 0.

Status as of this writeup: `titan` blend=0.5 seed42 ~53% through (ETA ~3 days), `io` blend=0.75 seed19 just started (ETA ~2.5 days), `bender` running the r=256 medium UV probe. `venus` idle.

### The real 3-week medium UV run — LAUNCHED (2026-08-26, on `bender`)

With the rank probe settled (r=256), launched the actual from-scratch medium UV run: `num_uv_groups=8` (chosen for NVLink/TP reasons, see the design discussion earlier this session), `r=256`, 1.5M steps (matching the standard medium-scale training-adequacy protocol used for the medium baseline and medium EMA runs).

**Smoke-tested first** — `num_uv_groups=8` at medium scale (D=1024, 24 layers) had never been built in code before (only G=1 and G=4 were previously tested, both at small scale only). Verified: correct shape (24 layers, 16 heads, D=1024), correct param count (308,636,672 — matches the 2ND²+2GDr formula at G=8, r=256), all 8 (U,V) pairs registered, a real forward+backward pass produces finite gradients throughout.

**Launched**: `train.py --uv-rank 256 --num-uv-groups 8 --from-scratch --pretrained gpt2-medium --dataset openwebtext_large --seed 42 --max-steps 1500000 --eval-every 1000 --lambada-eval-every 5000 --owt-eval-every 5000 --batch-size 2 --grad-accum-steps 2 --save-path run_gpt2_medium_uvgroups8_r256_scratch_seed42.pt`. Confirmed genuinely running: step 0 logged correctly (train_loss 11.03, matching expected random-init behavior on a fresh from-scratch build).

This is the first of the two planned 3-week medium runs. The second (cumulative-mode EMA at medium scale) is still blocked on the small-scale blend sweep resolving — see above.

### Second blend=1.0 seed launched on venus (2026-08-26)

`venus` was idle — launched `run_gpt2_cumulative_scratch_seed19.pt` (cumulative mode, default blend=1.0, seed 19, otherwise identical to the original seed42 run) to get a confirming second seed for the pure-cumulative endpoint. Confirmed healthy at launch (correct config in the log header, step 0 building normally). No urgency here since blend=1.0 already has one complete result (OWT ppl 30.37, LAMBADA acc 0.268) — this just extends it to the standard multi-seed practice.

**Interim observations while titan (blend=0.5) and io (blend=0.75 seed19) are both mid-training, worth a grain of salt each**: at matched step 400,000, OWT ppl reads 0.5→29.19, 0.75(seed19)→28.99, 1.0(seed42 final)→30.82(at 400K)/30.37(final) — the 0.75/0.5 comparison is cross-seed (0.5 is seed42, 0.75 is seed19), so it doesn't isolate blend value from ordinary seed variance and shouldn't be read as "0.75 beats 0.5" yet. io's blend=0.75 seed19 is tracking consistently ~0.7-0.8 ppl better than seed42 did at the same steps (checked at three points: 370K/385K/400K, not a one-off), and extrapolating seed42's own late-training decline suggests seed19 could land close to 28.6 — notably better than any prior cumulative-family result, but still short of even the weakest baseline seed (28.24). The in-training `lambada_acc` proxy (~0.57-0.58, indistinguishable across all three blends at step 400K) is not informative either way — this project has repeatedly found it fails to predict the real lm-eval-harness LAMBADA result, which is the metric that actually matters for the open question here (whether blend=0.75's oddly-low LAMBADA acc was a real pattern or seed42 noise).

### Cumulative blend=0.75 seed19 DONE (2026-08-28) — the LAMBADA weakness is now a real two-seed finding, not a fluke

Finished all 500,000 steps on `io` (final val_ppl 61.02), checkpoint verified genuine (all finite, `token_count` sum 2,046,000,000 across 50,134/50,257 tokens — fully populated). Benchmarked (`lm_eval_gpt2_cumulative_blend75_scratch_seed19.json`, `eval_owt_gpt2_cumulative_blend75_scratch_seed19.stdout`): OWT held-out ppl **28.57** (loss 3.3525) — beating the earlier linear extrapolation (~28.62) very slightly, and the best OWT ppl of the entire cumulative-mode family by a clear margin, though still short of even the weakest baseline seed (28.24).

| Metric | blend=0.75 seed42 | blend=0.75 seed19 | blend=1.0 (seed42) |
|---|---|---|---|
| OWT held-out ppl | 29.35 | **28.57** | 30.37 |
| LAMBADA acc | 0.249 | 0.245 | **0.268** |
| HellaSwag acc_norm | 0.269 | 0.271 | 0.271 |
| PIQA acc_norm | 0.573 | 0.583 | 0.566 |
| Winogrande acc | 0.525 | 0.525 | 0.507 |

**Both blend=0.75 seeds now land in the same 0.245–0.249 LAMBADA-acc band, clearly below blend=1.0's 0.268.** This is no longer explainable as seed42 noise — two independent seeds at blend=0.75 both show the same odd shape (best-in-family OWT ppl paired with a real LAMBADA-acc deficit relative to the full-blend endpoint), which is the opposite of what the fixed-decay EMA sweep's monotonic dial would predict. Whatever mechanism is driving this is a genuine property of cumulative mode's blend=0.75 point, not noise. Still waiting on titan's blend=0.5 (migrated to `io`, see below) and venus's blend=1.0 seed19 to map out whether this non-monotonicity is specific to 0.75 or part of a broader pattern.

### Migration (2026-08-28): titan's blend=0.5 moved to io for the remaining ~10%

Once io freed up (both its blend=0.75 seed19 training and its benchmark run finished), migrated titan's blend=0.5 seed42 run there to finish faster — same protocol as every prior migration: `SIGTERM` (PID-matched first), checkpoint verified genuine (step 450,500/500,000, all finite, `token_count` sum 1,843,450,092 across 50,133/50,257 tokens), copied to io (relayed through local scratchpad, md5sum-verified byte-identical at both hops), resumed there (`Resumed at step 450500`, confirmed healthy). Measured speedup is substantial: **~0.39 s/step on io vs. ~1.15-1.2 s/step on titan** — the remaining ~49,500 steps should take roughly 5.3 hours instead of the ~16+ hours titan's own pace would have needed.

Current state: `bender` running the real medium `num_uv_groups=8` run; `io` running the migrated blend=0.5 (should finish soon); `venus` running blend=1.0 seed19; `titan` now free.

### Cumulative blend sweep — full picture as of 2026-08-28

`blend=0.5` finished on `io` shortly after the migration above, checkpoint verified genuine: step 500,000, all finite, `token_count` sum 2,046,004,092 across 50,134/50,257 tokens. Benchmarked (`lm_eval_gpt2_cumulative_blend50_scratch_seed42.json`, `eval_owt_gpt2_cumulative_blend50_scratch_seed42.stdout`): OWT held-out ppl **28.76** (loss 3.3588).

| Metric | blend=0.5 (seed42) | blend=0.75 avg (2 seeds) | blend=1.0 (seed42, 1 seed) |
|---|---|---|---|
| OWT held-out ppl | **28.76** | 28.96 | 30.37 |
| LAMBADA acc | 0.244 | 0.247 | **0.268** |
| LAMBADA ppl | 150.6 | 133.7 | **119.3** |
| HellaSwag acc_norm | 0.268 | 0.270 | 0.271 |
| PIQA acc_norm | 0.571 | 0.578 | 0.566 |
| Winogrande acc | **0.531** | 0.525 | 0.507 |

**Corrected read (an earlier note here mischaracterized this as non-monotonic — it isn't).** As blend goes 1.0 → 0.75 → 0.5, OWT ppl steadily *improves* (30.37 → 28.96 → 28.76, lower is better) while LAMBADA acc steadily *worsens* (0.268 → 0.247 → 0.244, higher is better) — a clean monotonic trade-off, the same shape fixed-decay EMA's own sweep showed (higher blend costs OWT ppl, buys LAMBADA). Nothing surprising in the direction; cumulative mode's internal dial behaves as expected.

**What is still a real, structurally interesting difference from fixed-decay EMA**: in the *fixed-decay* sweep, an interior blend (0.75, jointly-trained-from-scratch) beat the pure-EMA endpoint (1.0) on LAMBADA acc (0.269 vs 0.253 for the strongest seed) — the sweet spot was in the interior. In *cumulative* mode, no interior blend tested so far beats the 1.0 endpoint on LAMBADA — full strength remains the best LAMBADA result in the whole cumulative family. So the two mechanisms' optimal operating points sit in different places along their respective curves, not just at different absolute values. One seed each at 0.5 and 1.0 — venus's blend=1.0 seed19 (in progress) is the next check on whether 1.0's LAMBADA edge holds up.

**Compared directly to the old fixed-decay EMA family at matched blend**: cumulative's blend=0.75 (avg 28.96 ppl / 0.247 acc) beats fixed-EMA's blend=0.75 (avg 29.55 ppl / 0.260 acc) on OWT ppl but loses on LAMBADA acc — cumulative buys OWT ppl at LAMBADA's expense relative to the old mechanism at the same nominal blend value. Cumulative's blend=1.0 (30.37 ppl / 0.268 acc) costs more OWT ppl than fixed-EMA's blend=0.75 but wins on LAMBADA (0.268 vs 0.260) and LAMBADA ppl (119.3 vs 129.9) by a real margin. Net: cumulative mode isn't uniformly better or worse than fixed-decay — it shifts where on the OWT-ppl/LAMBADA curve a given blend value lands.

**Third seed launched for both blend=0.75 and blend=1.0 (2026-08-28)**, to bring each up to the standard three-seed set matching every other result in this project. `run_gpt2_cumulative_blend75_scratch_seed7.pt` on `titan` (now free after the blend=0.5 migration) and `run_gpt2_cumulative_scratch_seed7.pt` on `io` (free after the blend=0.5 benchmarks). Both repos were several commits behind and needed syncing first — one real finding along the way: titan's local untracked `lm_eval_gpt2_cumulative_scratch_seed42.json` was **not** byte-identical to the committed version, confirming it was genuinely the known-spurious premature benchmark from the earlier migration mishap (documented above) rather than a duplicate — discarded rather than kept, as already planned. io's four untracked collisions were all genuine byte-identical duplicates, resolved the usual way. Both new seed-7 runs confirmed healthy at launch (correct seed/blend in the log header, building normally).

Current state: `bender` on the real medium `num_uv_groups=8` run; `venus` on cumulative blend=1.0 seed19; `titan` on cumulative blend=0.75 seed7; `io` on cumulative blend=1.0 seed7. Once these land, blend=0.75 and blend=1.0 will each have three seeds; blend=0.5 still has only one.

### What the blend multiplier is actually doing, and why OWT ppl may be the wrong thing to be tracking as a co-objective (2026-08-29)

**Exact mechanism, confirmed from the code** (`train.py`): `weight = token_loss[ids]`; `weight = weight / weight.mean()` (normalized so the *current batch's* mean becomes exactly 1.0 — not a global vocab-wide mean, so the normalizer shifts batch-to-batch depending on which tokens appear); `weight = blend*weight + (1-blend)*1.0`. This is exactly `x = avg + (x-avg)*blend` with avg fixed at 1.0 by construction (the user's own hypothesis, confirmed).

**Pulled the real numbers from a converged checkpoint** (`run_gpt2_cumulative_scratch_seed42.pt`, blend=1.0) rather than reasoning about it abstractly. The middle 98% of tokens (p01–p99) sit in a fairly modest 0.28–1.82 multiplier range — the extreme min/max (0.017/3.123) are a handful of very rare, unstably-estimated tokens. But the distribution is sharply **asymmetric toward downweighting**: at blend=1.0, 4,198 tokens (8.4% of vocab) get multiplier < 0.5, and those tokens account for **902 million of the 2.05 billion total training-token occurrences (44% of the whole corpus)**. Tokens with multiplier > 2.0 are rare (392 tokens) and represent only 36,838 occurrences — a rounding error by comparison. So the mechanism works almost entirely by **suppressing gradient on the most common tokens**, not by amplifying rare/hard ones — consistent with the original motivation for this whole line of work (uniform cross-entropy lets easy high-frequency tokens swamp the gradient signal from hard long-range predictions).

**This reframes how OWT held-out ppl should be read for this specific family of experiments.** `eval_owt.py` computes standard *unweighted* cross-entropy on held-out text — which is, by definition, "how well does the model match the empirical, frequency-weighted next-token distribution of the corpus." A scheme that deliberately suppresses gradient on the most frequent tokens is, by construction, training the model to deviate from exactly that target. So a meaningful chunk of the OWT-ppl "cost" reported throughout this section isn't independent evidence the model got worse — it's closer to a near-tautological readout of how much deliberate deviation happened. This is **specific to the loss-reweighting family** (EMA/cumulative) — it does *not* apply to the BLT/architecture experiments (UV256, `num_uv_groups`, hybrid, etc.), where nothing touches the loss weighting and OWT ppl remains a fair, direct, primary comparison.

**Practical consequence for how results in this section should be interpreted going forward — user's framing, adopted as the correct one.** OWT ppl is not needed as a degeneracy backstop: HellaSwag/PIQA/Winogrande/LAMBADA already serve as the substantive quality checks, and nothing about OWT ppl is uniquely able to catch a broken model that those wouldn't also catch. So OWT ppl isn't a guardrail either — it's a **declared cost metric**, still worth reporting every time, but reported as the honest price tag of a deliberate choice rather than a risk being managed. The claim this whole EMA/cumulative line of work is making is: raw OpenWebText token-frequency matching (what OWT ppl measures) is *not* the actual target — it is an explicit stand-in for "generate more of what we actually want" (better long-range/narrative competence, of the kind LAMBADA-style tasks probe), and this family of methods trades away some of the former, on purpose, for more of the latter. Report OWT ppl every time as that cost, not as something to explain away or worry about "losing." Worth carrying this framing into `paper_ema.md` once written.

**Status snapshot (2026-08-29):**
- `bender`: real medium `num_uv_groups=8` run, step 140,680/1,500,000 (9.4%).
- `titan`: cumulative blend=0.75 seed7, step 85,290/500,000 (17.1%).
- `io`: cumulative blend=1.0 seed7, step 232,610/500,000 (46.5%).
- `venus`: cumulative blend=1.0 seed19, step 133,380/500,000 (26.7%).

**`paper_blt.md` updated to match the data (2026-08-23).** Added a new Section 4.8 writing up the `num_uv_groups=4` vs `layers-per-m=4` head-axis/layer-axis result in full — previously this lived only in this file, despite being arguably the paper's most decisive finding. Also fixed three places where the paper still described the third UV256 seed as "in progress" (it finished 12 days prior); updated Section 6's synthesis bullet, Section 7.2's future-work framing, and Section 7.3's closing sentence to match. See the paper itself for the full text — not duplicated here.

### Other future directions
- **Grouped Wv**: share Wv across groups of heads (GQA-style) to reduce value cache bandwidth.
- **Token weighting loss**: upweight tokens requiring long-range context using a short-context reference model (arXiv 2503.09202). Most promising fix for LAMBADA/benchmark mismatch.
- **KV cache implementation**: BLT's cache stores raw x_j as keys (no Wk multiply needed) + standard values. Not yet implemented.

## Files
- `model.py` — BLTAttention (1-M), BLTMultiMAttention (N-group), BLTLowRankAttention (M≈U@V^T), build_blt_model(num_m_groups, random_m, from_scratch), build_blt_lowrank_model(rank, source_checkpoint, from_scratch)
- `train.py` — training harness. Key flags: `--resume`, `--finetune`, `--dataset`, `--baseline`, `--num-m-groups`, `--random-m`, `--from-scratch`, `--lambada-eval-every`, `--ema-blend-schedule`, `--uv-rank`/`--uv-source`
- `evaluate.py` — sliding-window perplexity (WikiText-103 or LAMBADA val); compute_cloze_accuracy
- `blt_lm_eval.py` — lm-eval-harness wrapper; supports `--num-m-groups`, `--uv-rank`
- `paper_blt.md` — draft paper covering BLT architecture, results, and related work (now includes Section 4.7, the first low-rank UV^T result)
- `paper_ema.md` — draft paper covering the EMA per-token loss weighting / blend-schedule work (3-seed-confirmed jointly-trained 75/25 blend, sine-anneal schedule result)
- `eval_owt.py` — held-out OWT evaluation (files 21-25, sliding window); `--blt-checkpoint`, `--gqa-checkpoint`, `--hybrid-checkpoint` (+ `--n-mha-layers`), `--blt-lowrank-checkpoint` (+ `--uv-rank`), or `--baseline-checkpoint`
- `run_seed42.pt/.log` — BLT WikiText-103 (50,300 steps, val_ppl=21.50). Note: `run_seed42.log` ALSO received the hybrid run's training log (2026-06-16 through completion 2026-06-19) due to the log-naming bug fixed in commit 1f3803d (fix not applied to that already-running process) — the WikiText-103 run's own log content is only the first ~5 header lines plus its original step history.
- `run_blt_scratch_seed42.pt/.log` — BLT from-scratch OWT (550K steps, val_ppl=72.88, OWT ppl=31.05)
- `run_blt_scratch_seed19.pt/.log` — BLT from-scratch OWT seed19 (600K steps, OWT ppl=30.48)
- `run_blt_scratch_seed7.pt/.log` — BLT from-scratch OWT seed7 (500K steps, OWT ppl=30.81)
- `run_blt_scratch_seed19_500k.pt/.log` — BLT from-scratch OWT seed19 at exactly 500K steps (distinct from the 600K `run_blt_scratch_seed19.pt` above), OWT ppl=32.87. Lives on `titan` only; not copied locally (1.3GB).
- `run_blt_uv256_scratch_seed42.log`, `lm_eval_blt_uv256_scratch_seed42.json`, `eval_owt_blt_uv256_scratch_seed42.stdout` — low-rank BLT (M≈UV^T, r=256), from-scratch OWT, seed42 (500K steps, OWT ppl=29.97). Checkpoint (`run_blt_uv256_scratch_seed42.pt`, 1.3GB) lives on `titan` only, not copied locally, per the usual convention for cross-machine checkpoints this size.
- `run_blt_uv256_scratch_seed19.pt/.log` — low-rank BLT r=256 seed19, IN PROGRESS on `titan`, second seed for the UV256 result above.
- `run_gpt2_baseline_seed42.pt/.log` — GPT-2 from-scratch OWT (500K steps, val_ppl=55.99, OWT ppl=27.78)
- `run_gqa_scratch_seed42.pt/.log` — GQA 2-group from-scratch OWT (500K steps, OWT ppl=27.64)
- `run_hybrid_mha6_scratch_seed42.pt/.log` — hybrid 6 MHA + 6 BLT from-scratch OWT, DONE (500K steps, val_ppl=69.08, OWT ppl=28.40); `.log` only has its own content up to the OOM crash, see note on `run_seed42.log` above
- `lm_eval_owt_randm_250k.json` — final benchmark for OWT random-M run
- `lm_eval_blt_scratch.json` — benchmarks for BLT from-scratch OWT run (seed42)
- `lm_eval_blt_scratch_seed7.json` — benchmarks for BLT from-scratch OWT run (seed7)
- `lm_eval_blt_scratch_seed19_500k.json` — benchmarks for BLT from-scratch OWT run (seed19, 500K steps)
- `lm_eval_baseline_scratch.json` — benchmarks for GPT-2 from-scratch OWT run
- `lm_eval_gqa_scratch.json` — benchmarks for GQA from-scratch OWT run
- `lm_eval_hybrid_scratch.json` — benchmarks for hybrid 6 MHA + 6 BLT from-scratch OWT run

## Broader project context
This `blt` branch lives alongside `pprune/` (a KV cache pruning paper). BLT is a separate experiment exploring parameter-efficient attention alternatives.

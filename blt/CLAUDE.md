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

### Current status snapshot (2026-07-16) — read this first for hand-off
- **`bender`** (local): GPT-2 **medium** (355M, 24 layers/16 heads/1024 dim) baseline from-scratch OWT, seed 42, running — `run_gpt2_medium_baseline_seed42.pt`, target **1.5M steps**, now on `--dataset openwebtext_large` (single-epoch 75-file corpus, no repetition) after two rounds of OOM debugging in the dataset loader (see "GPT-2 medium" below for the full story — model-size code changes, batch-size fix, thermal-throttling analysis, the resume data-replay bug, the corpus-repetition fix, and the two OOM bugs in `openwebtext_large`'s loader). Confirmed healthy past step 100 as of 2026-07-17 evening.
- **`titan`**: GPT-2 baseline (standard CE, no EMA) from-scratch OWT, seed 19, running (`run_gpt2_baseline_seed19.pt`) — third baseline seed to match BLT's 42/19/7. A watcher (`queue_blt_seed19_500k_watcher.sh`, PID recorded in its own log) is waiting for this to finish, then will launch a genuine BLT from-scratch seed19-at-500K run (`run_blt_scratch_seed19_500k.pt`) — see "BLT seed19-at-500K bookkeeping" below for why this is needed.
- **`venus`** (new machine as of 2026-07-11, see Known Machines): third genuine 75/25 EMA-blend seed replicate, seed 7, running (`run_gpt2_ema_blend75_scratch_seed7_v2.pt`) — verified genuine via checkpoint `ema_loss` buffer at step 2,000 (std 0.775, not the seed7-bug's std=0.0).
- **Two genuinely completed 75/25-blend seeds now exist** (seed42: LAMBADA acc 0.269, seed19: 0.253 — both well above the non-EMA baseline's 0.225); seed7_v2 above is the third, in progress.
- **`io`**: GPT-2 **medium** with EMA 75/25-blend (`run_gpt2_medium_ema_blend75_seed42.pt`), seed 42, 1.5M-step target, running — the EMA counterpart to `bender`'s non-EMA baseline. Fresh clone at `/home/benzene/llmopt_medium_ema/blt` (not the older, in-flux checkouts). Verified genuine via checkpoint `ema_loss` buffer at step 1,000. See "GPT-2 medium" below for the io-specific cache-build saga (its own tokenization attempt ballooned to 77GB+ VSZ despite the fix that worked cleanly on bender — worked around by copying bender's already-built cache instead of debugging further).

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

### BLT seed19-at-500K bookkeeping (resolved 2026-07-13)
Investigating a discrepancy in `run_blt_scratch_seed42_500k.json` on `titan` (different LAMBADA/HellaSwag/PIQA/Winogrande numbers than the committed version under the same filename) surfaced a cluster of confusing but ultimately-explained loose ends:
- **titan's `run_blt_scratch_seed42_500k.pt`/.json is a different, unexplained checkpoint** — internally tagged seed 42, genuinely completed its own 500K-step schedule (confirmed via `scheduler_state`: `last_epoch=500000`, saved LR exactly `0.0`, ruling out "mid-run snapshot of the 550K run"), standard single-shared-M architecture, but final val_ppl 49.16 doesn't match any documented seed42 run (550K→72.88, the canonical 500K→76.64). Preserved as `lm_eval_blt_scratch_seed42_500k_titan_version.json` rather than overwritten. **Not fully explained — left as an untraceable artifact, not used for anything.**
- **A separate, previously-undocumented local file set was found**: `run_blt_scratch_seed19_500k.pt/.log/.stdout` (dated 2026-07-06, right in the bender-overheating window) — internally seed 19, from-scratch, targeting 500K steps, but **killed at step 910** and never resumed. This matches the user's memory of a seed19 attempt killed early due to overheating. **This is real but incomplete — not usable data.**
- **Net effect**: BLT has never had a genuine, complete, from-scratch 500K-step seed19 run. It has `run_blt_scratch_seed19.pt` (600K steps, DONE, OWT ppl 30.48) and the killed 910-step attempt above, but nothing at exactly 500K to match seed7 (30.81) and seed42 (31.69) for an iso-step three-way comparison — mirroring what GPT-2 baseline now has (seed42, seed7, and seed19 in progress on `titan`). **Fix queued**: watcher on `titan` launches this the moment the GPT-2 baseline seed19 run finishes (see status snapshot above).

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

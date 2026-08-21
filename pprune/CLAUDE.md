# pprune — KV Cache Compression Papers

## Project overview

Two papers split from a single codebase:

**`paper_kv_faithfulness.tex`** — Primary paper (LaTeX source). KV cache compression faithfulness:
positional displacement is the dominant failure mode for SnapKV and PyramidKV. Key re-rotation
(SnapKV+rot) fixes this: 27× KL improvement at 65%, best method overall. Pyr+rot matches
SnapKV+rot at 65% (0.047 vs 0.043) and ties exactly at 35% (budget collapse to uniform); a gap
opens only at 50% where the pyramid reaches its window_size floor at layer 31. No phrase
compression content. Compile with XeLaTeX (requires fontspec + unicode-math).

**`paper_phrase_compression.md`** — Base for future phrase compression paper. Contains all
phr128/phr160 content, §7 Phrase-Based Context Compression, and the TTFT comparison. Not the
active submission.

Branch: `kvpress-impl`
Models: `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`
Data: `lb_data_raw/data/*.jsonl` (16 LongBench tasks, n=200 for gov_report/multi_news, n=100 others)

## Key methods (as they appear in the paper)

| Name in paper | Method string | Notes |
|---|---|---|
| Naive | `naive_65pct` / `naive_35pct` etc. | Prompt truncation, head_frac=0.10 |
| phr160 | `chunk_word160_t25` | 160-tok phrases, 25% tail, word overlap scoring |
| phr128 | `chunk_word128_t20` | 128-tok phrases, 20% tail, word overlap scoring |
| SnapKV | `snapkv_press` / `snapkv_press_f35` etc. | kvpress SnapKVPress, post-prefill eviction |
| SnapKV+rot | `snapkv_rerotated` / `snapkv_rerotated_f35` etc. | SnapKV + KeyRerotationPress |
| PyramidKV | `pyramidkv` / `pyramidkv_f35` etc. | kvpress PyramidKVPress |
| Pyr+rot | `pyramidkv_rerotated` / `pyramidkv_rerotated_f35` etc. | PyramidKV + PyramidKVRerotationPress (per-layer budget preserved) |
| Streaming | `streaming_rerotated` / `streaming_rerotated_f35` etc. | kvpress KeyRerotationPress (corrected) |

SnapKV-Select (`snapkv_select`) is a diagnostic tool only — §6.2 only.

**RADAR methods** (implemented in `kl_faith_eval_ystar.py`, dropped from paper — selection irrelevance claim removed):
| Name | Method string | Notes |
|---|---|---|
| RADAR pre-rope | `radar_pre_press` / `_f50` / `_f35` | Pre-RoPE max-dot scoring, no rerotation |
| RADAR post-rope | `radar_post_press` / `_f50` / `_f35` | Post-RoPE max-dot scoring, no rerotation |
| RADAR pre+rot | `radar_pre_rerotated` / `_f50` / `_f35` | Pre-RoPE scoring + KeyRerotationPress |
| RADAR post+rot | `radar_post_rerotated` / `_f50` / `_f35` | Post-RoPE scoring + KeyRerotationPress |

Rate suffix convention: no suffix = 65%, `_f50` = 50%, `_f35` = 35%. **f40 dropped from paper.**

## Eval harnesses

- `gt_eval_compression.py` — Ground-truth LongBench accuracy (full generation). Resumable checkpoint
  keyed by `task|idx|method`. Output dirs under `lb_results_base/gt_*/`.
  **IMPORTANT**: `generate_rerotated()` (added Jul 2026) handles KeyRerotationPress methods AND
  `PyramidKVRerotationPress`. Without it, decode position IDs are T+1, T+2, ... instead of
  M+1, M+2, ..., causing degenerate "In In In..." output. KL eval (teacher-forced) is NOT affected.
  **For `PyramidKVRerotationPress`**: right-aligned re-rotation maps each layer's keys to
  `[T−n_kept, T)`, so all layers share endpoint T−1 and decode begins at T (matching the full
  model). `generate_rerotated()` uses `M = T` for `PyramidKVRerotationPress` instances. GT
  results for Pyr+rot with right-aligned implementation are valid.
- `kl_faith_eval_ystar.py` — KL faithfulness using y* shared prefix (teacher-forced). All
  `_KVPRESS_METHODS` defined here including rerotated variants.
- `kl_faith_eval.py` — Core configs (METHOD_CONFIGS, CHUNK_CONFIGS). Imported by both above.
- `longbench_eval.py` — Scoring functions. `qa_f1_score()` used for F_out computation.
- `queue_runner.sh` — File-based job queue. Reads `lb_results_base/queue.txt`, runs first
  non-comment line, removes it, repeats. Edit queue.txt live to reorder/add jobs.

## Key data files

| File | Contents |
|---|---|
| `lb_results_base/checkpoint.json` | Llama: full, naive, snapkv, streaming, etc. All 16 tasks, n≥100 |
| `lb_results_base/gt_comp/checkpoint.json` | phr128/phr160 + snapkv/radar at 50/40/35%, 9 tasks, n=100 |
| `lb_results_base/gt_comp_remaining/checkpoint.json` | Same methods, remaining 7 tasks, n=100 |
| `lb_results_base/gt_pyramidkv/checkpoint.json` | pyramidkv (65%), all 16 tasks, n=100 |
| `lb_results_base/gt_pyramidkv_comp/checkpoint.json` | pyramidkv at f50/f40/f35, all 16 tasks, n=100 |
| `lb_results_base/gt_mistral_table/checkpoint.json` | Mistral: old pcfg snapkv/streaming, pyramidkv, 65%, n=100 |
| `lb_results_base/kl_ystar_tables_v3.json` | KL for prompt-construction + old snapkv/streaming, all 16 tasks, n=100 |
| `lb_results_base/kl_ystar_pyramidkv_all_v2.json` | KL for pyramidkv at all rates, all 16 tasks, n=100 |
| `lb_results_base/ystar_cache_v3.pt` | Cached y* tokens + log_p_full for all 16 tasks, n=100 (Llama) |
| `lb_results_base/ystar_cache_mistral.pt` | Same for Mistral |
| `lb_results_base/kl_mechanism_llama_f65.json` | KL: snapkv_press + streaming_rerotated, Llama 65% ✓ |
| `lb_results_base/kl_mechanism_llama_f50.json` | Same at 50% ✓ |
| `lb_results_base/kl_mechanism_llama_f35.json` | Same at 35% ✓ |
| `lb_results_base/kl_mechanism_mistral_f*.json` | Same for Mistral, all 3 rates ✓ |
| `lb_results_base/gt_mechanism_llama_f65/checkpoint.json` | GT: snapkv_press ✓, streaming_rerotated ✓ |
| `lb_results_base/gt_mechanism_llama_f50/checkpoint.json` | GT: snapkv_press ✓, streaming_rerotated ✓ |
| `lb_results_base/gt_mechanism_llama_f35/checkpoint.json` | GT: snapkv_press ✓, streaming_rerotated ✓ |
| `lb_results_base/gt_mechanism_mistral_f65/checkpoint.json` | GT: snapkv_press ✓, streaming_rerotated ✓ |
| `lb_results_base/gt_mechanism_mistral_f50/checkpoint.json` | GT: snapkv_press ✓, streaming_rerotated ✓ |
| `lb_results_base/gt_mechanism_mistral_f35/checkpoint.json` | GT: snapkv_press ✓, streaming_rerotated ✓ |
| `lb_results_base/kl_instruct_pilot.json` | KL for snapkv_press + snapkv_rerotated + streaming_rerotated on Llama-3.1-8B-Instruct, 65%, 6 tasks, n=100 ✓ |
| `lb_results_base/ystar_cache_instruct.pt` | Cached y* tokens + log_p_full for Instruct model, 6 tasks, n=100 |
| `lb_results_base/ystar_instruct_llama.pt` | Cached y* tokens + log_p_full for Instruct model, 6 tasks, n=100 (used by b256/b1024 evals) |
| `lb_results_base/kl_instruct_llama_b256.json` | KL: all methods, Llama-3.1-8B-Instruct, budget=256, ws=8 (snap/pyr), ws=32 (streaming/naive), 6 tasks, n=100 ✓ |
| `lb_results_base/gt_instruct_llama_b256/checkpoint.json` | GT: all methods, Llama-3.1-8B-Instruct, budget=256, ws=8, 16 tasks, n=100 ✓ |
| `lb_results_base/perstep_kl_rerotated.json` | Per-step KL for snapkv_rerotated + pyramidkv_rerotated on gov_report, n=31, 512 steps (used in Fig 5) |

## Completed data

**KL (Llama):**
- `kl_snapkv_rerotated.json` — 65%, all 16 tasks ✓ (mean KL=0.043)
- `kl_snapkv_rerotated_f50.json` — 50% ✓ (mean KL=0.077)
- `kl_snapkv_rerotated_f35.json` — 35% ✓ (mean KL=0.124)
- `kl_snapkv_rerotated_f40.json` — 40% ✓ (f40 dropped from paper)
- `kl_pyramidkv_rerotated.json` — 65%, all 16 tasks ✓ (mean KL=0.047; right-aligned)
- `kl_pyramidkv_rerotated_f50.json` — 50%, all 16 tasks ✓ (mean KL=0.096; pyramid at window_size floor)
- `kl_pyramidkv_rerotated_f35.json` — 35%, all 16 tasks ✓ (mean KL=0.124; budget collapse → ties SnapKV+rot)
- `gap_structure.json` — 4 geometries × 4 presentations, 6 tasks, n=20 ✓
- `kl_instruct_pilot.json` — Llama-3.1-8B-Instruct, 65%, 6 tasks, n=100 ✓ (snapkv_press=0.94, snapkv_rerotated=0.039, 24×; cited in §5.2)
- `kl_instruct_llama_b256.json` — Llama-3.1-8B-Instruct, budget=256, 6 tasks, n=100 ✓ (snapkv=0.769, snapkv_rot=0.352, pyramidkv=0.882, pyramidkv_rot=0.385, streaming=1.133, streaming_rot=0.736, naive=2.003)

**KL (Mistral):**
- `kl_snapkv_rerotated_mistral.json` — 65%, all 16 tasks ✓ (mean KL=0.030)
- `kl_snapkv_rerotated_mistral_f50.json` — 50% ✓ (mean KL=0.058)
- `kl_snapkv_rerotated_mistral_f35.json` — 35% ✓ (mean KL=0.096)
- `kl_pyramidkv_rerotated_mistral.json` — 65%, all 16 tasks ✓ (mean KL=0.034; matches Llama pattern: 0.034 vs SnapKV+rot 0.030)
- `kl_pyramidkv_rerotated_mistral_f50.json` — 50%, all 16 tasks ✓ (mean KL=0.079)
- `kl_pyramidkv_rerotated_mistral_f35.json` — 35%, all 16 tasks ✓ (mean KL=0.096; ties SnapKV+rot exactly on all 16 tasks — budget collapse confirmed on Mistral)

**GT (Llama):**
- `gt_snapkv_rerotated/` — 65%, all 16 tasks ✓ (1600/1600; position-ID fix applied)
- `gt_snapkv_rerotated_f50/` — 50%, all 16 tasks ✓ (1600/1600)
- `gt_snapkv_rerotated_f35/` — 35%, all 16 tasks ✓ (1600/1600)
- `gt_pyramidkv_rerotated/` — 65%, all 16 tasks ✓ (1600/1600; right-aligned re-rotation; F_out avg=71.6%)
- `gt_pyramidkv_rerotated_f50/` — 50%, all 16 tasks ✓ (1600/1600; F_out avg=61.3%)
- `gt_pyramidkv_rerotated_f35/` — 35%, all 16 tasks ✓ (1600/1600; F_out avg=63.0%)

**GT (Mistral):**
- `gt_snapkv_rerotated_mistral/` — 65% ✓ (1600/1600; avg F_out=78.2, avg GT=22.8)
- `gt_snapkv_rerotated_mistral_f50/` — 50% ✓ (1600/1600; avg F_out=74.3, avg GT=22.7)
- `gt_snapkv_rerotated_mistral_f35/` — 35% ✓ (1600/1600; avg F_out=70.1, avg GT=22.3)
- `gt_pyramidkv_rerotated_mistral/` — 65% ✓ (1600/1600; avg F_out=77.0, avg GT=22.9)
- `gt_pyramidkv_rerotated_mistral_f50/` — 50% ✓ (1600/1600; avg F_out=66.5, avg GT=22.1)
- `gt_pyramidkv_rerotated_mistral_f35/` — 35% ✓ (1600/1600; avg F_out=70.0, avg GT=22.3)

## Currently running

**Llama-3.1-8B-Instruct — all complete (all 16 tasks):**
- `kl_instruct_llama_b256.json` ✓ (6 tasks, original; numbers cited in §9)
- `kl_instruct_llama_b1024.json` ✓ (16 tasks; snapkv_rot=0.146, snapkv=0.648, pyr=0.652, pyr_rot=0.174)
- `gt_instruct_llama_b256/checkpoint.json` ✓ (16 tasks, 100/100)
- `gt_instruct_llama_b1024/checkpoint.json` ✓ (16 tasks, 100/100)

**Mistral-7B-Instruct-v0.3 — in progress:**
1. `kl_instruct_mistral_b256.json` — **complete** (all 16 tasks). Mean KL: snapkv=1.912, snapkv_rot=0.614, pyr=1.872, pyr_rot=0.689, streaming=2.076, streaming_rot=1.207, naive=2.561 → Tab D7
2. `gt_instruct_mistral_b256/` — **complete** (all 16 tasks, 800 entries/task) → Tab D8 (F_out, verified correct), Tab D9 (GT, corrected Aug 2026: PassageCount/LCC/RepoBench-P were wrong). Avg GT: Full=32.8, SnapKV=31.3, Pyr=31.6 (best compressed)
3. `kl_instruct_mistral_b1024.json` — **complete** (all 16 tasks). Mean KL: snapkv=1.766, snapkv_rot=0.294, pyr=1.596, pyr_rot=0.349, streaming=1.912, streaming_rot=0.894, naive=1.532 → Tab D10 (added, SnapKV+rot leads all 16 tasks)
4. `gt_instruct_mistral_b1024/` — **running** (started 2026-08-21; narrativeqa in progress) → Tab D11 (F_out), D12 (GT) pending

## Key empirical results (Llama, F_out macro over 16 tasks)

| Method | 65% | 50% | 35% |
|---|---|---|---|
| Pyr (no rot) | 79.1 | 64.2 | 66.9 |
| SnapKV (no rot) | 78.5 | 73.2 | 66.8 |
| SnapKV+rot | 71.1 | 67.8 | 63.0 |
| Pyr+rot | 71.6 | 61.3 | 63.0 |
| Naive | 68.6 | 52.1 | 46.9 |
| Streaming (corrected) | 61.2 | 56.4 | 51.8 |
| phr128 | 54.5 | 52.8 | 48.6 |

## Paper structure notes (paper_kv_faithfulness.tex)

Paper split complete as of Jul 2026. `paper_kv_faithfulness.tex` is the active submission.
Section numbering: §1–§6 unchanged, §7 Main Experiments, §8 Why Post-Prefill KV Eviction...,
§9 High-Compression Results on Instruction-Tuned Model, §10 Conclusion.
"post-hoc" replaced with "post-prefill" throughout.

- §5.2: KL faithfulness metric. Includes Instruct model generalization paragraph: Llama-3.1-8B-Instruct
  reproduces base model finding (snapkv_press 0.94→0.039 nats, 24×) on same 6 tasks (`kl_instruct_pilot.json`).
- §6.1: Mechanism section restructured as itemize list (prefill enrichment / gap corruption / positional
  misalignment). Notes that re-rotation creates tension with pre-eviction enrichment.
- §6.2: SnapKV-Select diagnostic (KL only). Only place Sel appears. Includes explanation of why ROG
  (recompute-over-gaps) doesn't show rotation damage: without enrichment, scattered positions cause
  limited harm; enrichment × scatter interaction is what makes SnapKV catastrophic.
- §6.3: Synthetic gap structure table — 4 geometries × 3 presentations (gapless/evicted/rerotated).
  **Gapped presentation removed** (evaluation bug: y* tokens were evicted from KV cache during
  single-pass eval, inflating KL). Tab:t2 has 7 columns. Ends with prediction: SnapKV should
  benefit enormously from re-rotation; Streaming should see a meaningfully smaller gain. §7.2 confirms.
- §6.4: **Removed** — content absorbed into §7.2.
- §7.1: Setup (models, benchmark, methods table).
- §7.2: KL Faithfulness main results (Tab:t3). SnapKV+rot leads at every rate on both models.
  Right-aligned re-rotation explanation for PyramidKV (naive [0,n_kept) is inconsistent across
  layers; [T−n_kept,T) fixes it). Pyr+rot three-rate story. Streaming improvement table (Tab:t4-ish);
  value-corruption explanation for why Streaming trails SnapKV+rot after re-rotation. Confirms §6.3 predictions.
- §7.3: Output Faithfulness (F_out) results.
- §7.4: Inference Performance (TTFT / TPT).
- §8: First-token advantage mechanism; §8.1 short/long F_out breakdown (Table 5), KL by output length
  (Table 6), long-form F_out cross-rate (Table 7). Concludes with deployment implications.
- §9: High-Compression Results on an Instruction-Tuned Model — Llama-3.1-8B-Instruct at budgets 256
  and 1024 tokens. Four subsections: why moderate compression is more informative; instruct output
  style amplifies first-token advantage (F_out useless at extreme compression); SnapKV+rot best at
  every budget (KL: 0.352 at b=256, 0.146 at b=1024); deployment perspective with short/long
  F_out and KL tables (Tab:instruct-fout-shortlong, Tab:instruct-kl-shortlong) and timing table
  (Tab:instruct-timing). Mistral-Instruct results noted as pending. Appendix D has per-task results.
- §10: Conclusion — metric inversion (Pyr/SnapKV best F_out, worst KL), 27× KL improvement from
  re-rotation, Pyr+rot cross-layer disparity explanation, generalization claims, operational message.
- KL metric uses y* shared prefix (fixed path-dependence flaw); see `kl_faith_eval_ystar.py`.

## Data validity note

All **KL** data is valid — teacher-forced, unaffected by position-ID bug.

**GT data**: `generate_rerotated()` fix applied Jul 2026. All KeyRerotationPress GT checkpoints
created before the fix are invalid. Affected checkpoints have been purged and re-run:
- `gt_snapkv_rerotated*/` — purged and re-run ✓
- `gt_mechanism_*/` — streaming_rerotated entries purged and re-run ✓

**§4 paper GT tables fixed (Jul 2026)**: The §4 Streaming columns for both Llama and Mistral were
populated from pre-fix data (near-zeros on QA tasks; Llama avg=18.5, Mistral avg=17.8). Updated
to valid re-run values from `gt_mechanism_*/` checkpoints (Llama=23.7, Mistral=22.9). Bolding
corrected: Streaming now leads on NarrativeQA (both models), TriviaQA/SAMSum (Mistral),
SAMSum/2WikiMQA (Llama). The "nearly zeroing out fact-retrieval" claim in §4 narrative was
removed — it was based on corrupted output.

**PyramidKVRerotationPress fix (Jul 2026)**: `KeyRerotationPress(PyramidKVPress)` silently used
uniform budget (bypassed `get_layer_budget()`), making it identical to SnapKV+rot. Fixed by
`PyramidKVRerotationPress` subclass in `kl_faith_eval_ystar.py` that calls `get_layer_budget()`.
Right-aligned re-rotation (`[T−n_kept, T)` per layer) ensures all layers share endpoint T−1 and
decode begins at T — GT generation is valid for all tasks at all retention rates. Old
left-aligned `kl_pyramidkv_rerotated*.json` and `gt_pyramidkv_rerotated*/checkpoint.json` purged
and re-run with the right-aligned implementation.

`diagnose_distribution.py` (untracked) has a head-truncation bug — NOT used for paper tables.

## LaTeX formatting state (Aug 2026)

`paper_kv_faithfulness.tex` has been reformatted for arXiv single-column submission:

- **Layout**: Single-column (`\documentclass[10pt,letterpaper]{article}`); abstract uses `\normalsize`
- **Table placement**: All tables non-floating (`[H]` via `float` package); all `table*` converted to `table` (fixes label registration with `[H]`)
- **Paired tables**: All Llama/Mistral table pairs merged into a single float using `minipage` side-by-side layout with one shared caption and one footnote below the table (outside `\caption{}`)
- **Tab:t1** (§6.2, four-configuration KL): booktabs style — `\toprule`/`\midrule`/`\bottomrule`, no `|` vertical rules, no `\hline` between rows
- **Tab:methods** (§5): `p{}` column spec with description column at 62% of `\textwidth` (no wrapping)
- **Tab:t7** (§8.1): transposed — rates as rows, methods as columns with `\multicolumn`/`\cmidrule` grouped headers (SnapKV / PyramidKV)
- **Hyperparameters paragraph**: wrapped in `\begin{sloppypar}...\end{sloppypar}` to handle long `\texttt{}` sequences
- **Appendices**: `\clearpage` before Appendix B and Appendix C

## Paper edits completed (Aug 2026 session)

All edits are in `paper_kv_faithfulness.tex` on branch `kvpress-impl`.

**Three inline figures added** (commit 906a1e9):
- §5.2: TikZ eval-setup fork diagram (full model → y* → two teacher-forced branches → F_KL box)
- §6.1: TikZ RoPE displacement schematic (post-eviction strip with Δ=4 gap vs re-rotated compact Δ=1)
- §8.1: pgfplots per-step KL trajectory (PyramidKV solid red, SnapKV dashed blue, Naive dotted green; 50-step rolling mean over gov_report, n=32; data from `lb_results_base/perstep_kl_longbench.json`)
- Preamble additions: `\usetikzlibrary{calc}`, `\usepackage{pgfplots}`, `\pgfplotsset{compat=1.18}`

**Forward reference fixes** (commit e795e71): Removed results imported from §8 into §5 and §7. Soft pointers ("§8 investigates...") retained; hard citations of specific results removed.

**Undefined terms and hedging fixes** (commit dbe63c7): Defined RoPE and TTFT on first use in §2; removed hedging on observed data (e.g., "appears to be detracting" → direct).

**§5 restructure and figure** (committed Aug 2026):
- **F1/ROUGE-L figure** (`fig:fout-f1`): TikZ four-panel figure inserted at end of §5.1 fourth bullet. Panels: (a) standard partial overlap, (b) order-blind failure (winner/loser swapped, F1=1.0), (c) robust reordering (F1=1.0, ROUGE-L=0.750 due to LCS), (d) paraphrase blindness (both=0.0). Each panel shows F1 and ROUGE-L calculations inline. Preamble: color defs (`wov@c` green, `wrf@c` amber, `wpr@c` blue), word-box commands `\wov`/`\wrf`/`\wpr`/`\wst`, `\captionsetup{font=small}`.
- **§4 closing paragraph removed**: "Taken at face value, PyramidKV is the clear winner. But in §5 we argue..." — was a forward reference.
- **§5.1 first paragraph**: Last sentence ("We instead ask that question directly...") removed.
- **§5.1 fourth bullet rewritten**: Integrates F1/ROUGE-L failure modes with panel references (b, c, d); itemize split around figure.
- **§5.1 "Low full-model accuracy" bullet third paragraph**: Tightened from ~6 sentences to 3.
- **§5.1 last paragraph deleted**: PyramidKV/SnapKV paragraph that previewed §8 results.
- **§5.2 opener rewritten**: "Ground Truth's failures" → "ground-truth evaluation's failures"; "those chosen by a third party" → "a human-written reference"; filler transition sentence removed; "We believe that the best method... is by using" → "Compression fidelity is best evaluated by...".
- **§5.2 last paragraph trimmed**: Instruct model numbers (0.94→0.039 nats, 24×) and §6.4 forward reference removed; first sentence ("This distinction is what lets a result generalize...as §8 demonstrates") folded into previous paragraph.
- **§5.3 Caveat emptor opener**: Added cross-reference to Figure 1 panels (b) and (d).

**§6.3 / §7 restructure** (committed Aug 2026):
- **Gapped presentation removed** from Tab:t2 and §6.3 prose (evaluation bug: single-pass eval
  evicted y* tokens from KV cache, inflating KL for the gapped condition; ROG in Tab:t1 is unaffected).
- **ROG mechanistic note added to §6.2**: explains enrichment × scatter interaction — scatter alone
  causes limited damage; enrichment makes it catastrophic.
- **§6.4 "Re-rotation Confirms" removed as a section**: content absorbed into §7.2. Right-aligned
  re-rotation explanation and streaming value-corruption explanation now live in §7.2. §7.2 opener
  confirms §6.3 predictions. §6.3 closing now says "§7.2 confirms both predictions."
- **Section renumbering**: §7.4 Output Faithfulness → §7.3; §7.5 Inference Performance → §7.4.
- **Figure 4 caption**: centered via `\captionsetup{justification=centering}` (not `\centering` inside
  `\caption{}`, which broke cross-reference counters).

**§8 short/long definition and Tab:rog** (committed Aug 2026):
- **Short/long defined** at start of §8 intro paragraph (≤90 words / >90 words), before first use.
- **Tab:rog added** (§8 Mechanistic confirmation): 3-row diagnostic table — Recompute-over-gaps,
  SnapKV, Streaming — showing Short/Long F_out. Isolates first-token advantage from geometry effects.
  Caption: "first-token advantage diagnostic (Llama, 65%)".
- **Mechanistic confirmation paragraph** revised to cite Tab:rog instead of inline numbers; still
  references §6.2 (Tab:t1) for the mechanism.
- **"Streaming and the limits" paragraph** cut from ~9 sentences to 4; now cites Tab:rog directly.
- **"Short-answer F_out nevertheless falls" paragraph** moved from §8 body into §8.1 (before "The
  two re-rotated methods" paragraph, where it sets up that discussion naturally).

**TTFT table equalized to n=20** (committed Aug 2026): All methods use the same 20 examples per task (indices from `timing_snap_stream.json`). Naive TTFT from `kl_ystar_timing_sweep.json` phr128 total_ttft; PyramidKV from `timing_kvpress.json`; full-context baseline from `timing_full.json`.
- Full-context baseline: **6425 ms** (was 6348 ms)
- Naive: 2400 / 2082 / 1632 ms at 65/50/35% (was 2464/2139/1666)
- SnapKV: 6621 / 6872 / 6834 ms — unchanged
- Streaming: 6796 / 6750 / 6861 ms — unchanged
- Pyr: 6973 / 7245 / 7276 ms (was 7022/7109/7143)
- Prose overhead ranges: Pyr 9--13%, SnapKV+Streaming +3--7%, Naive cuts 63--75%
- Table footnote: "n=20 examples per task"

**§9 added and Appendix D expanded** (Aug 2026): Full "High-Compression Results on an Instruction-Tuned
Model" section written (Llama-3.1-8B-Instruct, budgets 256 and 1024). All Llama data now complete.
Appendix D tables D1–D6 cover Llama b256 and b1024 (KL, F_out, GT per task). Timing table
(tab:instruct-timing) breaks down short/long wall-clock time by budget. **Known issue**: Table D7
is cited in §9 text ("full per-method breakdown") but does not exist — either add it or fix the
reference. Mistral-Instruct tables (D7+) not yet written; waiting on gt_instruct_mistral_b256
(running) and kl/gt_instruct_mistral_b1024 (queued).

## Instruct eval — interpretation notes (Aug 2026)

§9 is now written. These notes record the reasoning behind the key interpretive choices in that
section.

**Why we started with base model + moderate compression (35–65%)**
This was the right choice. The instruct model at budget=256 (≈5% retention) produces results
that would have been much harder to interpret had we started there:

- **F_out is near-trivially high for selection methods**: At budget=256, snap/pyr/snap+rot/pyr+rot
  all score ≈72–73% F_out across 4 tasks — indistinguishable from each other and comparable to
  the base model's 65% results. This is an artifact of the instruct model's terse output style:
  hotpotqa answers average **3.9 words** (vs 21.7 for the base model at 65%), narrativeqa
  averages **4.5 words** (vs 83.2). With a 4-word answer, the compressed model nearly always
  reproduces the full model's words exactly, regardless of positional fidelity.

- **F_out cannot distinguish rotated from unrotated at b256**: snap vs snap+rot = 72.6 vs 72.7%;
  pyr vs pyr+rot = 72.5 vs 72.0%. Both are within noise. KL clearly separates them: snap_rot
  (0.352) vs snap (0.769) — still a 2.1× improvement from re-rotation at 5% retention (vs 27×
  at 65%). Positional corruption is real at b256; F_out just can't see it.

- **KL is the honest metric at extreme compression for instruct models**: The naive method scores
  2.003 KL (vs 0.352 for snap_rot) despite streaming being worse on F_out (47% vs 72%). The
  task-level pattern is also diagnostic: naive shows highest KL on passage_count and
  passage_retrieval (KL ≈5.8 and 3.8), exactly where block-end truncation is most harmful.

- **Window size matters at extreme budgets**: budget=256 / 32 layers = 8 tokens/layer average.
  window_size=32 (SnapKV paper default) > 8 → PyramidKV budget collapses to uniform SnapKV.
  Must use window_size ≤ average per-layer budget. b256 runs use ws=8 for snap/pyr.
  At budget=1024 (32 tokens/layer average) the pyramid works regardless of ws ≤ 32, so ws=32
  is fine.

- **Re-rotation still helps at 5% retention**: 2.1× KL improvement (snap_rot vs snap at b256).
  Positional corruption is multiplicative with information loss, not additive — fixing it still
  halves KL even when retained keys carry very little signal.

**How to interpret the new section in the paper**
When writing the high-compression instruct results section: lead with KL (honest signal), explain
why F_out appears high (short answers, near-zero surface area for divergence), cite the word-count
numbers as evidence, and note that this reinforces rather than contradicts the KL-first argument.

## Timing notes (Llama-3.1-8B, V100 32GB, fp16)

At 35% retention: ~8s/example per chunk method, ~9s for pyramidkv_f35.
At 65%: ~11s/example. Higher retention = longer budget = slower prefill + generation.
Single GPU — all GT eval runs must be strictly sequential.

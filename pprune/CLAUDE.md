# pprune — KV Cache Compression Paper

## Project overview

Research paper on KV cache compression faithfulness. Central finding: positional displacement
(scattered retained tokens at original RoPE positions) is the dominant failure mode for SnapKV and
PyramidKV. Key re-rotation fixes this: `snapkv_rerotated` dramatically improves KL (27× at 65%)
and beats phr160 on all 16 tasks. Critically, `pyramidkv_rerotated` = `snapkv_rerotated` on every
task at every rate — selection strategy is irrelevant once positions are corrected. Paper is
undergoing major structural overhaul.

Paper: `paper_phrase_compression.md`
Branch: `kvpress-impl`
Models: `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`
Data: `lb_data_raw/data/*.jsonl` (16 LongBench tasks, n=200 for gov_report/multi_news, n=100 others)

## Key methods (as they appear in the paper)

| Name in paper | Method string | Notes |
|---|---|---|
| Naive | `naive_65pct` / `naive_35pct` etc. | Prompt truncation, head_frac=0.10 |
| phr160 | `chunk_word160_t25` | 160-tok phrases, 25% tail, word overlap scoring |
| phr128 | `chunk_word128_t20` | 128-tok phrases, 20% tail, word overlap scoring |
| SnapKV | `snapkv_press` / `snapkv_press_f35` etc. | kvpress SnapKVPress, post-hoc eviction |
| SnapKV+rot | `snapkv_rerotated` / `snapkv_rerotated_f35` etc. | SnapKV + KeyRerotationPress |
| PyramidKV | `pyramidkv` / `pyramidkv_f35` etc. | kvpress PyramidKVPress |
| Pyr+rot | `pyramidkv_rerotated` / `pyramidkv_rerotated_f35` etc. | PyramidKV + KeyRerotationPress — identical to SnapKV+rot on all tasks/rates |
| Streaming | `streaming_rerotated` / `streaming_rerotated_f35` etc. | kvpress KeyRerotationPress (corrected) |

SnapKV-Select (`snapkv_select`) is a diagnostic tool only — §6.2 only.

**Experimental (RADAR resurrection — not yet in paper):**
| Name | Method string | Notes |
|---|---|---|
| RADAR pre-rope | `radar_pre_press` / `_f50` / `_f35` | Pre-RoPE max-dot scoring, post-hoc eviction, no rerotation |
| RADAR post-rope | `radar_post_press` / `_f50` / `_f35` | Post-RoPE max-dot scoring, post-hoc eviction, no rerotation |
| RADAR pre+rot | `radar_pre_rerotated` / `_f50` / `_f35` | Pre-RoPE scoring + KeyRerotationPress |
| RADAR post+rot | `radar_post_rerotated` / `_f50` / `_f35` | Post-RoPE scoring + KeyRerotationPress |

Implemented in `kl_faith_eval_ystar.py` as `RadarPreRopePress` / `RadarPostRopePress` (ScorerPress subclasses).
KL eval pending (`run_kl_radar.sh` in queue).

Rate suffix convention: no suffix = 65%, `_f50` = 50%, `_f35` = 35%. **f40 dropped from paper.**

## Eval harnesses

- `gt_eval_compression.py` — Ground-truth LongBench accuracy (full generation). Resumable checkpoint
  keyed by `task|idx|method`. Output dirs under `lb_results_base/gt_*/`.
  **IMPORTANT**: `generate_rerotated()` (added Jul 2026) handles KeyRerotationPress methods.
  Without it, decode position IDs are T+1, T+2, ... instead of M+1, M+2, ..., causing degenerate
  "In In In..." output. KL eval (teacher-forced) is NOT affected by this bug.
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

## Completed data

**KL (Llama):**
- `kl_snapkv_rerotated.json` — 65%, all 16 tasks ✓ (mean KL=0.043)
- `kl_snapkv_rerotated_f50.json` — 50% ✓ (mean KL=0.077)
- `kl_snapkv_rerotated_f35.json` — 35% ✓ (mean KL=0.124)
- `kl_snapkv_rerotated_f40.json` — 40% ✓ (f40 dropped from paper)
- `kl_pyramidkv_rerotated.json` — 65% ✓ (identical to snapkv_rerotated)
- `kl_pyramidkv_rerotated_f50.json` — 50% ✓ (identical)
- `kl_pyramidkv_rerotated_f35.json` — 35% ✓ (identical)
- `gap_structure.json` — 4 geometries × 4 presentations, 6 tasks, n=20 ✓

**KL (Mistral):**
- `kl_snapkv_rerotated_mistral.json` — 65%, all 16 tasks ✓ (mean KL=0.030)
- `kl_snapkv_rerotated_mistral_f50.json` — 50% ✓ (mean KL=0.058)
- `kl_snapkv_rerotated_mistral_f35.json` — 35% ✓ (mean KL=0.096)

**GT (Llama):**
- `gt_snapkv_rerotated/` — 65%, all 16 tasks ✓ (position-ID fix applied)
- `gt_snapkv_rerotated_f50/` — 50%, all 16 tasks ✓
- `gt_snapkv_rerotated_f35/` — 35%, all 16 tasks ✓
- `gt_pyramidkv_rerotated/` — 65%, all 16 tasks ✓ (= snapkv_rerotated on every task)
- `gt_pyramidkv_rerotated_f50/` — 50%, all 16 tasks ✓ (= snapkv_rerotated_f50)
- `gt_pyramidkv_rerotated_f35/` — 35%, all 16 tasks ✓ (= snapkv_rerotated_f35)

**GT (Mistral):**
- `gt_snapkv_rerotated_mistral/` — 65% ✓ (1600/1600; avg F_out=78.2)
- `gt_snapkv_rerotated_mistral_f50/` — 50% ✓ (1600/1600; avg F_out=74.3)
- `gt_snapkv_rerotated_mistral_f35/` — 35% ✓ (1600/1600; avg F_out=70.1)

## Currently running

**Queue runner stopped (all prior jobs complete). Restart with: `nohup ./queue_runner.sh &`**

Next in queue: `run_kl_radar.sh` — RADAR KL eval (pre-RoPE + post-RoPE, ±rerotation) at 65/50/35% on Llama.

Mechanism sweep complete: all GT files for gt_mechanism_mistral_f*/checkpoint.json ✓

## Key empirical results (Llama, F_out macro over 16 tasks)

| Method | 65% | 50% | 35% |
|---|---|---|---|
| Pyr (no rot) | 79.1 | 64.2 | 66.9 |
| SnapKV (no rot) | 78.5 | 73.2 | 66.8 |
| SnapKV+rot = Pyr+rot | 71.1 | 67.8 | 63.0 |
| Naive | 68.6 | 52.1 | 46.9 |
| Streaming (corrected) | ~59† | — | — |
| phr128 | 54.5 | 52.8 | 48.6 |

†Streaming F_out recomputed after position-ID fix; f65 partial (12/16 tasks).

## Paper structure notes

Paper is undergoing major overhaul. Tables 3, 4 and Appendices A–C updated Jul 2026.
Remaining sections need rewrite to reflect:
- Positional displacement as central failure mode
- SnapKV+rot as best KL method (beats even prompt-construction methods)
- Pyr+rot = SnapKV+rot: selection strategy irrelevant once positions corrected
- Streaming corrected F_out (Llama: 61.2/56.4/51.8% at 65/50/35%) replaces near-zero corrupted values
- Mistral SnapKV+rot F_out now complete: 78.2% / 74.3% / 70.1% at 65/50/35%

Current section state:
- §6.2: SnapKV-Select diagnostic (KL only). Only place Sel appears.
- §6.3: Gap structure table — 4 geometries × gapless/evicted/rerotated presentations. ✓
- §8: Tables updated. Narrative needs major revision.
- §9: PyramidKV case study — Table 5 (short/long F_out) uses corrected streaming data when available.
- KL metric uses y* shared prefix (fixed path-dependence flaw); see `kl_faith_eval_ystar.py`.

## Data validity note

All **KL** data is valid — teacher-forced, unaffected by position-ID bug.

**GT data**: `generate_rerotated()` fix applied Jul 2026. All KeyRerotationPress GT checkpoints
created before the fix are invalid. Affected checkpoints have been purged and re-run:
- `gt_snapkv_rerotated*/` — purged and re-run ✓
- `gt_mechanism_*/` — streaming_rerotated entries purged, re-running via mechanism sweep
- Old streaming F_out values in paper (14%, 12.7%, etc.) were from corrupted runs — now marked †

`diagnose_distribution.py` (untracked) has a head-truncation bug — NOT used for paper tables.

## Timing notes (Llama-3.1-8B, V100 32GB, fp16)

At 35% retention: ~8s/example per chunk method, ~9s for pyramidkv_f35.
At 65%: ~11s/example. Higher retention = longer budget = slower prefill + generation.
Single GPU — all GT eval runs must be strictly sequential.

# pprune — KV Cache Compression Papers

## Project overview

Two papers split from a single codebase:

**`paper_kv_faithfulness.md`** — Primary paper. KV cache compression faithfulness: positional
displacement is the dominant failure mode for SnapKV and PyramidKV. Key re-rotation (SnapKV+rot)
fixes this: 27× KL improvement at 65%, best method overall. Pyr+rot matches SnapKV+rot at 65%
(0.047 vs 0.043) and ties exactly at 35% (budget collapse to uniform); a gap opens only at 50%
where the pyramid reaches its window_size floor at layer 31. No phrase compression content.
"post-hoc" → "post-prefill" throughout.

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
- `gt_snapkv_rerotated_mistral/` — 65% ✓ (1600/1600; avg F_out=78.2)
- `gt_snapkv_rerotated_mistral_f50/` — 50% ✓ (1600/1600; avg F_out=74.3)
- `gt_snapkv_rerotated_mistral_f35/` — 35% ✓ (1600/1600; avg F_out=70.1)
- `gt_pyramidkv_rerotated_mistral/` — pending (queued)
- `gt_pyramidkv_rerotated_mistral_f50/` — pending
- `gt_pyramidkv_rerotated_mistral_f35/` — pending

## Currently running

Mistral KL Pyr+rot all rates complete. Remaining:
1. `gt_pyramidkv_rerotated_mistral/` 65/50/35% (`run_gt_pyramidkv_rerotated_mistral.sh`, in queue)

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

## Paper structure notes (paper_kv_faithfulness.md)

Paper split complete as of Jul 2026. `paper_kv_faithfulness.md` is the active submission.
Section numbering: §1–§6 unchanged, §7 Main Experiments, §8 Why Post-Prefill KV Eviction...,
§9 Discussion, §10 Conclusion. "post-hoc" replaced with "post-prefill" throughout.

- §6.2: SnapKV-Select diagnostic (KL only). Only place Sel appears.
- §6.3: Gap structure table — 4 geometries × gapless/evicted/rerotated presentations. ✓
- §6.4: Re-rotation results for all three methods. SnapKV+rot (27× KL), Streaming (1.9×),
  PyramidKV three-rate story (65%: Pyr+rot≈SnapKV+rot; 50%: gap from window_size floor;
  35%: tied from budget collapse). F_out tradeoff covers all methods. Right-aligned re-rotation
  implementation note. §6.5 removed — merged into §6.4.
- §7: Main experiment tables (KL, TTFT, F_out). All Llama Pyr+rot values filled. No phr128.
- §8: PyramidKV case study — Table 5 (short/long F_out), first-token advantage explanation.
- §9: Discussion — "budget allocation and re-rotation interact" paragraph reflects three-rate story.
- KL metric uses y* shared prefix (fixed path-dependence flaw); see `kl_faith_eval_ystar.py`.

## Data validity note

All **KL** data is valid — teacher-forced, unaffected by position-ID bug.

**GT data**: `generate_rerotated()` fix applied Jul 2026. All KeyRerotationPress GT checkpoints
created before the fix are invalid. Affected checkpoints have been purged and re-run:
- `gt_snapkv_rerotated*/` — purged and re-run ✓
- `gt_mechanism_*/` — streaming_rerotated entries purged, re-running via mechanism sweep

**PyramidKVRerotationPress fix (Jul 2026)**: `KeyRerotationPress(PyramidKVPress)` silently used
uniform budget (bypassed `get_layer_budget()`), making it identical to SnapKV+rot. Fixed by
`PyramidKVRerotationPress` subclass in `kl_faith_eval_ystar.py` that calls `get_layer_budget()`.
Right-aligned re-rotation (`[T−n_kept, T)` per layer) ensures all layers share endpoint T−1 and
decode begins at T — GT generation is valid for all tasks at all retention rates. Old
left-aligned `kl_pyramidkv_rerotated*.json` and `gt_pyramidkv_rerotated*/checkpoint.json` purged
and re-run with the right-aligned implementation.

`diagnose_distribution.py` (untracked) has a head-truncation bug — NOT used for paper tables.

## Timing notes (Llama-3.1-8B, V100 32GB, fp16)

At 35% retention: ~8s/example per chunk method, ~9s for pyramidkv_f35.
At 65%: ~11s/example. Higher retention = longer budget = slower prefill + generation.
Single GPU — all GT eval runs must be strictly sequential.

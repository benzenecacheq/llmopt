# pprune — KV Cache Compression Paper

## Project overview

Research paper on KV cache compression faithfulness. Central finding: positional displacement
(scattered retained tokens at original RoPE positions) is the dominant failure mode for SnapKV and
PyramidKV. Key re-rotation fixes this: `snapkv_rerotated` beats phr160 on all 16 tasks in KL and
matches or exceeds pyramidkv in F_out at the same retention rate. Paper is undergoing major
structural overhaul to reflect these findings.

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
| PyramidKV+rot | `pyramidkv_rerotated` / `pyramidkv_rerotated_f35` etc. | PyramidKV + KeyRerotationPress |
| Streaming | `streaming_rerotated` / `streaming_rerotated_f35` etc. | kvpress KeyRerotationPress (corrected) |

SnapKV-Select (`snapkv_select`) is a diagnostic tool only — used in §6.2 to isolate structural
corruption from token selection. Does NOT appear in the main cross-rate tables.

Rate suffix convention: no suffix = 65%, `_f50` = 50%, `_f35` = 35%. **f40 is dropped from the
paper entirely.**

## Eval harnesses

- `gt_eval_compression.py` — Ground-truth LongBench accuracy (full generation). Resumable checkpoint
  keyed by `task|idx|method`. Output dirs under `lb_results_base/gt_*/`.
  **IMPORTANT**: Contains `generate_rerotated()` (added Jul 2026) for KeyRerotationPress methods.
  These methods re-rotate keys to compact positions 0..M-1; without the fix, `model.generate()`
  uses decode position IDs T+1, T+2, ... instead of M+1, M+2, ..., causing degenerate "In In In..."
  output. KL eval (teacher-forced) is NOT affected by this bug.
- `kl_faith_eval_ystar.py` — KL faithfulness using y* shared prefix (teacher-forced). Checkpoint
  JSON files directly in `lb_results_base/kl_*.json`. Defines all `_KVPRESS_METHODS` including
  `pyramidkv_rerotated` variants.
- `kl_faith_eval.py` — Core configs (METHOD_CONFIGS, CHUNK_CONFIGS). Imported by both above.
- `longbench_eval.py` — Scoring functions. `qa_f1_score()` used for F_out computation.
- `queue_runner.sh` — File-based job queue. Reads `lb_results_base/queue.txt`, runs first
  non-comment line, removes it, repeats. Edit queue.txt live to reorder/add jobs.

All data loading uses simple unshuffled JSONL reads — idx alignment is safe across checkpoints
from different scripts pointing to the same data_dir.

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
| `lb_results_base/kl_mechanism_llama_f65.json` | KL: snapkv_press + streaming_rerotated, Llama 65%, all 16 tasks, n=100 ✓ |
| `lb_results_base/kl_mechanism_llama_f50.json` | Same at 50% ✓ |
| `lb_results_base/kl_mechanism_llama_f35.json` | Same at 35% ✓ |
| `lb_results_base/kl_mechanism_mistral_f*.json` | Same for Mistral, all 3 rates ✓ |
| `lb_results_base/gt_mechanism_llama_f65/checkpoint.json` | GT: snapkv_press (done), streaming_rerotated (re-running with fix) |
| `lb_results_base/gt_mechanism_llama_f50/checkpoint.json` | GT: snapkv_press done, streaming_rerotated re-running |
| `lb_results_base/gt_mechanism_llama_f35/checkpoint.json` | GT: snapkv_press done, streaming_rerotated re-running |
| `lb_results_base/gt_mechanism_mistral_f*/checkpoint.json` | Same for Mistral (f35 snapkv_press partial, resuming) |

## Completed data

- `lb_results_base/kl_snapkv_rerotated.json` — snapkv_rerotated KL, Llama 65%, all 16 tasks ✓
- `lb_results_base/kl_snapkv_rerotated_f50.json` — same at 50% ✓
- `lb_results_base/kl_snapkv_rerotated_f35.json` — same at 35% ✓
- `lb_results_base/kl_snapkv_rerotated_f40.json` — same at 40% ✓ (f40 dropped from paper)
- `lb_results_base/gap_structure.json` — all 4 geometries × 4 presentations (gapless/evicted/rerotated/gapped), 6 tasks, n=20 ✓
- `lb_results_base/gt_snapkv_rerotated_f50/checkpoint.json` — GT snapkv_rerotated 50%, in progress (gov_report running)

## Currently running

**Queue runner PID 47806. Edit `lb_results_base/queue.txt` to modify.**

Currently running: `run_gt_snapkv_rerotated_f50.sh` (gov_report in progress)

Remaining queue:
1. `run_kl_pyramidkv_rerotated_f35.sh` — KL pyramidkv_rerotated at 35%
2. `run_gt_snapkv_rerotated_f35.sh` — GT snapkv_rerotated 35%, all 16 tasks
3. `run_gt_pyramidkv_rerotated_f35.sh` — GT pyramidkv_rerotated 35%, all 16 tasks
4. `run_gt_snapkv_rerotated.sh` — GT snapkv_rerotated 65%, all 16 tasks
5. `run_gt_mechanism_sweep.sh` — GT streaming_rerotated at 65/50/35% (Llama + Mistral); resumes Mistral f35 snapkv_press
6. `run_kl_pyramidkv_rerotated_f50.sh` — KL pyramidkv_rerotated at 50%
7. `run_gt_pyramidkv_rerotated_f50.sh` — GT pyramidkv_rerotated 50%, all 16 tasks
8. `run_kl_pyramidkv_rerotated.sh` — KL pyramidkv_rerotated at 65%
9. `run_gt_pyramidkv_rerotated.sh` — GT pyramidkv_rerotated 65%, all 16 tasks

## Paper structure notes

Paper is undergoing major overhaul. Original thesis ("use prompt construction instead of KV
pruning") is superseded by the positional displacement finding. New framing:
- Positional displacement (scattered tokens at original RoPE positions) is the dominant failure mode
- Re-rotation to compact positions fixes it: snapkv_rerotated KL is 27× lower than snapkv_press
- phr160 is the TTFT-efficient alternative (no prefill penalty), not the superior method
- pyramidkv_rerotated is being evaluated as the potential strongest method

- §6.2: SnapKV-Select diagnostic (KL only). Only place Sel appears.
- §6.3: Gap structure table — 4 geometries × gapless/evicted/rerotated presentations.
- §8: Cross-rate tables — methods: Naive, phr160, phr128, SnapKV, SnapKV+rot, Streaming, Pyr, Pyr+rot (TBD)
- §9: PyramidKV case study (structure TBD pending new results)
- KL metric uses y* shared prefix (fixed path-dependence flaw); see `kl_faith_eval_ystar.py`.

## Data validity note

All **KL** data is valid — teacher-forced evaluation is unaffected by the position-ID bug.

**GT data validity**: `generate_rerotated()` fix was applied Jul 2026. Any GT checkpoint for a
KeyRerotationPress method (snapkv_rerotated, streaming_rerotated, pyramidkv_rerotated) created
BEFORE this fix is invalid (degenerate outputs). Affected checkpoints have been purged:
- `gt_snapkv_rerotated/` — cleared, re-running
- `gt_snapkv_rerotated_f35/` — cleared, re-running
- `gt_mechanism_llama_f*/` and `gt_mechanism_mistral_f*/` — streaming_rerotated entries purged,
  snapkv_press entries preserved; streaming_rerotated re-running via mechanism sweep

`diagnose_distribution.py` (untracked) has a head-truncation bug — NOT used for paper tables.

## Timing notes (Llama-3.1-8B, V100 32GB, fp16)

At 35% retention: ~8s/example per chunk method, ~9s for pyramidkv_f35.
At 65%: ~11s/example. Higher retention = longer budget = slower prefill + generation.
Single GPU — all GT eval runs must be strictly sequential.

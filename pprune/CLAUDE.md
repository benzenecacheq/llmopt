# pprune — KV Cache Compression Paper

## Project overview

Research paper arguing that KV cache pruning introduces structural corruption (causal gaps from
scattered retained positions) that limits faithfulness, and that phrase-based prompt construction
avoids this entirely. Two faithfulness metrics: KL divergence (y*-anchored) and Output Faithfulness
(F_out, word-level F1 against the full model's greedy output).

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
| phr160_h10 | `chunk_word160_t25_h10` | + head_frac=0.10 (under investigation) |
| phr128_h10 | `chunk_word128_t20_h10` | + head_frac=0.10 (under investigation) |
| SnapKV | `snapkv` / `snapkv_35pct` etc. | KV pruning, pooled attention scores |
| PyramidKV | `pyramidkv` / `pyramidkv_f35` etc. | KV pruning via kvpress library |
| Streaming | `streaming` | KV pruning, sink+recency window |

SnapKV-Select (`snapkv_select`) is a diagnostic tool only — used in §6.2 to isolate structural
corruption from token selection. It does NOT appear in the main cross-rate tables (removed as of
this commit); keep it in §6 discussion only.

Rate suffix convention: no suffix = 65%, `_f50` = 50%, `_f40` = 40%, `_f35` = 35%.

## Eval harnesses

- `gt_eval_compression.py` — Ground-truth LongBench accuracy (full generation). Resumable checkpoint
  keyed by `task|idx|method`. Output dirs under `lb_results_base/gt_*/`.
- `kl_faith_eval_ystar.py` — KL faithfulness using y* shared prefix (teacher-forced). Checkpoint
  JSON files directly in `lb_results_base/kl_ystar_*.json`.
- `kl_faith_eval.py` — Core configs (METHOD_CONFIGS, CHUNK_CONFIGS). Imported by both above.
- `longbench_eval.py` — Scoring functions. `qa_f1_score()` used for F_out computation.

All data loading uses simple unshuffled JSONL reads — idx alignment is safe across checkpoints
from different scripts pointing to the same data_dir.

## Key data files

| File | Contents |
|---|---|
| `lb_results_base/checkpoint.json` | Llama: full, naive, snapkv, streaming, etc. All 16 tasks, n≥100 |
| `lb_results_base/gt_comp/checkpoint.json` | phr128/phr160 + snapkv/radar at 50/40/35%, 9 tasks, n=100 |
| `lb_results_base/gt_comp_remaining/checkpoint.json` | Same methods, remaining 7 tasks, n=100 |
| `lb_results_base/gt_pyramidkv/checkpoint.json` | pyramidkv (65%), all 16 tasks, n=100 |
| `lb_results_base/gt_mistral_table/checkpoint.json` | Mistral: naive, snapkv, pyramidkv, streaming, 65%, n=100 |
| `lb_results_base/kl_ystar_tables_v3.json` | KL for prompt-construction methods, all 16 tasks, n=100 |
| `lb_results_base/kl_ystar_pyramidkv_all_v2.json` | KL for pyramidkv at all rates, all 16 tasks, n=100 |
| `lb_results_base/ystar_cache_v3.pt` | Cached y* tokens + log_p_full for all 16 tasks, n=100 |
| `lb_results_base/gt_h10_llama_f35/` | phr128_h10 + phr160_h10 at 35%, all 16 tasks, n=100 ✓ done |
| `lb_results_base/gt_h10_llama_f40/` | Same at 40% — in progress (run_h10_full_sweep.sh) |

## Currently running

`run_h10_full_sweep.sh` — full h10 sweep, sequential queue:
1. Llama 35% ✓ done
2. Llama 40% — IN PROGRESS
3. Llama 50%
4. Llama 65%
5–8. Mistral 35/40/50/65%

Log: `lb_results_base/h10_full_sweep.log`
Methods per stage: `chunk_word128_t20_h10_f{35,40,50}` / `chunk_word128_t20_h10` (65%) and phr160 equivalents.

## h10 findings so far (Llama 35%, n=100, all 16 tasks)

Average delta vs no-h10 baseline: phr128 +0.4 GT pts, phr160 −0.7 GT pts.
Key: h10 helps multi_news (+1.9 phr128) and qmsum (+1.3), hurts lcc (−2.5/−3.6) and trec (−1.0/−2.0).
phr160 base had anomalous +10pt on passage_retrieval_en vs phr128 — diagnosed as chunk boundary
artifact (phr128 systematically off-by-one on paragraph numbering). Not a data error.

## Paper structure notes

- §6.2: SnapKV-Select diagnostic table (KL + brief F_out note). This is the ONLY place Sel appears.
- §8: Main cross-rate tables — Sel column removed. Methods: Naive, phr160, phr128, SnapKV, Pyr.
- §9: PyramidKV case study. Table 5 (F_out by length category) added in this branch.
- §9.5: Synthesis paragraph — PyramidKV fails on its own terms (slow on short, no better on long).
- KL metric uses y* shared prefix (fixed path-dependence flaw); see `kl_faith_eval_ystar.py`.

## Timing notes (Llama-3.1-8B, V100 32GB, fp16)

At 35% retention: ~8s/example per chunk method, ~9s for pyramidkv_f35.
At 65%: ~11s/example. Higher retention = longer budget = slower prefill + generation.
Single GPU — all GT eval runs must be strictly sequential.

#!/bin/bash
# Full KL faithfulness re-computation using corrected y* cache (ystar_cache_v2.pt).
# The old cache was generated without stop tokens so y* contained post-answer
# context regurgitation, inflating all KL scores.  ystar_cache_v2.pt truncates
# each entry at the first \n / \n\n for short-answer tasks.
#
# Run order (priority order for paper tables):
#   1. Primary methods (naive, kq_post_rope, phrase, snapkv variants)     → kl_ystar_full_v2.json
#   2. Timing sweep (chunk128/160 + snapkv_select at all rates)           → kl_ystar_timing_sweep_v2.json
#   3. Select methods (snapkv_select, radar_select at all rates)          → kl_ystar_select_v2.json
#   4. PyramidKV base                                                      → kl_ystar_pyramidkv_v2.json
#   5. PyramidKV f50/f40/f35                                              → kl_ystar_pyramidkv_comp_v2.json
#   6. kq_only + streaming                                                 → kl_ystar_kqonly_streaming_v2.json
#   7. GT eval: pyramidkv_f50/f40/f35 on Llama  (GT unaffected by cache)
#   8. GT eval: pyramidkv on Mistral             (GT unaffected by cache)

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/kl_recompute.log
exec >> "$LOG" 2>&1

echo "=== run_kl_recompute.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

CACHE=lb_results_base/ystar_cache_v3.pt
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

# ── 1. Primary methods ──────────────────────────────────────────────────────
echo ""
echo "=== [1/8] Primary methods at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods naive_65pct,naive_50pct,naive_40pct,naive_35pct,kq_post_rope,phrase_w64_t25,phrase_w64_t25_f50,phrase_w64_t25_f40,phrase_w64_t25_f35,snapkv,snapkv_50pct,snapkv_40pct,snapkv_35pct,radar_50pct,radar_40pct,radar_35pct \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_full_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_full_v2.log
echo "=== [1/8] done at $(date) ==="

# ── 2. Timing sweep (chunk + snapkv_select at all rates) ────────────────────
echo ""
echo "=== [2/8] Timing sweep at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods chunk_word160_t25,chunk_word160_t25_f50,chunk_word160_t25_f40,chunk_word160_t25_f35,chunk_word128_t20,chunk_word128_t20_f50,chunk_word128_t20_f40,chunk_word128_t20_f35,snapkv_select,snapkv_select_f50,snapkv_select_f40,snapkv_select_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_timing_sweep_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_timing_sweep_v2.log
echo "=== [2/8] done at $(date) ==="

# ── 3. Select methods ────────────────────────────────────────────────────────
echo ""
echo "=== [3/8] Select methods at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods snapkv_select,snapkv_select_f50,snapkv_select_f40,snapkv_select_f35,radar_select,radar_select_f50,radar_select_f40,radar_select_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_select_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_select_v2.log
echo "=== [3/8] done at $(date) ==="

# ── 4. PyramidKV base (65%) ─────────────────────────────────────────────────
echo ""
echo "=== [4/8] PyramidKV base at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods pyramidkv \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_pyramidkv_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_pyramidkv_v2.log
echo "=== [4/8] done at $(date) ==="

# ── 5. PyramidKV compression sweep ──────────────────────────────────────────
echo ""
echo "=== [5/8] PyramidKV f50/f40/f35 at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_pyramidkv_comp_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_pyramidkv_comp_v2.log
echo "=== [5/8] done at $(date) ==="

# ── 6. kq_only + streaming ───────────────────────────────────────────────────
echo ""
echo "=== [6/8] kq_only + streaming at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods kq_only,streaming \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_kqonly_streaming_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_kqonly_streaming_v2.log
echo "=== [6/8] done at $(date) ==="

# ── 7. GT eval: pyramidkv_f50/f40/f35 on Llama ──────────────────────────────
echo ""
echo "=== [7/8] GT eval pyramidkv f50/f40/f35 (Llama) at $(date) ==="
python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_comp \
    --n 100
echo "=== [7/8] done at $(date) ==="

# ── 8. GT eval: pyramidkv on Mistral-7B-v0.3 ────────────────────────────────
echo ""
echo "=== [8/8] GT eval pyramidkv (Mistral) at $(date) ==="
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --methods pyramidkv \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_mistral \
    --n 100
echo "=== [8/8] done at $(date) ==="

echo "=== All KL recompute runs complete at $(date) ==="

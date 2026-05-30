#!/bin/bash
# Priority run: pyramid KL first, then timing sweep with pyramid included.
#
# Step 1: All 4 PyramidKV variants KL (builds y* cache from scratch)
# Step 2: Timing sweep (chunk160/128 + snapkv_select) + pyramid variants
#         (y* already cached from step 1, so no regen cost)
#
# After both steps complete, decide whether to run Mistral GT separately.

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/pyramid_priority.log
exec >> "$LOG" 2>&1

echo "=== run_pyramid_priority.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

CACHE=lb_results_base/ystar_cache_v3.pt
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

# ── 1. PyramidKV all compression ratios ─────────────────────────────────────
echo ""
echo "=== [1/2] PyramidKV KL (all ratios) at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods pyramidkv,pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_pyramidkv_all_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_pyramidkv_all_v2.log
echo "=== [1/2] done at $(date) ==="

# ── 2. Timing sweep + pyramid (y* hot from step 1) ──────────────────────────
echo ""
echo "=== [2/2] Timing sweep + pyramid at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods chunk_word160_t25,chunk_word160_t25_f50,chunk_word160_t25_f40,chunk_word160_t25_f35,chunk_word128_t20,chunk_word128_t20_f50,chunk_word128_t20_f40,chunk_word128_t20_f35,snapkv_select,snapkv_select_f50,snapkv_select_f40,snapkv_select_f35,pyramidkv,pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_timing_sweep_v2.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_timing_sweep_v2.log
echo "=== [2/2] done at $(date) ==="

echo "=== run_pyramid_priority.sh complete at $(date) ==="

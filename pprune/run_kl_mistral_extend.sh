#!/bin/bash
# KL faithfulness eval for Mistral-7B-v0.3 — two phases:
#   Phase 1: add cw160, cw128, sel at 65% to complete the §4.4 table
#   Phase 2: all 6 methods at 35% retention for cross-rate consistency check
#
# Phase 1 output → lb_results_base/kl_ystar_mistral_65_ext.json
# Phase 2 output → lb_results_base/kl_ystar_mistral_f35.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/kl_mistral_extend.log
exec >> "$LOG" 2>&1

echo "=== run_kl_mistral_extend.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

# --- Phase 1: complete the 65% table with prompt-construction methods ---
echo "=== Phase 1: 65% prompt-construction methods at $(date) ==="

python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods "chunk_word160_t25,chunk_word128_t20,snapkv_select" \
    --n 100 \
    --output lb_results_base/kl_ystar_mistral_65_ext.json \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --log "$LOG"

echo "=== Phase 1 complete at $(date) ==="

# --- Phase 2: all 6 methods at 35% retention ---
echo "=== Phase 2: 35% retention all methods at $(date) ==="

python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods "naive_35pct,chunk_word160_t25_f35,chunk_word128_t20_f35,snapkv_select_f35,snapkv_35pct,pyramidkv_f35" \
    --n 100 \
    --output lb_results_base/kl_ystar_mistral_f35.json \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --log "$LOG"

echo "=== Phase 2 complete at $(date) ==="
echo "=== run_kl_mistral_extend.sh finished at $(date) ==="

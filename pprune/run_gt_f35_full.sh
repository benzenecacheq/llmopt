#!/bin/bash
# Full monty: 35% retention, all 16 LongBench tasks, n=100, all main
# compression methods including the chunk_word*_h10 head_frac variants
# (the latter being the thing under active investigation).
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_f35_full.log
exec >> "$LOG" 2>&1
echo "=== run_gt_f35_full.sh started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
METHODS="naive_35pct,snapkv_35pct,snapkv_select_f35,pyramidkv_f35,chunk_word128_t20_f35,chunk_word128_t20_h10_f35,chunk_word160_t25_f35,chunk_word160_t25_h10_f35"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --output lb_results_base/gt_f35_full \
    --n 100

echo "=== run_gt_f35_full.sh complete at $(date) ==="

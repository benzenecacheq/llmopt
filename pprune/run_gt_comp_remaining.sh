#!/bin/bash
# GT evaluation for 11 tasks missing from §7.4 cross-rate tables.
# Runs cw160, cw128, snapkv_select at 65%; naive/cw160/cw128/sel/snapkv/radar at 50/40/35%.
# naive/snapkv/radar at 65% already exist in §4.3 table and don't need recomputation.
# Output → lb_results_base/gt_comp_remaining/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_comp_remaining.log
exec >> "$LOG" 2>&1

echo "=== run_gt_comp_remaining.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="multifieldqa_en,musique,gov_report,qmsum,multi_news,trec,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "$TASKS" \
    --output lb_results_base/gt_comp_remaining \
    --n 100

echo "=== run_gt_comp_remaining.sh complete at $(date) ==="

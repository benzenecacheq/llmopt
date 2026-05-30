#!/bin/bash
# GT evaluation for pyramidkv_f40_clean_first: PyramidKV at 40% retention
# with the clean-first-token fix (T1 attends to compressed KV, not full prefill).
#
# Output → lb_results_base/gt_pyramidkv_f40_clean_first/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_pyramidkv_f40_clean_first.log
exec >> "$LOG" 2>&1

echo "=== run_gt_pyramidkv_f40_clean_first.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_f40_clean_first \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_f40_clean_first \
    --n 100

echo "=== run_gt_pyramidkv_f40_clean_first.sh complete at $(date) ==="

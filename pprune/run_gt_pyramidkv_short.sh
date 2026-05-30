#!/bin/bash
# GT evaluation for pyramidkv_short: same PyramidKV compression (keep 65%),
# but input capped at max_seq_comp (~4096 tokens) instead of max_seq_full (~7168).
#
# Purpose: isolate whether PyramidKV's GT advantage over naive comes from
# (a) seeing more context during prefill, or (b) compression quality.
#
# Compare results to:
#   pyramidkv (full)   → lb_results_base/gt_pyramidkv/
#   naive_65pct        → lb_results_base/gt_comp/
#
# Output → lb_results_base/gt_pyramidkv_short/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_pyramidkv_short.log
exec >> "$LOG" 2>&1

echo "=== run_gt_pyramidkv_short.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_short \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_short \
    --n 100

echo "=== run_gt_pyramidkv_short.sh complete at $(date) ==="

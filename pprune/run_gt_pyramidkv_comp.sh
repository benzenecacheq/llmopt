#!/bin/bash
# GT evaluation for pyramidkv f50/f40/f35 on Llama-3.1-8B, all 16 tasks, n=100.
# Answers the key question: does pyramid's GT advantage hold at tighter compression?

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_pyramidkv_comp.log
exec >> "$LOG" 2>&1

echo "=== run_gt_pyramidkv_comp.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_comp \
    --n 100

echo "=== run_gt_pyramidkv_comp.sh complete at $(date) ==="

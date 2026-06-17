#!/bin/bash
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/pyramidkv_select_gt.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p" \
    --methods "pyramidkv_select,pyramidkv_select_f35" \
    --output lb_results_base/gt_pyramidkv_select \
    --n 100

echo "=== complete at $(date) ==="

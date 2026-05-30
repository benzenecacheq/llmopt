#!/bin/bash
# Run PyramidKV GT eval on Mistral-7B-v0.3 after current GT comp run (PID 825500) finishes.
# Purpose: test whether PyramidKV's near-full-context GT scores on Llama replicate on Mistral
# (checking for LongBench overfitting hypothesis vs. fundamental KV pruning behavior).

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_pyramidkv_mistral.log
exec >> "$LOG" 2>&1

echo "=== run_pyramidkv_mistral.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

# Wait for GT comp run to finish
echo "Waiting for GT comp PID 825500..."
while kill -0 825500 2>/dev/null; do sleep 30; done
echo "GT comp finished at $(date)"

# Run PyramidKV GT eval on Mistral-7B-v0.3 (all 16 tasks)
echo ""
echo "=== Starting PyramidKV GT eval on Mistral-7B-v0.3 at $(date) ==="
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --methods pyramidkv \
    --tasks narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p \
    --output lb_results_base/gt_pyramidkv_mistral \
    --n 100

echo "=== PyramidKV Mistral GT eval finished at $(date) ==="

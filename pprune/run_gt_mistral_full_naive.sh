#!/bin/bash
# Fills the two missing columns in the Mistral 65% GT table:
#   naive_65pct  — prompt-construction baseline (fast)
#   full         — uncompressed full-context reference (slow)
# Output → lb_results_base/gt_mistral_table/ (same checkpoint as existing Mistral 65% GT)

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_mistral_full_naive.log
exec >> "$LOG" 2>&1

echo "=== run_gt_mistral_full_naive.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods naive_65pct,full \
    --output lb_results_base/gt_mistral_table \
    --n 100

echo "=== run_gt_mistral_full_naive.sh complete at $(date) ==="

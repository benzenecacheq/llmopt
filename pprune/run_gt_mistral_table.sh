#!/bin/bash
# GT evaluation for Mistral-7B-v0.3 at 65% retention, matching the Llama §4.3 table.
# Methods: naive_65pct, kq_post_rope (RADAR), kq_only, snapkv, streaming, pyramidkv
# Full baseline already in lb_results_mistral/faithfulness_results.json.
# Output → lb_results_base/gt_mistral_table/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_mistral_table.log
exec >> "$LOG" 2>&1

echo "=== run_gt_mistral_table.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
METHODS="kq_post_rope,kq_only,snapkv,streaming,pyramidkv"

python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --output lb_results_base/gt_mistral_table \
    --n 100

echo "=== run_gt_mistral_table.sh complete at $(date) ==="

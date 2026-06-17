#!/bin/bash
# KL faithfulness eval for Mistral-7B-v0.3, matching §4.4 Llama columns.
# Methods: naive_65pct, kq_post_rope (RADAR), kq_only, snapkv, streaming, pyramidkv
# Queued to run after run_gt_mistral_table.sh (PID 2540541) finishes.
# Output → lb_results_base/kl_ystar_mistral.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/kl_mistral_table.log
exec >> "$LOG" 2>&1

echo "=== run_kl_mistral_table.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

echo "Waiting for GT eval PID 2540541 to finish..."
while kill -0 2540541 2>/dev/null; do sleep 30; done
echo "GT eval finished at $(date)"

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
METHODS="naive_65pct,kq_post_rope,kq_only,snapkv,streaming,pyramidkv"

python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --n 100 \
    --output lb_results_base/kl_ystar_mistral.json \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --log lb_results_base/kl_mistral_table.log

echo "=== run_kl_mistral_table.sh complete at $(date) ==="

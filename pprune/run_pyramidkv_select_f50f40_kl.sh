#!/bin/bash
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/pyramidkv_select_f50f40_kl.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "$TASKS" \
    --methods "pyramidkv_select_f50,pyramidkv_select_f40" \
    --n 100 \
    --output lb_results_base/kl_pyramidkv_select_f50f40.json \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --log "$LOG"

echo "=== complete at $(date) ==="

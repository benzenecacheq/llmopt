#!/bin/bash
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/chunk_headfrac_kl.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="gov_report,multi_news"

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "$TASKS" \
    --methods "naive_65pct,chunk_word128_t20,chunk_word128_t20_h10,naive_35pct,chunk_word128_t20_f35,chunk_word128_t20_h10_f35" \
    --n 25 \
    --output lb_results_base/kl_chunk_headfrac.json \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --log "$LOG"

echo "=== complete at $(date) ==="

#!/bin/bash
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/chunk_headfrac_gt.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "gov_report,multi_news" \
    --methods "naive_65pct,chunk_word128_t20,chunk_word128_t20_h10,naive_35pct,chunk_word128_t20_f35,chunk_word128_t20_h10_f35" \
    --output lb_results_base/gt_chunk_headfrac \
    --n 25

echo "=== complete at $(date) ==="

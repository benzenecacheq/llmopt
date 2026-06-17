#!/bin/bash
# Fills out the phr160 side of the head_frac comparison on the two no-query
# tasks (gov_report, multi_news), matching what was already run for phr128.
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/phr160_headfrac_gt.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "gov_report,multi_news" \
    --methods "chunk_word160_t25,chunk_word160_t25_h10,chunk_word160_t25_f35,chunk_word160_t25_h10_f35" \
    --output lb_results_base/gt_phr160_headfrac \
    --n 25

echo "=== complete at $(date) ==="

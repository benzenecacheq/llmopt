#!/bin/bash
# Test whether PyramidKV's tail window drives first-token accuracy on TREC and LCC.
# Compares pyramidkv (window_size=128) vs pyramidkv_no_window (window_size=0).
# Output appended to lb_results_base/rank_analysis.json (resume-safe).

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/rank_no_window.log
exec >> "$LOG" 2>&1

echo "=== run_rank_no_window.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python rank_analysis.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks trec,lcc \
    --n 25 \
    --methods "pyramidkv,pyramidkv_no_window" \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --output lb_results_base/rank_no_window.json \
    --log "$LOG"

echo "=== run_rank_no_window.sh complete at $(date) ==="

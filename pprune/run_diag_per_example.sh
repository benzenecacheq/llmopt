#!/bin/bash
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/diag_per_example.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
python diagnose_distribution.py \
    --tasks trec,lcc \
    --n 25 \
    --methods "naive_65pct,chunk_word128_t20,snapkv,pyramidkv" \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --output lb_results_base/diag_per_example.json
echo "=== complete at $(date) ==="

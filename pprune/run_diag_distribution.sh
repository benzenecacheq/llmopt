#!/bin/bash
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/diag_distribution.log
exec >> "$LOG" 2>&1
echo "=== run_diag_distribution.sh started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
python diagnose_distribution.py \
    --tasks trec,lcc,narrativeqa \
    --n 25 \
    --methods "naive_65pct,chunk_word128_t20,snapkv,pyramidkv" \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --output lb_results_base/diag_distribution.json
echo "=== run_diag_distribution.sh complete at $(date) ==="

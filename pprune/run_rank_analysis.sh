#!/bin/bash
# Per-token rank and probability analysis across all 16 tasks.
# Tests H1 (argmax preserved at high KL) and H2 (KL grows with token position).
# Methods: naive, cw128, sel, snapkv, pyramidkv — all at 65% retention.
# Output → lb_results_base/rank_analysis.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/rank_analysis.log
exec >> "$LOG" 2>&1

echo "=== run_rank_analysis.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python rank_analysis.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks all \
    --n 25 \
    --methods "naive_65pct,chunk_word128_t20,snapkv_select,snapkv,pyramidkv" \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --output lb_results_base/rank_analysis.json \
    --log "$LOG"

echo "=== run_rank_analysis.sh complete at $(date) ==="

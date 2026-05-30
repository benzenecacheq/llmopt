#!/bin/bash
# Chain: rank diagnostic first, then compression comparison.
set -e

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

CACHE=lb_results_base/ystar_cache_v3.pt

echo "=== [1/2] rank diagnostic starting at $(date) ==="
python diagnose_pyramid_ranks.py \
    --tasks hotpotqa,trec,gov_report \
    --n 50 \
    --methods naive_65pct,snapkv,pyramidkv,pyramidkv_f35 \
    --ystar_cache "$CACHE" \
    > lb_results_base/rank_diagnostic.log 2>&1
echo "=== [1/2] rank diagnostic complete at $(date) ==="

echo ""
echo "=== [2/2] compression comparison starting at $(date) ==="
bash run_compression_comparison.sh
echo "=== [2/2] compression comparison complete at $(date) ==="

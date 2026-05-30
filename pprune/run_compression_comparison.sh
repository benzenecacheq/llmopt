#!/bin/bash
# Compression-rate comparison: naive vs snapkv vs pyramidkv at 65/50/40/35% retention.
#
# Goal: isolate where the "more compression → lower KL" anomaly originates.
#   - naive:    uniform random truncation (no selection)
#   - snapkv:   attention-scored selection, uniform across all layers
#   - pyramidkv: attention-scored selection, pyramid layer allocation
#
# If snapkv mirrors pyramid (KL decreases with compression on QA tasks), the
# anomaly is in the SnapKV scoring mechanism.  If snapkv mirrors naive (KL
# increases with compression), the anomaly is in pyramid's layer heterogeneity.
#
# Tasks: 4 short-answer QA + 2 summarization, n=50.
# y* cache reused from ystar_cache_v3.pt (populated by run_pyramid_priority step 1).

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/compression_comparison.log
exec >> "$LOG" 2>&1

echo "=== run_compression_comparison.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

CACHE=lb_results_base/ystar_cache_v3.pt
TASKS="hotpotqa,2wikimqa,musique,narrativeqa,gov_report,qmsum"

python kl_faith_eval_ystar.py \
    --methods naive_65pct,naive_50pct,naive_40pct,naive_35pct,snapkv,snapkv_50pct,snapkv_40pct,snapkv_35pct,pyramidkv,pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --n 50 \
    --output lb_results_base/kl_compression_comparison.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_compression_comparison.log

echo "=== run_compression_comparison.sh complete at $(date) ==="

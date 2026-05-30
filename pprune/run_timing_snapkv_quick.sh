#!/bin/bash
# Quick SnapKV TTFT/TPT validation: n=20 per task on 5 tasks (~5 min).
# Compare against timing_kvpress.json pyramidkv numbers.
# Output → lb_results_base/timing_snapkv_quick.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/timing_snapkv_quick.log
exec >> "$LOG" 2>&1

echo "=== run_timing_snapkv_quick.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python time_snapkv_quick.py

echo "=== run_timing_snapkv_quick.sh complete at $(date) ==="

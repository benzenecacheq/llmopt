#!/bin/bash
# GT eval comparing tail-window variants on TREC and LCC.
# Tests whether forced tail retention (not layer-adaptive allocation) drives
# PyramidKV's GT advantage.
#
# Methods:
#   pyramidkv          — standard (window_size=128)
#   pyramidkv_no_window — window_size=1 (attention scoring only)
#   snapkv             — standard (always_keep_last=16)
#   snapkv_keep128     — always_keep_last=128 (mirrors pyramid's tail guarantee)
#
# Output → lb_results_base/gt_window_test/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_window_test.log
exec >> "$LOG" 2>&1

echo "=== run_gt_window_test.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks trec,lcc \
    --methods "pyramidkv,pyramidkv_no_window,snapkv,snapkv_keep128" \
    --output lb_results_base/gt_window_test \
    --n 100

echo "=== run_gt_window_test.sh complete at $(date) ==="

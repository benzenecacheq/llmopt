#!/bin/bash
# Measure TTFT and TPT for PyramidKV at all 4 retention rates.
# Output → lb_results_base/timing_kvpress.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/timing_kvpress.log
exec >> "$LOG" 2>&1

echo "=== run_timing_kvpress.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python time_kvpress.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks 2wikimqa,multifieldqa_en,qasper,repobench-p,triviaqa \
    --methods pyramidkv,pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --n 100 \
    --output lb_results_base/timing_kvpress.json

echo "=== run_timing_kvpress.sh complete at $(date) ==="

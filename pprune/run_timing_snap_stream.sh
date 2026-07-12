#!/bin/bash
# Measure TTFT and TPT for snapkv_press and streaming_rerotated at all 4 retention rates.
# Same 5 canonical tasks and n=20 as timing_snapkv_quick.json for comparability
# against timing_full.json (full context, n=100/task) and timing_kvpress.json (pyramidkv, n=100/task).
# Output → lb_results_base/timing_snap_stream.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/timing_snap_stream.log
exec >> "$LOG" 2>&1

echo "=== run_timing_snap_stream.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python time_kvpress.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks 2wikimqa,multifieldqa_en,qasper,repobench-p,triviaqa \
    --methods snapkv_press,snapkv_press_f50,snapkv_press_f40,snapkv_press_f35,streaming_rerotated,streaming_rerotated_f50,streaming_rerotated_f40,streaming_rerotated_f35 \
    --n 20 \
    --output lb_results_base/timing_snap_stream.json

echo "=== run_timing_snap_stream.sh complete at $(date) ==="

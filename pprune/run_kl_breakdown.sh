#!/bin/bash
LOG=/home/benzene/llmopt/pprune/lb_results_base/kl_breakdown.log
exec >> "$LOG" 2>&1
echo "=== run_kl_breakdown.sh started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

# Wait for pyramid comp eval to finish
echo "Waiting for pyramid comp PID 832328..."
while kill -0 832328 2>/dev/null; do sleep 60; done
echo "Pyramid comp finished at $(date)"

python diagnose_kl_breakdown.py \
    --tasks triviaqa,trec,2wikimqa,gov_report \
    --n 10 \
    --output lb_results_base/kl_breakdown.json

echo "=== Done at $(date) ==="

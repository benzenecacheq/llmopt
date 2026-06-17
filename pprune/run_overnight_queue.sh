#!/bin/bash
# Overnight queue: waits for the currently-running headfrac GT job (PID passed as $1)
# to exit, then runs the rest of the broader head_frac comparison sweep in sequence.
# Single V100 GPU -> must be strictly sequential.
set -e
WAIT_PID="$1"
LOG=/home/benzene/llmopt/pprune/lb_results_base/overnight_queue.log
exec >> "$LOG" 2>&1
echo "=== queue started at $(date), waiting on PID $WAIT_PID ==="

while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "=== PID $WAIT_PID exited at $(date), starting queue ==="

cd /home/benzene/llmopt/pprune

echo "--- [1/3] 16-task head_frac robustness GT ---"
bash run_headfrac_robustness_gt.sh
echo "--- [1/3] done at $(date) ---"

echo "--- [2/3] phr160 head_frac fill-in GT (gov_report, multi_news) ---"
bash run_phr160_headfrac_gt.sh
echo "--- [2/3] done at $(date) ---"

echo "--- [3/3] KL for original phr128 head_frac ablation (gov_report, multi_news) ---"
bash run_chunk_headfrac_kl.sh
echo "--- [3/3] done at $(date) ---"

echo "=== queue complete at $(date) ==="

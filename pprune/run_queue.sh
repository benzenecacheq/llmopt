#!/bin/bash
# Queued runs — waits for GT eval (PID 675860) to finish, then runs:
#   1. time_full_prefill.py  — uncompressed TTFT/TPT for the 5 timed tasks
#   2. kl_faith_eval_ystar.py — y* KL for kq_only and streaming (§4.4)

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/run_queue.log
exec >> "$LOG" 2>&1

echo "=== run_queue.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

# Wait for GT eval to finish
echo "Waiting for GT eval PID 675860..."
while kill -0 675860 2>/dev/null; do sleep 30; done
echo "GT eval finished at $(date)"

# ── Run 1: uncompressed timing ──────────────────────────────────────────────
echo ""
echo "=== Starting time_full_prefill.py at $(date) ==="
python time_full_prefill.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks 2wikimqa,multifieldqa_en,qasper,repobench-p,triviaqa \
    --n 100 \
    --output lb_results_base/timing_full.json

echo "=== time_full_prefill.py finished at $(date) ==="

# ── Run 2: kq_only + streaming y* eval ─────────────────────────────────────
echo ""
echo "=== Starting kl_faith_eval_ystar.py (kq_only,streaming) at $(date) ==="
python kl_faith_eval_ystar.py \
    --methods kq_only,streaming \
    --tasks narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p \
    --output lb_results_base/kl_ystar_kqonly_streaming.json \
    --ystar_cache lb_results_base/ystar_cache.pt \
    --log lb_results_base/kl_ystar_kqonly_streaming.log

echo "=== kl_faith_eval_ystar.py finished at $(date) ==="
echo "=== All queued runs complete at $(date) ==="

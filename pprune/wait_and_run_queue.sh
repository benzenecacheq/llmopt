#!/bin/bash
# Wait for the training job to finish, then start queue_runner.

TRAIN_PID=100245
PPRUNE=/home/benzene/llmopt/pprune
LOG=$PPRUNE/lb_results_base/job_queue.log

echo "[$(date '+%Y-%m-%d %H:%M')] Watching training PID $TRAIN_PID" | tee -a "$LOG"

while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M')] Training job done. Starting queue_runner." | tee -a "$LOG"
cd $PPRUNE
bash queue_runner.sh

#!/bin/bash
# queue_runner.sh — reads queue.txt and executes jobs one at a time.
# Edit queue.txt at any time to add, remove, or reorder pending jobs.
# Lines starting with # and blank lines are preserved and skipped.
# Each executed line is deleted from the file before running.

QUEUE="lb_results_base/queue.txt"
LOG="lb_results_base/job_queue.log"

cd /home/benzene/llmopt/pprune
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt

echo "[$(date '+%Y-%m-%d %H:%M')] queue_runner started (PID $$)" | tee -a "$LOG"

while true; do
    # Read queue, pull first non-comment non-blank line, preserve the rest
    CMD=""
    TMPFILE=$(mktemp)
    FOUND=0
    while IFS= read -r line; do
        stripped="${line#"${line%%[![:space:]]*}"}"
        if [ "$FOUND" -eq 0 ] && [ -n "$stripped" ] && [[ "$stripped" != \#* ]]; then
            CMD="$line"
            FOUND=1
        else
            printf '%s\n' "$line" >> "$TMPFILE"
        fi
    done < "$QUEUE"

    if [ "$FOUND" -eq 0 ]; then
        rm -f "$TMPFILE"
        echo "[$(date '+%Y-%m-%d %H:%M')] Queue empty — runner done." | tee -a "$LOG"
        break
    fi

    mv "$TMPFILE" "$QUEUE"
    echo "[$(date '+%Y-%m-%d %H:%M')] START  $CMD" | tee -a "$LOG"
    bash -c "$CMD"
    STATUS=$?
    echo "[$(date '+%Y-%m-%d %H:%M')] DONE($STATUS) $CMD" | tee -a "$LOG"
done

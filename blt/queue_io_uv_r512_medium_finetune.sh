#!/bin/bash
# Wait for the OWT-large cache transfer to finish AND io's current unrelated
# job to exit, then launch the r=512 medium UV warm-start fine-tune
# (counterpart to bender's r=256 run -- rank-comparison probe before
# committing a full 3-week from-scratch medium UV run).
cd ~/llmopt_medium_ema/blt
LOG=queue_io_uv_r512_medium_finetune.log
exec >> "$LOG" 2>&1

EXPECTED_CACHE_SIZE=33861225755
echo "=== $(date): waiting for OWT-large cache transfer to finish ==="
while true; do
  SZ=$(stat -c%s ~/.cache/blt_owt_75files_blocks.pt 2>/dev/null)
  if [ "$SZ" = "$EXPECTED_CACHE_SIZE" ]; then
    break
  fi
  sleep 60
done
echo "=== $(date): cache transfer complete ($SZ bytes) ==="

echo "=== $(date): waiting on io's current job PID 2134 ==="
while kill -0 2134 2>/dev/null; do
  sleep 60
done
echo "=== $(date): io job exited, launching r=512 medium UV fine-tune ==="

nohup ~/miniconda3/bin/conda run --no-capture-output -n blt python train.py \
  --uv-rank 512 --uv-mha-warmstart run_gpt2_medium_baseline_seed42.pt --pretrained gpt2-medium \
  --dataset openwebtext_large --seed 42 --max-steps 50000 \
  --eval-every 1000 --lambada-eval-every 5000 --owt-eval-every 5000 \
  --batch-size 4 \
  --save-path run_gpt2_medium_uv_r512_finetune_seed42.pt \
  > run_gpt2_medium_uv_r512_finetune_seed42.stdout 2>&1 &
PID=$!
echo "$(date): launched, PID $PID"
sleep 60
if kill -0 "$PID" 2>/dev/null && [ -f run_gpt2_medium_uv_r512_finetune_seed42.log ]; then
  echo "$(date): VERIFIED alive: $(tail -3 run_gpt2_medium_uv_r512_finetune_seed42.log)"
else
  echo "$(date): *** LAUNCH FAILED *** see run_gpt2_medium_uv_r512_finetune_seed42.stdout"
fi

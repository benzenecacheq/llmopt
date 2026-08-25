#!/bin/bash
# Wait for bender's current cumulative-mode blend=0.75 run to finish, then
# launch the r=256 medium UV warm-start fine-tune (rank-comparison probe,
# see CLAUDE.md/model.py mha_warmstart_checkpoint -- deciding r=256 vs r=512
# before committing a full 3-week from-scratch medium UV run).
cd /home/benzene/llmopt/blt
LOG=queue_bender_finish_then_uv_r256_medium_finetune.log
exec >> "$LOG" 2>&1

echo "=== $(date): watcher started, waiting on bender PID 90902 ==="
while kill -0 90902 2>/dev/null; do
  sleep 60
done
echo "=== $(date): bender job exited, launching r=256 medium UV fine-tune ==="

nohup conda run --no-capture-output -n blt python train.py \
  --uv-rank 256 --uv-mha-warmstart run_gpt2_medium_baseline_seed42.pt --pretrained gpt2-medium \
  --dataset openwebtext_large --seed 42 --max-steps 50000 \
  --eval-every 1000 --lambada-eval-every 5000 --owt-eval-every 5000 \
  --batch-size 2 --grad-accum-steps 2 \
  --save-path run_gpt2_medium_uv_r256_finetune_seed42.pt \
  > run_gpt2_medium_uv_r256_finetune_seed42.stdout 2>&1 &
PID=$!
echo "$(date): launched, PID $PID"
sleep 60
if kill -0 "$PID" 2>/dev/null && [ -f run_gpt2_medium_uv_r256_finetune_seed42.log ]; then
  echo "$(date): VERIFIED alive: $(tail -3 run_gpt2_medium_uv_r256_finetune_seed42.log)"
else
  echo "$(date): *** LAUNCH FAILED *** see run_gpt2_medium_uv_r256_finetune_seed42.stdout"
fi

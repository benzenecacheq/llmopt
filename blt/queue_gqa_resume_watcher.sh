#!/bin/bash
# Waits for the per-layer-M warm-start run (PID 8970) to exit, then resumes GQA seed7.
while kill -0 8970 2>/dev/null; do
  sleep 30
done
echo "$(date): per-layer-M training PID 8970 exited, resuming GQA seed7..."
cd /home/benzene/llmopt/blt
/home/benzene/miniconda3/envs/blt/bin/python train.py --gqa --from-scratch --dataset openwebtext --seed 7 \
  --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 \
  --save-path run_gqa_scratch_seed7.pt --resume run_gqa_scratch_seed7.pt \
  > run_gqa_scratch_seed7_resume.stdout 2>&1
echo "$(date): GQA seed7 resume process exited."

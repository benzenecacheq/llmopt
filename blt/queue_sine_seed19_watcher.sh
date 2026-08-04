#!/bin/bash
# Runs on bender, polling venus's training PID remotely (venus lacks direct
# ssh access to other machines, so orchestration happens from here).
# Waits for the sine-blend seed19 training run on venus (PID 133040) to exit,
# then runs both the lm-eval-harness benchmark and the OWT held-out eval
# directly on venus (the 5 held-out parquet files, 21-25, were copied there
# so no relay to io is needed this time).
while ssh venus 'true; kill -0 133040' 2>/dev/null; do
  sleep 60
done
echo "$(date): sine-blend seed19 training (venus PID 133040) exited, running benchmarks..."

ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python blt_lm_eval.py --checkpoint run_gpt2_ema_blend75_sine_scratch_seed19.pt --baseline --output lm_eval_gpt2_ema_blend75_sine_scratch_seed19.json' \
  > /home/benzene/llmopt/blt/lm_eval_gpt2_ema_blend75_sine_scratch_seed19.stdout 2>&1
echo "$(date): lm-eval-harness benchmark done (venus)."

ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python eval_owt.py --baseline-checkpoint run_gpt2_ema_blend75_sine_scratch_seed19.pt' \
  > /home/benzene/llmopt/blt/eval_owt_gpt2_ema_blend75_sine_scratch_seed19.stdout 2>&1
echo "$(date): OWT held-out eval done (venus)."

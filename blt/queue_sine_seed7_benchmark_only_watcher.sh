#!/bin/bash
# Benchmark-only watcher for venus's sine-blend seed7 run (PID 216841).
# Replaces queue_venus_seed7_then_uvgroups4.sh, which also launched
# num_uv_groups=4 on venus afterward -- that experiment is now running on
# titan instead (finished first), so this watcher only benchmarks seed7.
while ssh venus 'true; kill -0 216841' 2>/dev/null; do
  sleep 60
done
echo "$(date): sine-blend seed7 training (venus PID 216841) exited, running benchmarks..."

ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python blt_lm_eval.py --checkpoint run_gpt2_ema_blend75_sine_scratch_seed7.pt --baseline --output lm_eval_gpt2_ema_blend75_sine_scratch_seed7.json' \
  > /home/benzene/llmopt/blt/lm_eval_gpt2_ema_blend75_sine_scratch_seed7.stdout 2>&1
echo "$(date): lm-eval-harness benchmark done (venus)."

ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python eval_owt.py --baseline-checkpoint run_gpt2_ema_blend75_sine_scratch_seed7.pt' \
  > /home/benzene/llmopt/blt/eval_owt_gpt2_ema_blend75_sine_scratch_seed7.stdout 2>&1
echo "$(date): OWT held-out eval done (venus)."

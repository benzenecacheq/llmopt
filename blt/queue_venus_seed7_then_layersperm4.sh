#!/bin/bash
# Runs on bender. Waits for venus's sine-blend seed7 run (PID 216841) to
# finish, benchmarks it, then launches a from-scratch OWT layers-per-m=4
# run on the now-free venus -- the isolate-the-layer-axis-alone experiment,
# complementing num_uv_groups=4 (isolate-the-head-axis-alone) running on titan.

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

echo "$(date): pulling venus's repo before launching layers-per-m=4..."
ssh venus 'true; cd /home/benzene/llmopt/blt && git pull' > /home/benzene/llmopt/blt/queue_layersperm4_gitpull.stdout 2>&1
echo "$(date): venus repo pulled -- see queue_layersperm4_gitpull.stdout for any collisions to check by hand."

HAS_FLAG=$(ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python train.py --help 2>&1 | grep -c layers-per-m')
if [ "$HAS_FLAG" -eq 0 ]; then
  echo "$(date): ABORT -- venus's train.py does not have --layers-per-m after git pull. NOT launching on stale code. Needs manual intervention."
  exit 1
fi
echo "$(date): confirmed --layers-per-m present, safe to launch."

LAUNCH_OUT=$(ssh venus 'true; cd /home/benzene/llmopt/blt && nohup ~/miniconda3/envs/blt/bin/python train.py --layers-per-m 4 --from-scratch --dataset openwebtext --seed 42 --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 --save-path run_blt_layersperm4_scratch_seed42.pt > run_blt_layersperm4_scratch_seed42.stdout 2>&1 < /dev/null & disown; sleep 3; ps aux | grep train.py | grep -v grep')
echo "$LAUNCH_OUT"
LPM_PID=$(echo "$LAUNCH_OUT" | grep layers-per-m | awk '{print $2}' | head -1)
echo "$(date): layers-per-m=4 run launched on venus, PID $LPM_PID"

sleep 120
echo "$(date): post-launch health check --"
ssh venus 'true; head -n 6 /home/benzene/llmopt/blt/run_blt_layersperm4_scratch_seed42.log; echo ---; tail -n 6 /home/benzene/llmopt/blt/run_blt_layersperm4_scratch_seed42.log'

if [ -z "$LPM_PID" ]; then
  echo "$(date): WARNING -- could not determine PID for the new run, no follow-up watcher started. Check manually."
  exit 1
fi

cat > /home/benzene/llmopt/blt/queue_layersperm4_watcher.sh <<WATCHER
#!/bin/bash
while ssh venus 'true; kill -0 $LPM_PID' 2>/dev/null; do
  sleep 60
done
echo "\$(date): layers-per-m=4 training (venus PID $LPM_PID) exited, running benchmarks..."
ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python blt_lm_eval.py --checkpoint run_blt_layersperm4_scratch_seed42.pt --layers-per-m 4 --output lm_eval_blt_layersperm4_scratch_seed42.json' \\
  > /home/benzene/llmopt/blt/lm_eval_blt_layersperm4_scratch_seed42.stdout 2>&1
echo "\$(date): lm-eval-harness benchmark done (venus)."
ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python eval_owt.py --blt-checkpoint run_blt_layersperm4_scratch_seed42.pt --blt-layers-per-m 4' \\
  > /home/benzene/llmopt/blt/eval_owt_blt_layersperm4_scratch_seed42.stdout 2>&1
echo "\$(date): OWT held-out eval done (venus)."
WATCHER
chmod +x /home/benzene/llmopt/blt/queue_layersperm4_watcher.sh
nohup bash /home/benzene/llmopt/blt/queue_layersperm4_watcher.sh > /home/benzene/llmopt/blt/queue_layersperm4_watcher.log 2>&1 < /dev/null &
disown
echo "$(date): follow-up watcher for the new run armed (PID $!)."

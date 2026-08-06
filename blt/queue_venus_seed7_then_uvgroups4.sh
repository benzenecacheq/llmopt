#!/bin/bash
# Runs on bender, polling venus's training PID remotely (venus lacks direct
# ssh access to other machines, so orchestration happens from here) --
# same pattern as queue_sine_seed19_watcher.sh.
#
# 1. Waits for the sine-blend seed7 run on venus (PID 216841) to exit.
# 2. Benchmarks it (lm-eval-harness + OWT held-out), same protocol as seed19.
# 3. Launches the num_uv_groups=4 from-scratch OWT experiment on the now-free venus.
# 4. Confirms the new run starts healthy (correct param count, declining loss)
#    and sets up a follow-up watcher for its own eventual benchmarks.

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

echo "$(date): launching num_uv_groups=4 experiment on venus..."
ssh venus 'true; cd /home/benzene/llmopt/blt && git pull' > /home/benzene/llmopt/blt/queue_uvgroups4_gitpull.stdout 2>&1
echo "$(date): venus repo pulled -- see queue_uvgroups4_gitpull.stdout for any collisions to check by hand."

HAS_FLAG=$(ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python train.py --help 2>&1 | grep -c num-uv-groups')
if [ "$HAS_FLAG" -eq 0 ]; then
  echo "$(date): ABORT -- venus's train.py does not have --num-uv-groups after git pull (pull likely failed or hit an unresolved collision, see queue_uvgroups4_gitpull.stdout). NOT launching on stale code. Needs manual intervention."
  exit 1
fi
echo "$(date): confirmed --num-uv-groups present in venus's train.py, safe to launch."

LAUNCH_OUT=$(ssh venus 'true; cd /home/benzene/llmopt/blt && nohup ~/miniconda3/envs/blt/bin/python train.py --uv-rank 256 --num-uv-groups 4 --from-scratch --dataset openwebtext --seed 42 --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 --save-path run_blt_uv256_groups4_scratch_seed42.pt > run_blt_uv256_groups4_scratch_seed42.stdout 2>&1 < /dev/null & disown; sleep 3; ps aux | grep train.py | grep -v grep')
echo "$LAUNCH_OUT"
UV_PID=$(echo "$LAUNCH_OUT" | grep uv-rank | awk '{print $2}' | head -1)
echo "$(date): num_uv_groups=4 run launched on venus, PID $UV_PID"

sleep 120
echo "$(date): post-launch health check --"
ssh venus 'true; head -n 6 /home/benzene/llmopt/blt/run_blt_uv256_groups4_scratch_seed42.log; echo ---; tail -n 6 /home/benzene/llmopt/blt/run_blt_uv256_groups4_scratch_seed42.log'

if [ -z "$UV_PID" ]; then
  echo "$(date): WARNING -- could not determine PID for the new run, no follow-up watcher started. Check manually."
  exit 1
fi

cat > /home/benzene/llmopt/blt/queue_uvgroups4_watcher.sh <<WATCHER
#!/bin/bash
while ssh venus 'true; kill -0 $UV_PID' 2>/dev/null; do
  sleep 60
done
echo "\$(date): num_uv_groups=4 training (venus PID $UV_PID) exited, running benchmarks..."
ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python blt_lm_eval.py --checkpoint run_blt_uv256_groups4_scratch_seed42.pt --uv-rank 256 --num-uv-groups 4 --output lm_eval_blt_uv256_groups4_scratch_seed42.json' \\
  > /home/benzene/llmopt/blt/lm_eval_blt_uv256_groups4_scratch_seed42.stdout 2>&1
echo "\$(date): lm-eval-harness benchmark done (venus)."
ssh venus 'true; cd /home/benzene/llmopt/blt && ~/miniconda3/envs/blt/bin/python eval_owt.py --blt-lowrank-checkpoint run_blt_uv256_groups4_scratch_seed42.pt --uv-rank 256 --num-uv-groups 4' \\
  > /home/benzene/llmopt/blt/eval_owt_blt_uv256_groups4_scratch_seed42.stdout 2>&1
echo "\$(date): OWT held-out eval done (venus)."
WATCHER
chmod +x /home/benzene/llmopt/blt/queue_uvgroups4_watcher.sh
nohup bash /home/benzene/llmopt/blt/queue_uvgroups4_watcher.sh > /home/benzene/llmopt/blt/queue_uvgroups4_watcher.log 2>&1 < /dev/null &
disown
echo "$(date): follow-up watcher for the new run armed (PID $!)."

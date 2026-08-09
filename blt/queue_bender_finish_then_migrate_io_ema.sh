#!/bin/bash
# Runs on bender. Waits for bender's own medium GPT-2 baseline (PID 4761) to
# finish, benchmarks it, then stops io's medium sine-blend EMA run, copies its
# checkpoint over, and resumes it on bender instead -- at a different batch
# size (16GB card can't fit io's true batch=4), using the new
# --resume-batch-size fix so the data-sampler position stays correct across
# that batch-size change.

cd /home/benzene/llmopt/blt

echo "$(date): waiting for bender's medium baseline (PID 4761) to finish..."
while kill -0 4761 2>/dev/null; do
  sleep 60
done
echo "$(date): bender's medium baseline training exited."

if ! grep -q "Final val perplexity" run_gpt2_medium_baseline_seed42.log; then
  echo "$(date): WARNING -- 'Final val perplexity' not found in run_gpt2_medium_baseline_seed42.log. May have crashed rather than completed cleanly. Continuing, but this needs manual review."
fi

/home/benzene/miniconda3/envs/blt/bin/python check_bender_medium_ckpt.py

echo "$(date): benchmarking bender's finished medium baseline..."
~/miniconda3/bin/conda run -n blt python eval_owt.py --baseline-checkpoint run_gpt2_medium_baseline_seed42.pt --pretrained gpt2-medium \
  > eval_owt_gpt2_medium_baseline_seed42.stdout 2>&1
echo "$(date): OWT held-out eval done -- $(grep -a 'OWT held-out' eval_owt_gpt2_medium_baseline_seed42.stdout | tail -1)"

~/miniconda3/bin/conda run -n blt python blt_lm_eval.py --checkpoint run_gpt2_medium_baseline_seed42.pt --baseline --pretrained gpt2-medium \
  --output lm_eval_gpt2_medium_baseline_seed42.json \
  > lm_eval_gpt2_medium_baseline_seed42.stdout 2>&1
echo "$(date): lm-eval-harness benchmark done."

echo "$(date): stopping io's medium sine-blend EMA run (PID 2782)..."
ssh io 'true; kill -TERM 2782'
sleep 10
# NOTE: deliberately no 2>/dev/null here -- verified live that tcsh on io mangles
# that redirect inside this exact "kill -0 PID && echo yes || echo no" construct,
# making it return "no" unconditionally (tested against both a real running PID
# and a nonexistent one, both returned "no"). Without the redirect it's correct
# both ways; the harmless "No such process" stderr line just isn't captured by $().
STILL_UP=$(ssh io 'true; kill -0 2782 && echo yes || echo no')
echo "$(date): io training process still up after SIGTERM+10s: $STILL_UP"
if [ "$STILL_UP" = "yes" ]; then
  echo "$(date): giving it another 30s before treating this as stuck..."
  sleep 30
  STILL_UP2=$(ssh io 'true; kill -0 2782 && echo yes || echo no')
  echo "$(date): still up after total 40s: $STILL_UP2"
  if [ "$STILL_UP2" = "yes" ]; then
    echo "$(date): ABORT -- io's training process did not exit cleanly after SIGTERM. NOT touching the checkpoint or killing further. Needs manual intervention."
    exit 1
  fi
fi

IO_STEP=$(ssh io 'true; /home/benzene/miniconda3/envs/blt/bin/python /home/benzene/llmopt_medium_ema/blt/check_io_ema_ckpt.py')
echo "$(date): io checkpoint after stop -- step/health: $IO_STEP"
if echo "$IO_STEP" | grep -q CORRUPTED; then
  echo "$(date): ABORT -- io's checkpoint is corrupted after stopping. NOT copying or launching. Needs manual intervention."
  exit 1
fi

echo "$(date): copying io's checkpoint to bender..."
scp io:/home/benzene/llmopt_medium_ema/blt/run_gpt2_medium_ema_blend75_sine_seed42.pt /home/benzene/llmopt/blt/run_gpt2_medium_ema_blend75_sine_seed42.pt
scp io:/home/benzene/llmopt_medium_ema/blt/run_gpt2_medium_ema_blend75_sine_seed42.log /home/benzene/llmopt/blt/run_gpt2_medium_ema_blend75_sine_seed42_from_io.log

REMOTE_SIZE=$(ssh io 'true; stat -c %s /home/benzene/llmopt_medium_ema/blt/run_gpt2_medium_ema_blend75_sine_seed42.pt')
LOCAL_SIZE=$(stat -c %s /home/benzene/llmopt/blt/run_gpt2_medium_ema_blend75_sine_seed42.pt)
echo "$(date): checkpoint size check -- io: $REMOTE_SIZE, bender: $LOCAL_SIZE"
if [ "$REMOTE_SIZE" != "$LOCAL_SIZE" ]; then
  echo "$(date): ABORT -- copied checkpoint size mismatch. NOT launching on a possibly-truncated file. Needs manual intervention."
  exit 1
fi

/home/benzene/miniconda3/envs/blt/bin/python check_copied_ema_ckpt.py

echo "$(date): launching EMA sine-blend medium run on bender (batch-size 2, grad-accum 2, resume-batch-size 4 to correct the sampler position)..."
nohup ~/miniconda3/bin/conda run --no-capture-output -n blt python train.py \
  --resume run_gpt2_medium_ema_blend75_sine_seed42.pt --baseline --from-scratch --pretrained gpt2-medium \
  --ema-loss-weighting --ema-blend 0.75 --ema-blend-schedule sine --dataset openwebtext_large --seed 42 \
  --batch-size 2 --grad-accum-steps 2 --resume-batch-size 4 \
  --max-steps 1500000 --eval-every 1000 --lambada-eval-every 5000 \
  --save-path run_gpt2_medium_ema_blend75_sine_seed42.pt \
  > run_gpt2_medium_ema_blend75_sine_seed42_bender.stdout 2>&1 < /dev/null &
disown
NEW_PID=$!
echo "$(date): launched, outer PID $NEW_PID"

sleep 180
echo "$(date): post-launch health check --"
ps aux | grep train.py | grep -v grep
echo "--- log head ---"
head -n 8 run_gpt2_medium_ema_blend75_sine_seed42.log
echo "--- log tail ---"
tail -n 8 run_gpt2_medium_ema_blend75_sine_seed42.log
echo "--- GPU state ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1

echo "$(date): DONE. Review the health check above -- confirm 'Resumed at step' matches io's final step ($IO_STEP), effective_blend is sane, no OOM/crash in the stdout file."

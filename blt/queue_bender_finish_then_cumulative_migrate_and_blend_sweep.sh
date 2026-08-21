#!/bin/bash
# Orchestrates, in order:
#   1. Wait for bender's current medium EMA sine-blend job (PID 3705) to finish.
#   2. Benchmark that finished run.
#   3. Stop titan's cumulative-mode from-scratch run cleanly, migrate it to
#      bender, resume there (bender becomes the new home for the full
#      500K-step "full monty" run, blend=1.0 baseline).
#   4. Titan is now free -- launch a clean from-scratch cumulative-mode run
#      at blend=0.5 there (mirrors venus's blend=0.75 run: no snapshot
#      dependency, avoids the undertrained-model confound the original
#      fine-tune-from-100K-snapshot plan had -- see CLAUDE.md 2026-08-21).
#
# Together with venus's blend=0.75 run and this migrated blend=1.0 "full
# monty", this gives three clean from-scratch points on the cumulative-mode
# blend curve (0.5, 0.75, 1.0) plus the existing non-EMA baseline (0.0).
set -x
cd /home/benzene/llmopt/blt
LOG=queue_bender_finish_then_cumulative_migrate_and_blend_sweep.log
exec >> "$LOG" 2>&1

echo "=== $(date): watcher started, waiting on bender PID 3705 ==="
while kill -0 3705 2>/dev/null; do
  sleep 60
done
echo "=== $(date): bender medium job exited, benchmarking ==="

~/miniconda3/bin/conda run --no-capture-output -n blt python eval_owt.py \
  --baseline-checkpoint run_gpt2_medium_ema_blend75_sine_seed42.pt --pretrained gpt2-medium \
  > eval_owt_gpt2_medium_ema_blend75_sine_seed42.stdout 2>&1
echo "$(date): medium run OWT eval done."

~/miniconda3/bin/conda run --no-capture-output -n blt python blt_lm_eval.py \
  --checkpoint run_gpt2_medium_ema_blend75_sine_seed42.pt --baseline --pretrained gpt2-medium \
  --output lm_eval_gpt2_medium_ema_blend75_sine_seed42.json \
  > lm_eval_gpt2_medium_ema_blend75_sine_seed42.stdout 2>&1
echo "$(date): medium run lm-eval done."

echo "=== $(date): stopping titan cumulative training cleanly ==="
TITAN_PID=2894048
echo "titan training PID: $TITAN_PID"
if ssh titan "ps -p $TITAN_PID -o cmd=" | grep -q run_gpt2_cumulative_scratch_seed42; then
  ssh titan "kill -TERM $TITAN_PID"
  sleep 15
else
  echo "$(date): ABORTING -- expected PID not found on titan, needs manual intervention"
  exit 1
fi

cat > /tmp/verify_finite.py << 'PYEOF'
import torch, sys
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
bad = [n for n, t in ckpt['model_state'].items() if not torch.isfinite(t).all()]
print('step:', ckpt.get('step'), 'bad params:', bad if bad else 'NONE -- all finite')
sys.exit(1 if bad else 0)
PYEOF
scp /tmp/verify_finite.py titan:/tmp/verify_finite.py
ssh titan "~/miniconda3/envs/blt/bin/python /tmp/verify_finite.py ~/llmopt/blt/run_gpt2_cumulative_scratch_seed42.pt"
echo "$(date): titan checkpoint verified finite."

echo "=== $(date): migrating checkpoint + log to bender ==="
scp titan:~/llmopt/blt/run_gpt2_cumulative_scratch_seed42.pt /home/benzene/llmopt/blt/run_gpt2_cumulative_scratch_seed42.pt
scp titan:~/llmopt/blt/run_gpt2_cumulative_scratch_seed42.log /home/benzene/llmopt/blt/run_gpt2_cumulative_scratch_seed42.log

echo "=== $(date): resuming cumulative run on bender ==="
cd /home/benzene/llmopt/blt
nohup conda run --no-capture-output -n blt python train.py \
  --resume run_gpt2_cumulative_scratch_seed42.pt \
  --baseline --from-scratch --ema-loss-weighting --loss-weighting-mode cumulative \
  --dataset openwebtext --seed 42 \
  --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 --owt-eval-every 5000 \
  --save-path run_gpt2_cumulative_scratch_seed42.pt \
  > run_gpt2_cumulative_scratch_seed42_bender_resume.stdout 2>&1 &
BENDER_PID=$!
echo "$(date): resumed on bender, PID $BENDER_PID"
sleep 60
if kill -0 "$BENDER_PID" 2>/dev/null && grep -q "Resumed at step" run_gpt2_cumulative_scratch_seed42.log; then
  echo "$(date): bender resume VERIFIED healthy ($(tail -1 run_gpt2_cumulative_scratch_seed42.log))"
else
  echo "$(date): *** BENDER RESUME FAILED OR DIED -- NEEDS MANUAL INTERVENTION *** see run_gpt2_cumulative_scratch_seed42_bender_resume.stdout"
fi

echo "=== $(date): titan is now free -- launching blend=0.5 from-scratch cumulative run ==="
ssh titan "cd ~/llmopt/blt && nohup ~/miniconda3/envs/blt/bin/python train.py --baseline --from-scratch --ema-loss-weighting --loss-weighting-mode cumulative --ema-blend 0.5 --dataset openwebtext --seed 42 --max-steps 500000 --eval-every 1000 --lambada-eval-every 5000 --owt-eval-every 5000 --save-path run_gpt2_cumulative_blend50_scratch_seed42.pt > run_gpt2_cumulative_blend50_scratch_seed42.stdout 2>&1 &"
sleep 30
echo "$(date): titan blend=0.5 launch attempted, verifying..."
ssh titan "ps aux | grep cumulative_blend50 | grep -v grep"

echo "=== $(date): orchestration complete ==="

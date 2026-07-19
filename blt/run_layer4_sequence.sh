#!/bin/bash
# Sequence: wait for small model warm-start → benchmarks → XL warm-start
set -e
CONDA="$HOME/miniconda3/bin/conda run -n blt"
LOG="run_layer4_sequence.log"

echo "[$(date)] Waiting for small model warm-start to finish..." | tee -a $LOG
until grep -q "Final val perplexity" run_blt_layer4_warmstart_small.log 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Small model done." | tee -a $LOG

echo "[$(date)] Running benchmarks..." | tee -a $LOG
$CONDA python blt_lm_eval.py \
    --checkpoint run_blt_layer4_warmstart_small.pt \
    --layers-per-m 4 \
    --tasks lambada_openai,hellaswag,piqa,winogrande \
    --output lm_eval_blt_layer4_warmstart_small.json \
    >> run_layer4_sequence.log 2>&1
echo "[$(date)] Benchmarks done." | tee -a $LOG

echo "[$(date)] Launching XL warm-start..." | tee -a $LOG
$CONDA python train.py \
    --pretrained gpt2-xl \
    --dataset wikitext103 \
    --max-steps 50300 \
    --warmup-steps 2000 \
    --layers-per-m 4 \
    --warmstart-scale 0.25 \
    --batch-size 4 \
    --fp16 \
    --grad-checkpointing \
    --eval-every 1000 \
    --lambada-eval-every 5000 \
    --save-path run_blt_xl_layer4_warmstart.pt \
    >> run_blt_xl_layer4_warmstart.stdout 2>&1
echo "[$(date)] XL warm-start done." | tee -a $LOG

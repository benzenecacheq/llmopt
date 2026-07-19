#!/bin/bash
# Sequence: BLT 3-M warm-start → benchmarks → GQA-3 warm-start → benchmarks
set -e
CONDA="$HOME/miniconda3/bin/conda run -n blt"
LOG="run_3m_gqa3_sequence.log"

echo "[$(date)] Starting BLT 3-M warm-start (small model, WikiText-103)..." | tee -a $LOG
$CONDA python train.py \
    --pretrained gpt2 \
    --dataset wikitext103 \
    --max-steps 50300 \
    --num-m-groups 3 \
    --save-path run_blt_3m_warmstart_small.pt \
    >> run_blt_3m_warmstart_small.stdout 2>&1
echo "[$(date)] BLT 3-M done." | tee -a $LOG

echo "[$(date)] Running BLT 3-M benchmarks..." | tee -a $LOG
$CONDA python blt_lm_eval.py \
    --checkpoint run_blt_3m_warmstart_small.pt \
    --num-m-groups 3 \
    --tasks lambada_openai,hellaswag,piqa,winogrande \
    --output lm_eval_blt_3m_warmstart_small.json \
    >> run_3m_gqa3_sequence.log 2>&1
echo "[$(date)] BLT 3-M benchmarks done." | tee -a $LOG

echo "[$(date)] Starting GQA-3 warm-start (small model, WikiText-103)..." | tee -a $LOG
$CONDA python train.py \
    --pretrained gpt2 \
    --dataset wikitext103 \
    --max-steps 50300 \
    --gqa --gqa-groups 3 \
    --save-path run_gqa3_warmstart_small.pt \
    >> run_gqa3_warmstart_small.stdout 2>&1
echo "[$(date)] GQA-3 done." | tee -a $LOG

echo "[$(date)] Running GQA-3 benchmarks..." | tee -a $LOG
$CONDA python blt_lm_eval.py \
    --checkpoint run_gqa3_warmstart_small.pt \
    --gqa --gqa-groups 3 \
    --tasks lambada_openai,hellaswag,piqa,winogrande \
    --output lm_eval_gqa3_warmstart_small.json \
    >> run_3m_gqa3_sequence.log 2>&1
echo "[$(date)] GQA-3 benchmarks done." | tee -a $LOG

echo "[$(date)] All done." | tee -a $LOG

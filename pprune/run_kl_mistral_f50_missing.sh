#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
echo "=== KL Mistral f50 missing (naive_50pct, pyramidkv_f50) started at $(date) ==="
python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --methods naive_50pct,pyramidkv_f50 \
    --output lb_results_base/kl_ystar_mistral_f50.json \
    --log lb_results_base/kl_ystar_mistral_f50.log \
    --n 100
echo "=== done at $(date) ==="

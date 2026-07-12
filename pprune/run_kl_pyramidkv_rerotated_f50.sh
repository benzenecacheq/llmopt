#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
echo "=== pyramidkv_rerotated_f50 KL eval started at $(date) ==="
python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods pyramidkv_rerotated_f50 \
    --output lb_results_base/kl_pyramidkv_rerotated_f50.json \
    --log lb_results_base/kl_pyramidkv_rerotated_f50.log \
    --n 100
echo "=== done at $(date) ==="

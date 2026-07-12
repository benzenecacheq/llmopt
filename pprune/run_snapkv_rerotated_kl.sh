#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

echo "=== snapkv_rerotated KL eval started at $(date) ==="
python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods snapkv_rerotated \
    --output lb_results_base/kl_snapkv_rerotated.json \
    --log lb_results_base/kl_snapkv_rerotated.log \
    --n 100
echo "=== done at $(date) ==="

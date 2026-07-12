#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
echo "=== snapkv_rerotated KL Mistral (65/50/35%) started at $(date) ==="
python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --methods snapkv_rerotated \
    --output lb_results_base/kl_snapkv_rerotated_mistral.json \
    --log lb_results_base/kl_snapkv_rerotated_mistral.log \
    --n 100
python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --methods snapkv_rerotated_f50 \
    --output lb_results_base/kl_snapkv_rerotated_mistral_f50.json \
    --log lb_results_base/kl_snapkv_rerotated_mistral_f50.log \
    --n 100
python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --methods snapkv_rerotated_f35 \
    --output lb_results_base/kl_snapkv_rerotated_mistral_f35.json \
    --log lb_results_base/kl_snapkv_rerotated_mistral_f35.log \
    --n 100
echo "=== done at $(date) ==="

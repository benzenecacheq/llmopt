#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
echo "=== KL streaming_press f50/f35 (unrotated, for §6.4 table) started at $(date) ==="
python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods streaming_press_f50,streaming_press_f35 \
    --output lb_results_base/kl_streaming_press_f50f35.json \
    --log lb_results_base/kl_streaming_press_f50f35.log \
    --n 100
echo "=== done at $(date) ==="

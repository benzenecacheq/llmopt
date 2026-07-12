#!/bin/bash
# Initial RADAR probe: radar_pre_rerotated and radar_post_rerotated at 65% and 35%.
# Six canonical tasks, n=100. If neither is close to snapkv_rerotated (KL=0.043 at 65%,
# 0.124 at 35%), abandon RADAR. If competitive, expand to all 16 tasks + Mistral + GT.
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="2wikimqa,multifieldqa_en,qasper,qmsum,repobench-p,triviaqa"

echo "=== RADAR KL probe started at $(date) ==="

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods radar_pre_rerotated,radar_post_rerotated \
    --tasks "$TASKS" \
    --output lb_results_base/kl_radar_probe_f65.json \
    --log lb_results_base/kl_radar_probe_f65.log \
    --n 100

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods radar_pre_rerotated_f35,radar_post_rerotated_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_radar_probe_f35.json \
    --log lb_results_base/kl_radar_probe_f35.log \
    --n 100

echo "=== RADAR KL probe complete at $(date) ==="

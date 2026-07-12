#!/bin/bash
# Full KL eval for radar_post_rerotated: all 16 tasks, all 3 paper rates (65/50/35%).
# radar_post_rerotated matched snapkv_rerotated on the 6-task probe (0.034 vs 0.033).
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

echo "=== radar_post_rerotated full KL eval started at $(date) ==="

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods radar_post_rerotated \
    --tasks "$TASKS" \
    --output lb_results_base/kl_radar_post_rerotated.json \
    --log lb_results_base/kl_radar_post_rerotated.log \
    --n 100

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods radar_post_rerotated_f50 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_radar_post_rerotated_f50.json \
    --log lb_results_base/kl_radar_post_rerotated_f50.log \
    --n 100

python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --methods radar_post_rerotated_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_radar_post_rerotated_f35.json \
    --log lb_results_base/kl_radar_post_rerotated_f35.log \
    --n 100

echo "=== radar_post_rerotated full KL eval complete at $(date) ==="

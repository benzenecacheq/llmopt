#!/bin/bash
# GT eval: radar_post_rerotated at 65/50/35% on Llama-3.1-8B, all 16 tasks, n=100.
# radar_post_rerotated matched snapkv_rerotated on KL at both 65% and 35% probes
# (avg 0.034 vs 0.033 at 65%; 0.100 vs 0.100 at 35%). GT run confirms selection
# strategy irrelevance extends to output faithfulness.
# Checkpoints are resumable — already-completed examples are skipped.
set -e
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
LLAMA="meta-llama/Llama-3.1-8B"

echo "=== GT radar_post_rerotated sweep started at $(date) ==="

python gt_eval_compression.py --model "$LLAMA" --tasks "$TASKS" \
    --methods radar_post_rerotated \
    --output lb_results_base/gt_radar_post_rerotated \
    --n 100

python gt_eval_compression.py --model "$LLAMA" --tasks "$TASKS" \
    --methods radar_post_rerotated_f50 \
    --output lb_results_base/gt_radar_post_rerotated_f50 \
    --n 100

python gt_eval_compression.py --model "$LLAMA" --tasks "$TASKS" \
    --methods radar_post_rerotated_f35 \
    --output lb_results_base/gt_radar_post_rerotated_f35 \
    --n 100

echo "=== GT radar_post_rerotated sweep complete at $(date) ==="

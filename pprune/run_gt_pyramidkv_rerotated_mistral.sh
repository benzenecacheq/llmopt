#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
echo "=== GT pyramidkv_rerotated Mistral (65/50/35%) started at $(date) ==="
python gt_eval_compression.py --model mistralai/Mistral-7B-v0.3 --tasks "$TASKS" \
    --methods pyramidkv_rerotated \
    --output lb_results_base/gt_pyramidkv_rerotated_mistral \
    --n 100
python gt_eval_compression.py --model mistralai/Mistral-7B-v0.3 --tasks "$TASKS" \
    --methods pyramidkv_rerotated_f50 \
    --output lb_results_base/gt_pyramidkv_rerotated_mistral_f50 \
    --n 100
python gt_eval_compression.py --model mistralai/Mistral-7B-v0.3 --tasks "$TASKS" \
    --methods pyramidkv_rerotated_f35 \
    --output lb_results_base/gt_pyramidkv_rerotated_mistral_f35 \
    --n 100
echo "=== done at $(date) ==="

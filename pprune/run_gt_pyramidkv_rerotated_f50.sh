#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
echo "=== GT pyramidkv_rerotated_f50 started at $(date) ==="
python gt_eval_compression.py --model meta-llama/Llama-3.1-8B --tasks "$TASKS" \
    --methods pyramidkv_rerotated_f50 \
    --output lb_results_base/gt_pyramidkv_rerotated_f50 \
    --n 100
echo "=== done at $(date) ==="

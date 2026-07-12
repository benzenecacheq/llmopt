#!/bin/bash
# GT sweep: snapkv_rerotated at 65% retention (Llama only, diagnostic).
set -e
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
LLAMA="meta-llama/Llama-3.1-8B"

echo "=== GT snapkv_rerotated sweep started at $(date) ==="
python gt_eval_compression.py --model "$LLAMA" --tasks "$TASKS" \
    --methods snapkv_rerotated \
    --output lb_results_base/gt_snapkv_rerotated \
    --n 100
echo "=== GT snapkv_rerotated sweep done at $(date) ==="

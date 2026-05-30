#!/bin/bash
# GT evaluation for pyramidkv_f35_clean_first: PyramidKV at 35% retention (keep 35%,
# compression_ratio=0.65) with the clean-first-token fix.
#
# Standard pyramidkv_f35 has the same first-token advantage as pyramidkv:
# T1 comes from dense attention over all tokens during prefill.  This run
# removes that asymmetry: full_ids[:, :-1] is compressed first, and
# model.generate starts from last_tok using the compressed cache.
#
# Compare against:
#   pyramidkv_f35   → lb_results_base/gt_pyramidkv_f35/   (if it exists)
#   pyramidkv_clean_first (65% retention) → lb_results_base/gt_pyramidkv_clean_first/
#
# Key question: at more aggressive compression (35% retention), does the
# full-prefill first-token advantage matter more?
#
# Output → lb_results_base/gt_pyramidkv_f35_clean_first/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_pyramidkv_f35_clean_first.log
exec >> "$LOG" 2>&1

echo "=== run_gt_pyramidkv_f35_clean_first.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_f35_clean_first \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_f35_clean_first \
    --n 100

echo "=== run_gt_pyramidkv_f35_clean_first.sh complete at $(date) ==="

#!/bin/bash
# GT evaluation for pyramidkv_clean_first: same PyramidKV compression (keep 65%),
# but T1 attends to the compressed KV cache rather than the full-context prefill.
#
# Standard pyramidkv has a first-token advantage: the step-0 logit comes from
# dense attention over all 7168 tokens, while T2+ attend only to the compressed
# cache (~4659 entries).  This run removes that asymmetry: full_ids[:, :-1] is
# compressed first, and model.generate starts from last_tok using that compressed
# cache.  T1 (and all subsequent tokens) see the same compressed context.
#
# Compare against:
#   pyramidkv (full)       → lb_results_base/gt_pyramidkv/
#   pyramidkv_short        → lb_results_base/gt_pyramidkv_short/
#
# If pyramidkv_clean_first ≈ pyramidkv: the full-prefill first token doesn't
# drive GT scores; compression quality does.
# If pyramidkv_clean_first drops significantly: the first-token advantage is real.
#
# Output → lb_results_base/gt_pyramidkv_clean_first/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_pyramidkv_clean_first.log
exec >> "$LOG" 2>&1

echo "=== run_gt_pyramidkv_clean_first.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_clean_first \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_clean_first \
    --n 100

echo "=== run_gt_pyramidkv_clean_first.sh complete at $(date) ==="

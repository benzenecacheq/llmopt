#!/bin/bash
# Robustness check: does adding head_frac=0.10 to phrase methods help or hurt on
# tasks that already have a real query signal (i.e. everywhere except
# gov_report/multi_news)? Runs the h10 variants across all 16 LongBench tasks at
# 65% retention; compare against the existing n=100 phr128/phr160 numbers already
# in the paper's main tables.
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/headfrac_robustness_gt.log
exec >> "$LOG" 2>&1
echo "=== started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p" \
    --methods "chunk_word128_t20_h10,chunk_word160_t25_h10" \
    --output lb_results_base/gt_headfrac_robustness \
    --n 25

echo "=== complete at $(date) ==="

#!/bin/bash
# Per-step KL experiment: test PyramidKV T0-advantage hypothesis.
#
# Two experiments:
#   1. Filtered LongBench: gov_report, qmsum, multi_news (long-answer tasks),
#      reusing existing y* cache, filtering to examples with >=50 generated tokens.
#   2. Synthetic long-answer prompts (20 examples, max_new_tokens=300).
#
# Both run on Llama-3.1-8B, 4 methods: Naive, phr128, SnapKV, PyramidKV.
# Analyze with: python analyze_perstep.py --input <output_files>

set -e
LOG=lb_results_base/perstep_eval.log
exec >> "$LOG" 2>&1
echo "=== run_perstep_eval.sh started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

METHODS="naive_65pct,chunk_word128_t20,snapkv,pyramidkv"
MODEL="meta-llama/Llama-3.1-8B"

echo "--- Fictional long-answer prompts (6 docs, 4 methods) ---"
python perStep_kl_eval.py \
    --model "$MODEL" \
    --data_dir synthetic_prompts \
    --tasks fictional_long \
    --methods "$METHODS" \
    --output lb_results_base/perstep_kl_fictional.json \
    --n 6 \
    --max_new_tokens 300

echo "--- Analysis ---"
python analyze_perstep.py \
    --input lb_results_base/perstep_kl_fictional.json \
    --methods "$METHODS" \
    --max_step 30

echo "=== run_perstep_eval.sh complete at $(date) ==="

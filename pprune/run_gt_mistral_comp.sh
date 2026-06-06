#!/bin/bash
# GT cross-rate eval for Mistral-7B-v0.3, matching Llama §7.4 coverage.
# Runs default RATIO_BATCHES (RADAR removed from paper):
#   65%: chunk_word160_t25, chunk_word128_t20, snapkv_select
#   50%: naive_50pct, chunk_word160_f50, chunk_word128_f50, snapkv_select_f50, snapkv_50pct
#   40%: naive_40pct, chunk_word160_f40, chunk_word128_f40, snapkv_select_f40, snapkv_40pct
#   35%: naive_35pct, chunk_word160_f35, chunk_word128_f35, snapkv_select_f35, snapkv_35pct
# Also adds pyramidkv at all rates via a separate pass.
# Output → lb_results_base/gt_mistral_comp/

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/gt_mistral_comp.log
exec >> "$LOG" 2>&1

echo "=== run_gt_mistral_comp.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

# Pass 1: pyramidkv_f35 (already done, checkpoint will skip)
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods pyramidkv_f35 \
    --output lb_results_base/gt_mistral_comp \
    --n 100

# Pass 2: chunk methods at 65%
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods chunk_word160_t25,chunk_word128_t20,snapkv_select \
    --output lb_results_base/gt_mistral_comp \
    --n 100

# Pass 3: all clean_first variants — run early for diagnostic value
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods pyramidkv_clean_first,pyramidkv_f35_clean_first,pyramidkv_f50_clean_first,pyramidkv_f40_clean_first \
    --output lb_results_base/gt_mistral_comp \
    --n 100

# Pass 4: 50/40/35% for naive + chunk + snapkv
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods naive_50pct,chunk_word160_t25_f50,chunk_word128_t20_f50,snapkv_select_f50,snapkv_50pct \
    --output lb_results_base/gt_mistral_comp \
    --n 100

python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods naive_40pct,chunk_word160_t25_f40,chunk_word128_t20_f40,snapkv_select_f40,snapkv_40pct \
    --output lb_results_base/gt_mistral_comp \
    --n 100

python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods naive_35pct,chunk_word160_t25_f35,chunk_word128_t20_f35,snapkv_select_f35,snapkv_35pct \
    --output lb_results_base/gt_mistral_comp \
    --n 100

# Pass 5: pyramidkv at 50% and 40%
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tasks "$TASKS" \
    --methods pyramidkv_f50,pyramidkv_f40 \
    --output lb_results_base/gt_mistral_comp \
    --n 100

echo "=== run_gt_mistral_comp.sh complete at $(date) ==="

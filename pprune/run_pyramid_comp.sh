#!/bin/bash
# Pyramid KV compression-ratio sweep — KL then GT on Llama, then Mistral GT.
#
# Run order:
#   1. KL eval  — pyramidkv_f50/f40/f35 on Llama (all 16 tasks)
#   2. GT eval  — pyramidkv_f50/f40/f35 on Llama (all 16 tasks)
#   3. GT eval  — pyramidkv (65%) on Mistral-7B-v0.3 (all 16 tasks)

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/pyramid_comp.log
exec >> "$LOG" 2>&1

echo "=== run_pyramid_comp.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

# ── 1. KL eval: pyramidkv f50/f40/f35 on Llama ─────────────────────────────
echo ""
echo "=== [1/3] KL eval (pyramidkv f50/f40/f35, Llama) at $(date) ==="
python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/kl_ystar_pyramidkv_comp.json \
    --ystar_cache lb_results_base/ystar_cache.pt \
    --log lb_results_base/kl_ystar_pyramidkv_comp.log

echo "=== KL eval done at $(date) ==="

# ── 2. GT eval: pyramidkv f50/f40/f35 on Llama ─────────────────────────────
echo ""
echo "=== [2/3] GT eval (pyramidkv f50/f40/f35, Llama) at $(date) ==="
python gt_eval_compression.py \
    --model meta-llama/Llama-3.1-8B \
    --methods pyramidkv_f50,pyramidkv_f40,pyramidkv_f35 \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_comp \
    --n 100

echo "=== Llama GT eval done at $(date) ==="

# ── 3. GT eval: pyramidkv (65%) on Mistral-7B-v0.3 ─────────────────────────
echo ""
echo "=== [3/3] GT eval (pyramidkv, Mistral) at $(date) ==="
python gt_eval_compression.py \
    --model mistralai/Mistral-7B-v0.3 \
    --methods pyramidkv \
    --tasks "$TASKS" \
    --output lb_results_base/gt_pyramidkv_mistral \
    --n 100

echo "=== Mistral GT eval done at $(date) ==="
echo "=== All pyramid comp runs complete at $(date) ==="

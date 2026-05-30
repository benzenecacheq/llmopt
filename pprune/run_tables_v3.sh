#!/bin/bash
# run_tables_v3.sh
# Recompute all KL table values with corrected y* (ystar_cache_v3.pt).
#
# Fixes applied before this run:
#   1. y* stop tokens: short-answer tasks now stop at newline (token 198/271).
#      Old y* ran to max_new_tokens, contaminating y* with few-shot continuation.
#   2. kvpress off-by-one: first logit now captured before del out.
#      (PyramidKV was separately corrected in kl_ystar_pyramidkv_all_v2.json.)
#
# Methods covered:
#   Table 1/2 (§7.2): naive x4, snapkv x4, snapkv_select x4, cw160 x4, cw128 x4
#   §4.4:             kq_post_rope, kq_only, streaming
#   §5.2:             radar_select (RADAR-Select prompt construction)
#
# All 16 LongBench tasks, n=100. y* loaded from ystar_cache_v3.pt (no regeneration).
# Output: lb_results_base/kl_ystar_tables_v3.json

set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/tables_v3.log
exec >> "$LOG" 2>&1

echo "=== run_tables_v3.sh started at $(date) ==="

source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

CACHE=lb_results_base/ystar_cache_v3.pt
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

METHODS="naive_65pct,naive_50pct,naive_40pct,naive_35pct,\
snapkv,snapkv_50pct,snapkv_40pct,snapkv_35pct,\
snapkv_select,snapkv_select_f50,snapkv_select_f40,snapkv_select_f35,\
chunk_word160_t25,chunk_word160_t25_f50,chunk_word160_t25_f40,chunk_word160_t25_f35,\
chunk_word128_t20,chunk_word128_t20_f50,chunk_word128_t20_f40,chunk_word128_t20_f35,\
kq_post_rope,kq_only,streaming,radar_select"

python kl_faith_eval_ystar.py \
    --methods "$METHODS" \
    --tasks "$TASKS" \
    --n 100 \
    --output lb_results_base/kl_ystar_tables_v3.json \
    --ystar_cache "$CACHE" \
    --log lb_results_base/kl_ystar_tables_v3_progress.log

echo "=== run_tables_v3.sh complete at $(date) ==="

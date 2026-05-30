#!/bin/bash
set -e
cd /home/benzene/llmopt/pprune

TASKS_6="2wikimqa,multifieldqa_en,repobench-p,triviaqa,qasper,qmsum"
TASKS_10="hotpotqa,narrativeqa,musique,gov_report,multi_news,passage_count,passage_retrieval_en,samsum,lcc,trec"
METHODS_TIMING="chunk_word160_t25,chunk_word160_t25_f50,chunk_word160_t25_f40,chunk_word160_t25_f35,chunk_word128_t20,chunk_word128_t20_f50,chunk_word128_t20_f40,chunk_word128_t20_f35,snapkv_select,snapkv_select_f50,snapkv_select_f40,snapkv_select_f35"
METHODS_FULL="chunk_word160_t25,chunk_word128_t20,snapkv_select,naive_65pct,phrase_w64_t25"

echo "=== Stage 1: Timing + compression sweep (6 tasks) ===" | tee -a lb_results_base/overnight.log
conda run -n llmopt python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --data_dir lb_data_raw/data \
    --tasks $TASKS_6 \
    --methods $METHODS_TIMING \
    --n 100 \
    --timing \
    --output lb_results_base/kl_ystar_timing_sweep.json \
    --ystar_cache lb_results_base/ystar_cache.pt \
    --log lb_results_base/kl_ystar_timing_sweep.log
echo "Stage 1 complete." | tee -a lb_results_base/overnight.log

echo "=== Stage 2: 10 remaining tasks (Llama) ===" | tee -a lb_results_base/overnight.log
conda run -n llmopt python kl_faith_eval_ystar.py \
    --model meta-llama/Llama-3.1-8B \
    --data_dir lb_data_raw/data \
    --tasks $TASKS_10 \
    --methods $METHODS_FULL \
    --n 100 \
    --output lb_results_base/kl_ystar_remaining_tasks.json \
    --ystar_cache lb_results_base/ystar_cache_remaining.pt \
    --log lb_results_base/kl_ystar_remaining_tasks.log
echo "Stage 2 complete." | tee -a lb_results_base/overnight.log

echo "=== Stage 3: Mistral — 6 primary tasks ===" | tee -a lb_results_base/overnight.log
conda run -n llmopt python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --data_dir lb_data_raw/data \
    --tasks $TASKS_6 \
    --methods $METHODS_FULL \
    --n 100 \
    --output lb_results_base/kl_ystar_mistral_primary.json \
    --ystar_cache lb_results_base/ystar_cache_mistral.pt \
    --log lb_results_base/kl_ystar_mistral_primary.log
echo "Stage 3 complete." | tee -a lb_results_base/overnight.log

echo "=== Stage 4: Mistral — 10 remaining tasks ===" | tee -a lb_results_base/overnight.log
conda run -n llmopt python kl_faith_eval_ystar.py \
    --model mistralai/Mistral-7B-v0.3 \
    --data_dir lb_data_raw/data \
    --tasks $TASKS_10 \
    --methods $METHODS_FULL \
    --n 100 \
    --output lb_results_base/kl_ystar_mistral_remaining.json \
    --ystar_cache lb_results_base/ystar_cache_mistral_remaining.pt \
    --log lb_results_base/kl_ystar_mistral_remaining.log
echo "Stage 4 complete." | tee -a lb_results_base/overnight.log

echo "=== All stages complete ===" | tee -a lb_results_base/overnight.log

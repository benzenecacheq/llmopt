#!/bin/bash
# Full h10-only sweep: chunk_word128_t20_h10 and chunk_word160_t25_h10 (the two
# phrase methods with head_frac=0.10), n=100, all 16 LongBench tasks, at all
# four compression levels (35/40/50/65%), Llama first then Mistral.
# Single GPU -> strictly sequential.
set -e
LOG=/home/benzene/llmopt/pprune/lb_results_base/h10_full_sweep.log
exec >> "$LOG" 2>&1
echo "=== run_h10_full_sweep.sh started at $(date) ==="
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

run_rate () {
    MODEL="$1"; OUT="$2"; METHODS="$3"; TAG="$4"
    echo "--- [$TAG] model=$MODEL methods=$METHODS ---"
    python gt_eval_compression.py \
        --model "$MODEL" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --output "$OUT" \
        --n 100
    echo "--- [$TAG] done at $(date) ---"
}

LLAMA="meta-llama/Llama-3.1-8B"
MISTRAL="mistralai/Mistral-7B-v0.3"

run_rate "$LLAMA"   lb_results_base/gt_h10_llama_f35 "chunk_word128_t20_h10_f35,chunk_word160_t25_h10_f35" "llama 35%"
run_rate "$LLAMA"   lb_results_base/gt_h10_llama_f40 "chunk_word128_t20_h10_f40,chunk_word160_t25_h10_f40" "llama 40%"
run_rate "$LLAMA"   lb_results_base/gt_h10_llama_f50 "chunk_word128_t20_h10_f50,chunk_word160_t25_h10_f50" "llama 50%"
run_rate "$LLAMA"   lb_results_base/gt_h10_llama_f65 "chunk_word128_t20_h10,chunk_word160_t25_h10"         "llama 65%"

run_rate "$MISTRAL" lb_results_base/gt_h10_mistral_f35 "chunk_word128_t20_h10_f35,chunk_word160_t25_h10_f35" "mistral 35%"
run_rate "$MISTRAL" lb_results_base/gt_h10_mistral_f40 "chunk_word128_t20_h10_f40,chunk_word160_t25_h10_f40" "mistral 40%"
run_rate "$MISTRAL" lb_results_base/gt_h10_mistral_f50 "chunk_word128_t20_h10_f50,chunk_word160_t25_h10_f50" "mistral 50%"
run_rate "$MISTRAL" lb_results_base/gt_h10_mistral_f65 "chunk_word128_t20_h10,chunk_word160_t25_h10"         "mistral 65%"

echo "=== run_h10_full_sweep.sh complete at $(date) ==="

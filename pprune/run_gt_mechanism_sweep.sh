#!/bin/bash
# GT sweep: snapkv_press + streaming_rerotated only (no streaming_select).
# Checkpoints are resumable — already-completed examples are skipped.
set -e
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune

TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
LLAMA="meta-llama/Llama-3.1-8B"
MISTRAL="mistralai/Mistral-7B-v0.3"

run() {
    MODEL="$1"; OUT="$2"; METHODS="$3"; TAG="$4"
    echo "--- [$TAG] methods=$METHODS ---"
    python gt_eval_compression.py --model "$MODEL" --tasks "$TASKS" \
        --methods "$METHODS" --output "$OUT" --n 100
    echo "--- [$TAG] done at $(date) ---"
}

echo "=== GT mechanism sweep (no streaming_select) started at $(date) ==="

run "$LLAMA"   lb_results_base/gt_mechanism_llama_f65  "snapkv_press,streaming_rerotated"         "Llama 65%"
run "$LLAMA"   lb_results_base/gt_mechanism_llama_f50  "snapkv_press_f50,streaming_rerotated_f50" "Llama 50%"
run "$LLAMA"   lb_results_base/gt_mechanism_llama_f35  "snapkv_press_f35,streaming_rerotated_f35" "Llama 35%"
run "$MISTRAL" lb_results_base/gt_mechanism_mistral_f65  "snapkv_press,streaming_rerotated"         "Mistral 65%"
run "$MISTRAL" lb_results_base/gt_mechanism_mistral_f50  "snapkv_press_f50,streaming_rerotated_f50" "Mistral 50%"
run "$MISTRAL" lb_results_base/gt_mechanism_mistral_f35  "snapkv_press_f35,streaming_rerotated_f35" "Mistral 35%"

echo "=== GT mechanism sweep complete at $(date) ==="

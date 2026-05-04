#!/bin/tcsh
# pret_faith.csh
# Generate 100 examples of passage_retrieval_en for kqpr_md 0.7, 0.8, 0.9
# (0.6 already has 100 from the full eval), then run perplexity faithfulness
# on passage_retrieval_en for all four decay values.

setenv PYTHONUNBUFFERED 1

set MODEL  = "meta-llama/Llama-3.1-8B"
set OUTPUT = "lb_results_base"
set TASK   = "passage_retrieval_en"

echo "================================================================"
echo "pret_faith: generate + score passage_retrieval_en, md=0.7/0.8/0.9"
echo "================================================================"

foreach DECAY (0.9 0.8 0.7)
    set LABEL = "kqpr_md${DECAY}"
    echo ""
    echo "--- min_decay=${DECAY}  label=${LABEL} ---"
    conda run -n llmopt python longbench_eval.py \
        --model          $MODEL \
        --tasks          $TASK \
        --max_examples   100 \
        --output         $OUTPUT \
        --budget_fraction 0.65 \
        --score_mode     kq_post_rope \
        --min_decay      $DECAY \
        --method_label   $LABEL \
        --naive_fraction 0.0
    echo "Finished min_decay=${DECAY}"
end

echo ""
echo "================================================================"
echo "Generation done.  Running perplexity faithfulness scoring..."
echo "================================================================"

conda run -n llmopt python clear_faith_tasks.py \
    decay_screen_faith.json passage_retrieval_en

conda run -n llmopt python faithfulness_deep.py \
    --model      $MODEL \
    --checkpoint ${OUTPUT}/checkpoint.json \
    --output     decay_screen_faith.json \
    --tasks      $TASK \
    --perplexity \
    --no_embedding

echo ""
echo "================================================================"
echo "Done.  Results in decay_screen_faith.json"
echo "================================================================"

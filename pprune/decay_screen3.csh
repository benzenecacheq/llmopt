#!/bin/tcsh
# decay_screen3.csh
# Fine-grain the curve with min_decay=0.55, 0.65, 0.75.

setenv PYTHONUNBUFFERED 1

set MODEL  = "meta-llama/Llama-3.1-8B"
set OUTPUT = "lb_results_base"
set TASKS  = "passage_retrieval_en,qmsum,samsum"
set N      = 20

echo "================================================================"
echo "decay_screen3: kq_post_rope + decay, md=0.55, 0.65, 0.75"
echo "Tasks : $TASKS"
echo "N     : $N examples per task"
echo "================================================================"

foreach DECAY (0.75 0.65 0.55)
    set LABEL = "kqpr_md${DECAY}"
    echo ""
    echo "--- min_decay=${DECAY}  label=${LABEL} ---"
    conda run -n llmopt python longbench_eval.py \
        --model   $MODEL \
        --tasks   $TASKS \
        --max_examples $N \
        --output  $OUTPUT \
        --budget_fraction 0.65 \
        --score_mode kq_post_rope \
        --min_decay $DECAY \
        --method_label $LABEL \
        --naive_fraction 0.0
    echo "Finished min_decay=${DECAY}"
end

echo ""
echo "================================================================"
echo "Generation done.  Running perplexity faithfulness scoring..."
echo "================================================================"

conda run -n llmopt python clear_faith_tasks.py \
    decay_screen_faith.json passage_retrieval_en qmsum samsum

conda run -n llmopt python faithfulness_deep.py \
    --model      $MODEL \
    --checkpoint ${OUTPUT}/checkpoint.json \
    --output     decay_screen_faith.json \
    --tasks      $TASKS \
    --perplexity \
    --no_embedding

echo ""
echo "================================================================"
echo "Done.  Results in decay_screen_faith.json"
echo "================================================================"

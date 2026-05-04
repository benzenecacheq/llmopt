#!/bin/tcsh
# decay_screen2.csh
# Add min_decay=0.6 and min_decay=0.8 to the faithfulness screen to fill
# in the curve between the existing 0.5/0.7 and 0.7/0.9 measurements.
# Appends to the existing checkpoint and decay_screen_faith.json.

setenv PYTHONUNBUFFERED 1

set MODEL  = "meta-llama/Llama-3.1-8B"
set OUTPUT = "lb_results_base"
set TASKS  = "passage_retrieval_en,qmsum,samsum"
set N      = 20

echo "================================================================"
echo "decay_screen2: kq_post_rope + distance decay, md=0.6 and md=0.8"
echo "Tasks : $TASKS"
echo "N     : $N examples per task"
echo "================================================================"

foreach DECAY (0.8 0.6)
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

# The scorer resumes from decay_screen_faith.json; passage_retrieval_en,
# qmsum, samsum are already scored but will be re-run because the new
# kqpr_md0.6 / kqpr_md0.8 methods now exist in the checkpoint.
# Remove existing entries for the three tasks so scorer re-processes them.
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

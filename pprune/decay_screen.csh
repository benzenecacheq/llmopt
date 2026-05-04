#!/bin/tcsh
# decay_screen.csh
# Quick faithfulness screen for kq_post_rope + distance decay.
#
# Generates outputs for 3 min_decay values on 3 tasks (20 examples each),
# then scores with perplexity faithfulness scoped to the same 3 tasks.
# Results are appended to lb_results_base/checkpoint.json under distinct
# method labels so they can be compared against the existing no-decay
# kq_post_rope baseline already in the checkpoint.
#
# Decision rule: if any decay value beats no-decay (kq_post_rope) by >2
# points on average perplexity faithfulness, run the full 16-task eval.
#
# Runtime: ~20min generation + ~10min perplexity scoring on V100 32GB.

set MODEL  = "meta-llama/Llama-3.1-8B"
set OUTPUT = "lb_results_base"
set TASKS  = "passage_retrieval_en,qmsum,samsum"
set N      = 20

echo "================================================================"
echo "decay_screen: kq_post_rope + distance decay faithfulness screen"
echo "Tasks : $TASKS"
echo "N     : $N examples per task"
echo "================================================================"

# min_decay=1.0 is the no-decay baseline (current kq_post_rope behaviour).
# Already in the checkpoint as method label "kq_post_rope"; skip re-running.
# Decay variants: 0.9, 0.7, 0.5

foreach DECAY (0.9 0.7 0.5)
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

#!/bin/tcsh
# full_eval_md06.csh
# Full 16-task LongBench eval for kq_post_rope with min_decay=0.6 (the
# decay screen winner).  Appends method label "kqpr_md0.6" to the existing
# checkpoint so it sits alongside the other methods for direct comparison.

setenv PYTHONUNBUFFERED 1

set MODEL  = "meta-llama/Llama-3.1-8B"
set OUTPUT = "lb_results_base"
set DECAY  = "0.6"
set LABEL  = "kqpr_md0.6"

echo "================================================================"
echo "full_eval_md06: kq_post_rope min_decay=0.6, all 16 tasks"
echo "Model  : $MODEL"
echo "Label  : $LABEL"
echo "Output : $OUTPUT"
echo "================================================================"

conda run -n llmopt python longbench_eval.py \
    --model          $MODEL \
    --output         $OUTPUT \
    --max_examples   100 \
    --budget_fraction 0.65 \
    --score_mode     kq_post_rope \
    --min_decay      $DECAY \
    --method_label   $LABEL \
    --naive_fraction 0.0

echo ""
echo "================================================================"
echo "Done.  Results in ${OUTPUT}/checkpoint.json"
echo "================================================================"

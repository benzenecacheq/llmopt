#!/bin/tcsh
# run_faithfulness_modes.sh
# Adds streaming, snapkv, vn_decay, kq_only, kq_post_rope predictions to
# lb_results_base so faithfulness can be computed across all methods.
# Each pass skips already-computed keys (full, naive, naive_65pct, pruned).

set TASKS = "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
set MODEL = "meta-llama/Llama-3.1-8B"
set OUTPUT = "lb_results_base"

foreach MODE (streaming snapkv vn_decay kq_only kq_post_rope)
    echo ""
    echo "================================================"
    echo "Starting mode: $MODE"
    echo "================================================"
    python longbench_eval.py \
        --model $MODEL \
        --tasks $TASKS \
        --max_examples 100 \
        --output $OUTPUT \
        --naive_fraction 0.0 \
        --budget_fraction 0.65 \
        --score_mode $MODE \
        --method_label $MODE
    echo "Finished mode: $MODE"
end

echo ""
echo "All modes complete. Running faithfulness scoring..."
python longbench_eval.py \
    --model $MODEL \
    --output $OUTPUT \
    --score_only --faithfulness

echo "Done."

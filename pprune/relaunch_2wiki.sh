#!/bin/bash
# Wait for narrativeqa to fully complete (all 3 methods), then relaunch on 2wikimqa
LOG=/home/benzene/llmopt/pprune/kl_naive_variants.log

until grep -c ">>> narrativeqa" "$LOG" 2>/dev/null | grep -q "^3$"; do
    sleep 10
done

echo "narrativeqa done — killing current eval"
kill $(pgrep -f "kl_naive_variants") 2>/dev/null
sleep 3

cd /home/benzene/llmopt/pprune
nohup /home/benzene/miniconda3/envs/llmopt/bin/python kl_faith_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --data_dir lb_data_raw/data \
    --output lb_results_base/kl_naive_variants.json \
    --methods naive_tail,naive_stream,naive_best_pre,naive_best_post \
    --tasks 2wikimqa \
    --n 100 \
    >> kl_naive_variants.log 2>&1 &

echo "Relaunched on 2wikimqa — PID: $!"

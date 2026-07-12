#!/bin/bash
source /home/benzene/miniconda3/etc/profile.d/conda.sh
conda activate llmopt
cd /home/benzene/llmopt/pprune
echo "=== gap_structure evicted presentation started at $(date) ==="
python gap_structure_eval.py \
    --model meta-llama/Llama-3.1-8B \
    --tasks 2wikimqa,multifieldqa_en,qasper,qmsum,repobench-p,triviaqa \
    --output lb_results_base/gap_structure.json \
    --ystar_cache lb_results_base/ystar_cache_v3.pt \
    --fraction 0.65 \
    --n 20
echo "=== done at $(date) ==="

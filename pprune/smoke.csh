#!/bin/csh
   set cmd = ( python needle_test.py \
      --model meta-llama/Llama-3.2-1B \
      --context_len 512 \
      --budget 512 \
      --num_trials 5 \
      --output sanity.json )
   echo $cmd
   $cmd

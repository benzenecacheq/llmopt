"""
Run LAMBADA benchmark on run_2m_seed42.pt at every 2500-step milestone >= 15000.
Appends results to lambada_2m_progress.log.
"""

import subprocess
import time
import os
import re

LOG = 'run_2m_seed42.log'
OUT = 'lambada_2m_progress.log'
CHECKPOINT = 'run_2m_seed42.pt'
START = 15000
INTERVAL = 2500

def latest_eval_step(log_path):
    """Return the highest step that has a val_ppl in the log."""
    best = 0
    try:
        with open(log_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5 and parts[4].strip():
                    try:
                        best = max(best, int(parts[0]))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return best

def run_benchmark():
    result = subprocess.run(
        ['conda', 'run', '-n', 'blt', 'python', 'blt_lm_eval.py',
         '--checkpoint', CHECKPOINT,
         '--num-m-groups', '2',
         '--tasks', 'lambada_openai',
         '--output', 'lm_eval_2m_blt.json',
         '--batch-size', '4'],
        capture_output=True, text=True
    )
    for line in (result.stdout + result.stderr).splitlines():
        if any(k in line for k in ('Loaded BLT', 'acc,none', 'perplexity,none')):
            yield line.strip()

done = set()

with open(OUT, 'a', buffering=1) as out:
    out.write(f'\n=== monitor started, checking every 60s from step {START} every {INTERVAL} ===\n')

    while True:
        step = latest_eval_step(LOG)
        milestone = (step // INTERVAL) * INTERVAL

        if milestone >= START and milestone not in done:
            out.write(f'\n--- step ~{milestone} (log at {step}) ---\n')
            for line in run_benchmark():
                out.write(line + '\n')
            done.add(milestone)
            out.write(f'done milestones: {sorted(done)}\n')

        # Stop when training finishes
        if os.path.exists(LOG):
            with open(LOG) as f:
                if 'Final val perplexity' in f.read():
                    out.write('\nTraining complete — monitor exiting.\n')
                    break

        time.sleep(60)

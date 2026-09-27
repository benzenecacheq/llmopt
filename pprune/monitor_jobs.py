#!/usr/bin/env python3
"""Monitor GT eval job progress.

Usage:
  python3 monitor_jobs.py          # show current job status
  python3 monitor_jobs.py --watch  # refresh every 30s
"""

import json, re, sys, time
from pathlib import Path
from collections import Counter

BASE = Path('/home/benzene/llmopt/pprune')
QUEUE = BASE / 'lb_results_base/queue.txt'

def parse_queue():
    lines = []
    if QUEUE.exists():
        for line in QUEUE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
    return lines

def detect_running_process():
    """Look for a running kl_faith_eval_ystar or gt_eval_compression process."""
    import subprocess
    try:
        out = subprocess.check_output(['ps', 'aux'], text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if ('kl_faith_eval_ystar.py' in line or 'gt_eval_compression.py' in line) and 'python3' in line and 'grep' not in line:
            # Reconstruct the command from ps output (columns: USER PID... COMMAND)
            parts = line.split(None, 10)
            if len(parts) >= 11:
                return parts[10]
            elif len(parts) >= 10:
                return parts[9]
    return None

def parse_job(cmd):
    """Extract relevant flags from a gt_eval_compression.py command."""
    info = {}
    m = re.search(r'--model\s+(\S+)', cmd)
    if m: info['model'] = m.group(1).split('/')[-1]
    m = re.search(r'--methods\s+(\S+)', cmd)
    if m: info['methods'] = m.group(1).split(',')
    m = re.search(r'--output\s+(\S+)', cmd)
    if m: info['output'] = BASE / m.group(1)
    m = re.search(r'--tasks\s+(\S+)', cmd)
    if m: info['tasks'] = m.group(1).split(',')
    m = re.search(r'--n\s+(\d+)', cmd)
    if m: info['n'] = int(m.group(1))
    # Detect job type
    info['is_kl'] = 'kl_faith_eval' in cmd
    info['is_gt'] = 'gt_eval_compression' in cmd
    return info

def count_checkpoint(ckpt_path, method, tasks=None, n=100):
    """Count how many entries exist per task for a given method."""
    if not Path(ckpt_path).exists():
        return {}
    with open(ckpt_path) as f:
        d = json.load(f)
    counts = Counter()
    for key in d:
        parts = key.split('|')
        if len(parts) != 3: continue
        task, idx, m = parts
        if m == method:
            if tasks is None or task in tasks:
                counts[task] += 1
    return dict(counts)

def count_kl(output_path, methods=None):
    """Count KL entries per task/method. Keys are task|method|idx."""
    if not Path(output_path).exists():
        return {}
    with open(output_path) as f:
        d = json.load(f)
    counts = Counter()
    for key in d:
        parts = key.split('|')
        if len(parts) != 3: continue
        task, m, idx = parts
        if methods is None or m in methods:
            counts[task] += 1
    # Each task gets n entries per method; normalize by method count
    if methods:
        n_methods = len(methods)
        return {t: v // n_methods for t, v in counts.items()}
    return dict(counts)

TASK_SHORT = {
    'narrativeqa': 'NarrQA', 'qasper': 'Qasper', 'multifieldqa_en': 'MultifieldQA',
    'hotpotqa': 'HotpotQA', '2wikimqa': '2WikiMQA', 'musique': 'MuSiQue',
    'gov_report': 'GovReport', 'qmsum': 'QMSum', 'multi_news': 'MultiNews',
    'trec': 'TREC', 'triviaqa': 'TriviaQA', 'samsum': 'SAMSum',
    'passage_count': 'PassCount', 'passage_retrieval_en': 'PassRetrieval',
    'lcc': 'LCC', 'repobench-p': 'RepoBench-P',
}

def show_status():
    queue = parse_queue()

    # Detect what's actually running (may have been removed from queue already)
    running_cmd = detect_running_process()

    if not running_cmd and not queue:
        print("Queue is empty and no eval process detected — all jobs done.")
        return

    # Show currently running job (from ps, not queue)
    current_cmd = running_cmd or (queue[0] if queue else None)
    if not current_cmd:
        return
    info = parse_job(current_cmd)

    print(f"{'─'*60}")
    print(f"RUNNING: {info.get('model','?')}")

    if info.get('is_gt') and 'output' in info:
        ckpt = info['output'] / 'checkpoint.json'
        for method in info.get('methods', []):
            counts = count_checkpoint(ckpt, method, info.get('tasks'), info.get('n', 100))
            n = info.get('n', 100)

            print(f"\n  Method: {method}")
            tasks = info.get('tasks') or list(counts.keys())

            in_progress = None
            for task in tasks:
                done = counts.get(task, 0)
                if done < n:
                    in_progress = (task, done)
                    break

            if in_progress:
                task, done = in_progress
                short = TASK_SHORT.get(task, task)
                pct = 100 * done / n
                bar = '█' * (done // 10) + '░' * ((n - done) // 10)
                print(f"  Current task: {short} [{bar}] {done}/{n} ({pct:.0f}%)")
            else:
                print(f"  All tasks complete.")

            # Summary of completed tasks
            done_tasks = [t for t in tasks if counts.get(t, 0) >= n]
            todo_tasks = [t for t in tasks if counts.get(t, 0) < n]
            print(f"  Done: {len(done_tasks)}/{len(tasks)} tasks", end='')
            if done_tasks:
                print(f"  ({', '.join(TASK_SHORT.get(t,t) for t in done_tasks)})", end='')
            print()
            if todo_tasks:
                print(f"  Remaining: {', '.join(TASK_SHORT.get(t,t) for t in todo_tasks)}")

    elif info.get('is_kl') and 'output' in info:
        output_path = info['output']
        methods = info.get('methods', [])
        n = info.get('n', 100)
        if not Path(output_path).exists():
            print(f"\n  Output not yet created: {Path(output_path).name}")
            print(f"  Methods: {', '.join(methods)}")
        else:
            counts = count_kl(output_path, methods)
            # KL runs all 16 tasks by default (no --tasks filter)
            all_16 = list(TASK_SHORT.keys())
            tasks = info.get('tasks') or all_16
            in_progress = [(t, counts.get(t, 0)) for t in tasks if counts.get(t, 0) < n]
            if in_progress:
                task, done = in_progress[0]
                short = TASK_SHORT.get(task, task)
                pct = 100 * done / n
                bar = '█' * (done // 10) + '░' * ((n - done) // 10)
                print(f"\n  Methods: {', '.join(methods)}")
                print(f"  Current task: {short} [{bar}] {done}/{n} ({pct:.0f}%)")
            done_tasks = [t for t in tasks if counts.get(t, 0) >= n]
            print(f"  Done: {len(done_tasks)}/{len(tasks)} tasks")

    # Show queue summary
    # If the running job was detected from ps (not in queue), queue has N waiting jobs
    if running_cmd:
        waiting = queue  # running job already removed from queue
    else:
        waiting = queue[1:]  # first queue line is the running one
    print(f"\n{'─'*60}")
    print(f"Waiting in queue: {len(waiting)} job(s)")
    for i, cmd in enumerate(waiting[:5], 1):
        inf = parse_job(cmd)
        method = ','.join(inf.get('methods', ['?']))
        model = inf.get('model', '?')
        ntask = len(inf.get('tasks', []))
        label = f"{ntask} tasks" if ntask else "all tasks"
        print(f"  [{i}] {model}  {method}  ({label})")
    if len(waiting) > 5:
        print(f"  ... {len(waiting)-5} more")

def main():
    watch = '--watch' in sys.argv
    interval = 30
    for arg in sys.argv[1:]:
        if arg.isdigit():
            interval = int(arg)

    if watch:
        print(f"Watching (refresh every {interval}s) — Ctrl+C to stop\n")
        try:
            while True:
                print(f"\033[2J\033[H", end='')  # clear screen
                import datetime
                print(datetime.datetime.now().strftime('%H:%M:%S'))
                show_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_status()

if __name__ == '__main__':
    main()

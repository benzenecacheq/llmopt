"""Monitor a training log and report when val_ppl stops improving materially."""

import argparse
import time

def parse_ppl_lines(path):
    results = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 5 and parts[4].strip():
                try:
                    results.append((int(parts[0]), float(parts[4])))
                except ValueError:
                    pass
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Min ppl improvement per interval to be considered material')
    parser.add_argument('--patience', type=int, default=3,
                        help='Consecutive sub-threshold intervals before declaring plateau')
    parser.add_argument('--poll', type=int, default=60, help='Seconds between checks')
    parser.add_argument('--then', type=str, default=None,
                        help='Shell command to run after plateau is detected')
    args = parser.parse_args()

    seen = set()
    history = []
    plateau_count = 0

    print(f'Monitoring {args.log} (threshold={args.threshold}, patience={args.patience})')
    print(f'{"step":>8}  {"val_ppl":>8}  {"delta":>8}  {"status"}')

    while True:
        readings = parse_ppl_lines(args.log)
        for step, ppl in readings:
            if step in seen:
                continue
            seen.add(step)
            history.append((step, ppl))

            if len(history) < 2:
                print(f'{step:>8}  {ppl:>8.2f}  {"—":>8}')
                continue

            delta = history[-2][1] - ppl
            if delta < args.threshold:
                plateau_count += 1
                status = f'sub-threshold ({plateau_count}/{args.patience})'
            else:
                plateau_count = 0
                status = 'improving'

            print(f'{step:>8}  {ppl:>8.2f}  {delta:>8.3f}  {status}', flush=True)

            if plateau_count >= args.patience:
                print(f'\nPLATEAU: improvement < {args.threshold} for {args.patience} '
                      f'consecutive intervals. Floor ~{ppl:.2f} at step {step}.')
                if args.then:
                    import subprocess
                    print(f'Running: {args.then}')
                    subprocess.run(args.then, shell=True)
                raise SystemExit(0)

        time.sleep(args.poll)

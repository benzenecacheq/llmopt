"""Remove named tasks from a faithfulness JSON so the scorer re-processes them."""
import json, os, sys

faith_path = sys.argv[1]
tasks = sys.argv[2:]

with open(faith_path) as f:
    d = json.load(f)
for t in tasks:
    d['perplexity'].pop(t, None)
    d.get('embedding', {}).pop(t, None)
tmp = faith_path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(d, f, indent=2)
os.replace(tmp, faith_path)
print(f'Cleared {tasks} from {faith_path}')

#!/usr/bin/env python3
import json, ast, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('lb_results_base/phrase_bpe_merges.json') as f:
    merges = json.load(f)

# Build merge-tree depth: how many original tokens compose each phrase token
depth = {}   # new_id -> constituent token count
for m in merges:
    a, b, new_id = m['a'], m['b'], m['new_id']
    depth[new_id] = depth.get(a, 1) + depth.get(b, 1)

word_counts, depths, freqs = [], [], []
for m in merges:
    try:
        text = ast.literal_eval(m['pair'])
        wc = len(re.findall(r'\w+', text))
    except Exception:
        wc = 0
    word_counts.append(wc)
    depths.append(depth[m['new_id']])
    freqs.append(m['freq'])

word_counts = np.array(word_counts)
depths      = np.array(depths)
freqs       = np.array(freqs, dtype=float)
total_savings = freqs.sum()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: scatter, log-y, coloured by word count ---
ax = axes[0]
cmap = plt.cm.get_cmap('tab10')
for wc in sorted(set(word_counts)):
    mask = word_counts == wc
    label = f'{wc} word{"s" if wc != 1 else ""}'
    ax.scatter(depths[mask], freqs[mask], s=5, alpha=0.35,
               color=cmap(wc % 10), label=label, zorder=2)
ax.set_yscale('log')
ax.set_xlabel('Constituent original tokens (merge depth)')
ax.set_ylabel('Corpus frequency (log scale)')
ax.set_title('Each phrase token: depth vs frequency')
ax.legend(fontsize=8, markerscale=2)
ax.grid(True, alpha=0.3)

# annotate top-8 most frequent
for i in np.argsort(freqs)[::-1][:8]:
    try:
        raw = ast.literal_eval(merges[i]['pair'])[:14]
        label = raw.replace('\n', r'\n').replace('\t', r'\t').replace('\r', r'\r')
    except Exception:
        label = '?'
    ax.annotate(repr(label)[1:-1], (depths[i], freqs[i]),
                fontsize=6.5, xytext=(4, 2), textcoords='offset points', color='darkred')

# --- Right: % of total savings by word count ---
ax2 = axes[1]
max_wc = int(word_counts.max())
bins = list(range(0, max_wc + 2))
pct_weights = freqs / total_savings * 100
ax2.hist(word_counts, bins=bins, weights=pct_weights,
         color='steelblue', alpha=0.75, edgecolor='white', linewidth=0.5,
         align='left')
ax2.set_xlabel('Words in phrase (regex \\w+ count)')
ax2.set_ylabel('% of total token savings')
ax2.set_title('Token savings by word count')
ax2.set_xticks(range(0, max_wc + 1))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = 'lb_results_base/phrase_bpe_length_vs_freq.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')

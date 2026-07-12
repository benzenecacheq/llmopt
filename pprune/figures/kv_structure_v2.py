#!/usr/bin/env python3
"""
figures/kv_structure_v2.py  —  Structural corruption in KV pruning
Combines: real text tokens + attention bars (ideas 1+2) + RoPE distance arc (idea 4)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── example: 11-token sentence, 7 retained ────────────────────────────────────
TOKENS = ["The", "report", "found", "mercury", "levels", "in",
          "fish", "exceeded", "safe", "limits", "."]
KEPT   = [True,  True,     True,   False,     True,    False,
          True,  False,    True,   False,     True]
N, M   = len(TOKENS), sum(KEPT)   # 11, 7

KEY_RET_IDX  = 4    # "fish" is 5th retained token (0-indexed among retained)
KEY_ORIG_IDX = 6    # "fish" is 7th token overall  (0-indexed → position 7)

T1_ORIG = N + 1     # position 12
T1_NEW  = M + 1     # position 8

ROPE_E = T1_ORIG - (KEY_ORIG_IDX + 1)   # 12 - 7 = 5
ROPE_P = T1_NEW  - (KEY_RET_IDX + 1)    # 8  - 5 = 3

# illustrative attention weights (sum to 1.0 each)
# retained order: The report found levels fish safe .
W_E = [0.04, 0.06, 0.08, 0.13, 0.18, 0.14, 0.37]   # post-hoc eviction
W_P = [0.03, 0.04, 0.07, 0.11, 0.28, 0.15, 0.32]   # prompt construction

# ── geometry ──────────────────────────────────────────────────────────────────
BW, BH = 0.72, 0.46    # box width / height
STEP   = 1.05           # x spacing between token centres
T1G    = 0.60           # extra gap before T1 box
# Both panels use the same x range so boxes are the same physical size
T1X_E  = N * STEP + T1G            # T1 x in eviction panel   (~12.15)
T1X_P  = M * STEP + T1G            # T1 x in prompt panel     (~7.95)
XMAX   = T1X_E + 0.75              # common right edge
XMIN   = -0.75

# ── colours ───────────────────────────────────────────────────────────────────
CK  = '#2B7BB9'   # retained box
CE  = '#D8D8D8'   # evicted box
CT1 = '#D95F02'   # T1
CB  = '#74BAE8'   # attention bar
CA_E = '#CC2020'  # arc eviction panel  (red   = large distance, bad)
CA_P = '#2A9D2A'  # arc prompt panel    (green = small distance, good)

# ── helpers ───────────────────────────────────────────────────────────────────
def draw_box(ax, x, text, fc, ec='#555', lw=1.4, ls='-',
             tc='white', fw='bold', fs=9.5):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - BW/2, -BH/2), BW, BH, boxstyle='round,pad=0.05',
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))
    ax.text(x, 0, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=fw, zorder=5)


def draw_bar(ax, x, w, scale=2.6):
    h = w * scale
    ax.add_patch(mpatches.Rectangle(
        (x - 0.19, BH/2 + 0.05), 0.38, h,
        facecolor=CB, edgecolor='#3A80BB', lw=0.75, alpha=0.90, zorder=3))
    if w >= 0.07:
        ax.text(x, BH/2 + 0.05 + h + 0.04, f'{w:.2f}',
                ha='center', va='bottom', fontsize=7, color='#1A5588', zorder=5)


def draw_arc(ax, x1, x2, color, label):
    """Downward half-ellipse from x1 to x2, with tick endpoints + distance label."""
    y0   = -BH/2 - 0.44
    rdep = (x2 - x1) * 0.12     # depth proportional to span → visual encoding of distance

    # Drop dotted lines
    for xv in (x1, x2):
        ax.plot([xv, xv], [-BH/2, y0], color=color,
                lw=0.9, ls=':', alpha=0.55, zorder=2)

    # Half-ellipse (bows downward)
    t     = np.linspace(0, np.pi, 400)
    arc_x = x1 + (x2 - x1) * (1 - np.cos(t)) / 2
    arc_y = y0 - rdep * np.sin(t)
    ax.plot(arc_x, arc_y, color=color, lw=2.0, solid_capstyle='round', zorder=4)

    # Endpoint tick marks
    for xv in (x1, x2):
        ax.plot([xv, xv], [y0 - 0.08, y0 + 0.08], color=color, lw=2.0, zorder=5)

    # Distance label
    ax.text((x1 + x2) / 2, y0 - rdep - 0.15, label,
            ha='center', va='top', fontsize=11, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor=color, lw=1.6, alpha=0.94), zorder=6)


# ── panels ────────────────────────────────────────────────────────────────────
def panel_eviction(ax):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-2.20, 2.50)
    ax.axis('off')
    ax.set_title('Post-hoc eviction  (e.g. SnapKV)',
                 fontsize=12, fontweight='bold', loc='left', x=0.0, pad=7)

    ri, key_x = 0, None
    for i, (tok, kept) in enumerate(zip(TOKENS, KEPT)):
        x = i * STEP
        if kept:
            is_key = (ri == KEY_RET_IDX)
            ec = '#FF8C00' if is_key else '#1A5E8A'
            lw = 2.8 if is_key else 1.4
            draw_box(ax, x, tok, CK, ec, lw=lw)
            draw_bar(ax, x, W_E[ri])
            ax.text(x, -BH/2 - 0.14, str(i + 1), ha='center', va='top',
                    fontsize=7.5, color='#444')
            if is_key:
                key_x = x
            ri += 1
        else:
            draw_box(ax, x, tok, CE, '#999', lw=1.1, ls='--',
                     tc='#888', fw='normal', fs=9.0)
            ax.text(x, -BH/2 - 0.14, str(i + 1), ha='center', va='top',
                    fontsize=7.5, color='#BBB')

    # T1
    draw_box(ax, T1X_E, 'T₁', CT1, '#8B3A00')
    ax.text(T1X_E, -BH/2 - 0.14, str(T1_ORIG), ha='center', va='top',
            fontsize=7.5, color='#444')

    # Span annotation — placed above all bars (tallest bar tops out ~1.25)
    AY = 1.65
    ax.annotate('', xy=(T1X_E, AY), xytext=(0, AY),
                arrowprops=dict(arrowstyle='<->', color='#999', lw=1.2))
    ax.text(T1X_E / 2, AY + 0.08, f'N = {N} original positions, {N - M} evicted',
            ha='center', va='bottom', fontsize=8, color='#888', style='italic')

    # "attn weight" label
    ax.text(XMIN + 0.05, 1.80, '↑ attention\nweight', ha='left', va='top',
            fontsize=7.5, color='#555', style='italic')

    draw_arc(ax, key_x, T1X_E, CA_E, f'RoPE distance  d = {ROPE_E}')


def panel_prompt(ax):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-2.20, 2.50)
    ax.axis('off')
    ax.set_title('Prompt construction  (phrase-based)',
                 fontsize=12, fontweight='bold', loc='left', x=0.0, pad=7)

    ri, key_x = 0, None
    for i, (tok, kept) in enumerate(zip(TOKENS, KEPT)):
        if not kept:
            continue
        x = ri * STEP
        is_key = (ri == KEY_RET_IDX)
        ec = '#FF8C00' if is_key else '#1A5E8A'
        lw = 2.8 if is_key else 1.4
        draw_box(ax, x, tok, CK, ec, lw=lw)
        draw_bar(ax, x, W_P[ri])
        ax.text(x, -BH/2 - 0.14, str(ri + 1), ha='center', va='top',
                fontsize=7.5, color='#444')
        ax.text(x, -BH/2 - 0.31, f'[{i + 1}]', ha='center', va='top',
                fontsize=6.5, color='#AAA')
        if ri == KEY_RET_IDX:
            key_x = x
        ri += 1

    # T1
    draw_box(ax, T1X_P, 'T₁', CT1, '#8B3A00')
    ax.text(T1X_P, -BH/2 - 0.14, str(T1_NEW), ha='center', va='top',
            fontsize=7.5, color='#444')
    ax.text(T1X_P, -BH/2 - 0.31, f'[{T1_ORIG}]', ha='center', va='top',
            fontsize=6.5, color='#AAA')

    # Span annotation — same height as eviction panel
    AY = 1.65
    ax.annotate('', xy=(T1X_P, AY), xytext=(0, AY),
                arrowprops=dict(arrowstyle='<->', color='#999', lw=1.2))
    ax.text(T1X_P / 2, AY + 0.08, f'M = {M} contiguous positions, 0 evicted',
            ha='center', va='bottom', fontsize=8, color='#888', style='italic')

    ax.text(XMIN + 0.05, 1.80, '↑ attention\nweight', ha='left', va='top',
            fontsize=7.5, color='#555', style='italic')

    # Bracket-notation legend
    ax.text(T1X_P + 0.55, -BH/2 - 0.28, '[ ] = original\nposition',
            ha='left', va='top', fontsize=6.5, color='#AAA')

    draw_arc(ax, key_x, T1X_P, CA_P, f'RoPE distance  d = {ROPE_P}')


# ── assemble ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.14, left=0.01, right=0.99, top=0.97, bottom=0.04)

panel_eviction(ax1)
panel_prompt(ax2)

fig.text(
    0.5, 0.006,
    'Bar height = illustrative attention weight from T₁ to each retained token.  '
    f'"fish" (key semantic token) sits at RoPE distance d = {ROPE_E} in post-hoc eviction '
    f'vs d = {ROPE_P} after prompt construction — same content, different positional encoding.',
    ha='center', va='bottom', fontsize=8.5, color='#555')

outpath = 'figures/kv_structure_v2.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {outpath}')

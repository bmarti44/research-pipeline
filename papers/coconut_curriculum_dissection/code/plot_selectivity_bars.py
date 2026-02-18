#!/usr/bin/env python3
"""Generate Figure 3: Step selectivity by thought position for M2 and M3.

Uses selectivity values at each model's peak probe accuracy layer:
  M2 (COCONUT): layer 0 for positions 0, 1, 3; layer 12 for position 2
  M3 (Pause): layer 12 for all positions

Data source: results/selectivity_recomputed.json

Note: Data files use Lambda-era keys (m3=COCONUT, m5=Pause).
Paper numbering: M2=COCONUT, M3=Pause.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Selectivity values at peak layer (percentage points)
# From selectivity_raw_grid in selectivity_recomputed.json
positions = [0, 1, 2, 3, 4]
pos_labels = ['Pos 0', 'Pos 1', 'Pos 2', 'Pos 3', 'Pos 4']

# M2 (COCONUT): layer 0 for pos 0,1,3; layer 12 for pos 2
m2_selectivity = [-15.6, -10.6, 9.4, 52.0, 0.0]

# M3 (Pause): layer 12 for all positions
m3_selectivity = [-12.0, -14.6, 10.2, 52.3, 0.0]

# Colors
m2_color = '#E8778B'    # pink/coral for M2 (COCONUT)
m3_color = '#5BA4B5'    # teal for M3 (Pause)
m2_text = '#C4475A'     # darker pink for annotations
m3_text = '#2E7D8A'     # darker teal for annotations

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(positions))
width = 0.35

bars_m2 = ax.bar(x - width/2, m2_selectivity, width, label='M2 (COCONUT)',
                  color=m2_color, edgecolor='white', linewidth=0.5)
bars_m3 = ax.bar(x + width/2, m3_selectivity, width, label='M3 (Pause)',
                  color=m3_color, edgecolor='white', linewidth=0.5)

# Zero line
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')


def annotate_bar(ax, x_pos, value, color, offset=3, fontsize=9, fontweight='normal'):
    """Annotate a bar with its value and a small connector line."""
    fmt = f'{value:+.1f}' if value != 0.0 else '0.0'
    if value >= 0:
        text_y = value + offset
    else:
        text_y = value - offset

    va = 'bottom' if value >= 0 else 'top'

    # Connector line from bar top/bottom to label
    line_start = value + (1 if value >= 0 else -1)
    line_end = text_y - (1 if value >= 0 else -1)

    if value != 0.0:
        ax.plot([x_pos, x_pos], [line_start, line_end],
                color=color, linewidth=0.8, solid_capstyle='round')

    ax.text(x_pos, text_y, fmt, ha='center', va=va,
            fontsize=fontsize, fontweight=fontweight, color=color)


# Annotate all bars (except position 4 which is 0.0 for both)
for i in range(4):  # positions 0-3
    is_hero = (i == 3)
    fs = 9.5 if is_hero else 9
    fw = 'normal'

    # For position 3, same height side by side (smaller font to fit)
    if is_hero:
        annotate_bar(ax, x[i] - width/2, m2_selectivity[i], m2_text,
                     offset=4, fontsize=9.5, fontweight=fw)
        annotate_bar(ax, x[i] + width/2, m3_selectivity[i], m3_text,
                     offset=4, fontsize=9.5, fontweight=fw)
    else:
        offset = 3.5 if abs(m2_selectivity[i]) > 5 else 2.5
        annotate_bar(ax, x[i] - width/2, m2_selectivity[i], m2_text,
                     offset=offset, fontsize=fs, fontweight=fw)
        annotate_bar(ax, x[i] + width/2, m3_selectivity[i], m3_text,
                     offset=offset, fontsize=fs, fontweight=fw)

# Labels and formatting
ax.set_xlabel('Thought Position', fontsize=12)
ax.set_ylabel('Selectivity (percentage points)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(pos_labels, fontsize=11)
ax.tick_params(axis='y', labelsize=10)
ax.legend(fontsize=11, loc='upper left')

# Y-axis range to accommodate annotations
ax.set_ylim(-28, 70)

# Light grid on y-axis
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save
output_dir = os.path.join(os.path.dirname(__file__),
                           '..', 'manuscript', 'figures')
output_path = os.path.join(output_dir, 'fig3_selectivity_bars.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")
plt.close()

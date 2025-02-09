import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.lines as mlines
import os

# Probability mass function
# P(Falling for n blocks) = (1-p)^n
p_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
n_values = np.arange(1, 11)

# Load Custom Fonts
try:
    aeonik_font_path = '/Users/hanson_zhang/research_projects/python-sim-research/scripts/fonts/aeonik.ttf'
    at_aero_font_path = '/Users/hanson_zhang/research_projects/python-sim-research/scripts/fonts/ataero.ttf'

    aeonik_font = fm.FontProperties(fname=aeonik_font_path)
    at_aero_font = fm.FontProperties(fname=at_aero_font_path)
except:
    aeonik_font = fm.FontProperties(family='Arial')
    at_aero_font = fm.FontProperties(family='Arial')

# Rebrand Styling
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0c111d')
fig.patch.set_facecolor('#0c111d')
ax.set_facecolor('#0c111d')

# Grid Styling
ax.grid(axis='y', linestyle='-', dashes=(3, 7), zorder=0, color='#3c404a')

# Plot probability lines for each p
colors = plt.cm.viridis(np.linspace(0, 1, len(p_values)))
for i, p in enumerate(p_values):
    probabilities = (1 - p) ** n_values
    ax.plot(n_values, probabilities, color=colors[i], linewidth=2, label=f'p={p:.2f}', zorder=2)

# X and Y axis formatting
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Set Labels and Title
ax.set_xlabel('Number of Blocks (n)', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Probability', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
fig.suptitle('Probability of Falling for n Blocks', color='#fbfbfc', x=0.08, y=0.95, fontproperties=at_aero_font, fontsize=30, fontweight='bold', ha='left')

# Tick Customization
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, pad=15)
plt.setp(ax.get_xticklabels(), fontproperties=aeonik_font, fontsize=10, color='#98a2b3')
plt.setp(ax.get_yticklabels(), fontproperties=aeonik_font, fontsize=10, color='#98a2b3')

# Add Custom Legend
lines = [mlines.Line2D([], [], color=colors[i], label=f'p={p:.2f}') for i, p in enumerate(p_values)]
fig.legend(handles=lines, loc='upper right', bbox_to_anchor=(0.95, 0.95),
           facecolor='#0c111d', edgecolor='#0c111d', labelcolor='#98a2b3',
           prop=aeonik_font, ncol=1)

# Add Watermark
logo_path = '/Users/hanson_zhang/research_projects/python-sim-research/scripts/fonts/logo_small.png'
if os.path.exists(logo_path):
    logo = plt.imread(logo_path)
    fig.figimage(logo, xo=(fig.bbox.xmax / 2) - (logo.shape[1] / 2),
                 yo=(fig.bbox.ymax / 2) - (logo.shape[0] / 2) - 30, alpha=0.3, zorder=3)
else:
    print("Logo file not found. Skipping watermark...")

# Final Layout and Display
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.85])
plt.show()
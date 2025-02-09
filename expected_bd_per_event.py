import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib import font_manager as fm
import os

#########################
##### Data Processing #####
#########################

# Manually input data
data = {
    "Asset": ["ETH", "BTC", "LINK", "CRV"],
    "Date": ["Feb 2, 2025", "Mar 19, 2024", "Dec 09, 2024", "June 13, 2024"],
    "Number": [127, 6, 10, 5],
    "Baseline": [0, 0, 0, 0],
    "Bad Debt Normal": [7287.08, 92.91, 25821.11, 958.06],
    "Bad Debt Extreme": [10846.34, 139.65, 38797.87, 1471.94]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Compute Expected Bad Debt Per Liquidation Event
df["Expected BD Normal"] = df["Bad Debt Normal"] / df["Number"]
df["Expected BD Extreme"] = df["Bad Debt Extreme"] / df["Number"]

#########################
##### Visualization #####
#########################

# === Load Custom Fonts ===
try:
    aeonik_font_path = '/Users/hanson_zhang/research_projects/python-sim-research/scripts/fonts/aeonik.ttf'
    at_aero_font_path = '/Users/hanson_zhang/research_projects/python-sim-research/scripts/fonts/ataero.ttf'

    aeonik_font = fm.FontProperties(fname=aeonik_font_path)
    at_aero_font = fm.FontProperties(fname=at_aero_font_path)
except:
    aeonik_font = fm.FontProperties(family='Arial')
    at_aero_font = fm.FontProperties(family='Arial')

# === Prepare Data for Histogram ===
x_labels = df["Asset"]
bar_width = 0.4  # Width of bars
x = range(len(x_labels))

# === Rebrand Styling ===
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0c111d')
fig.patch.set_facecolor('#0c111d')
ax.set_facecolor('#0c111d')

# === Grid Styling ===
ax.grid(axis='y', linestyle='-', dashes=(3, 7), zorder=0, color='#3c404a')

# === Plot the Data ===
ax.bar([i - bar_width/2 for i in x], df["Expected BD Normal"], bar_width, color='#C6FF8A', label='Expected BD (Normal)', zorder=2)
ax.bar([i + bar_width/2 for i in x], df["Expected BD Extreme"], bar_width, color='#FF6B6B', label='Expected BD (Extreme)', zorder=2)

# === X and Y Axis Formatting ===
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontproperties=aeonik_font, fontsize=12, color='#98a2b3')
ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# === Set Labels and Title ===
ax.set_xlabel('Asset', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Expected Bad Debt per Liquidation Event (USD)', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
fig.suptitle('Expected Bad Debt per Liquidation Event by Asset', color='#fbfbfc', x=0.08, y=0.95,
             fontproperties=at_aero_font, fontsize=30, fontweight='bold', ha='left')

# === Tick Customization ===
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, pad=15)
plt.setp(ax.get_yticklabels(), fontproperties=aeonik_font, fontsize=10, color='#98a2b3')

# === Add Custom Legend ===
lines = [
    mlines.Line2D([], [], color='#C6FF8A', marker='o', label='Expected BD (Normal)'),
    mlines.Line2D([], [], color='#FF6B6B', marker='s', label='Expected BD (Extreme)'),
]
fig.legend(handles=lines, loc='upper left', bbox_to_anchor=(0.1, 0.85),
           facecolor='#0c111d', edgecolor='#0c111d', labelcolor='#98a2b3',
           prop=aeonik_font, ncol=1)

# === Add Watermark ===
logo_path = '/Users/hanson_zhang/research_projects/python-sim-research/scripts/fonts/logo_small.png'
if os.path.exists(logo_path):
    logo = plt.imread(logo_path)
    fig.figimage(logo, xo=(fig.bbox.xmax / 2) - (logo.shape[1] / 2),
                 yo=(fig.bbox.ymax / 2) - (logo.shape[0] / 2) - 30, alpha=0.3, zorder=3)
else:
    print("Logo file not found. Skipping watermark...")

# === Final Layout and Display ===
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.85])
plt.show()

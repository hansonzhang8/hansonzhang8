import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib import font_manager as fm
import os

#########################
##### Data Processing #####
#########################

# Load dataset
file_path = "/Users/hanson_zhang/PycharmProjects/svr_research/data/eth_mev.csv"
df = pd.read_csv(file_path)

# Compute total MEV value
mev_value_total = df["mev_value_usd"].sum()

# Expected Bad Debt values
expected_bd_normal = 7287.08
expected_bd_extreme = 10846.34

# Data for visualization
data = {
    "Category": ["OEV Total", "Expected BD Normal", "Expected BD Extreme"],
    "Value": [mev_value_total, expected_bd_normal, expected_bd_extreme]
}
df_viz = pd.DataFrame(data)

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

# === Rebrand Styling ===
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0c111d')
fig.patch.set_facecolor('#0c111d')
ax.set_facecolor('#0c111d')

# === Grid Styling ===
ax.grid(axis='y', linestyle='-', dashes=(3, 7), zorder=0, color='#3c404a')

# === Plot the Data ===
colors = ['#4A90E2', '#C6FF8A', '#FF6B6B']
bar_width = 0.3  # Adjusted bar width to make columns slimmer
ax.bar(df_viz["Category"], df_viz["Value"], color=colors, width=bar_width, zorder=2)

# === X and Y Axis Formatting ===
ax.set_xticklabels(df_viz["Category"], fontproperties=aeonik_font, fontsize=12, color='#98a2b3')
ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# === Set Labels and Title ===
ax.set_xlabel('Category', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('USD Value', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
fig.suptitle('Max OEV Total vs Expected Bad Debt', color='#fbfbfc', x=0.08, y=0.95,
             fontproperties=at_aero_font, fontsize=30, fontweight='bold', ha='left')

# === Tick Customization ===
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, pad=15)
plt.setp(ax.get_yticklabels(), fontproperties=aeonik_font, fontsize=10, color='#98a2b3')

# === Add Custom Legend ===
lines = [
    mlines.Line2D([], [], color='#4A90E2', marker='o', label='OEV Total'),
    mlines.Line2D([], [], color='#C6FF8A', marker='o', label='Expected BD (Normal)'),
    mlines.Line2D([], [], color='#FF6B6B', marker='s', label='Expected BD (Extreme)')
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

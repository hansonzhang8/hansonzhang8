import pandas as pd
import numpy as np

#########################
##### Configuration #####
#########################

LIQUIDATION_BONUS = 0.05

def sqrtPriceX96_to_price(sqrt_price_x96: float) -> float:
    return (sqrt_price_x96 / 2**96) ** 2 * 1e12

def compute_bad_debt_baseline(
        debt_amount: float,
        debtAsset_price: float,  # <-- Updated variable name
        liquidated_collateral_amount: float,
        eth_price: float
) -> float:
    """
    Compute the baseline bad debt for a single liquidation event, given:
      - debt_amount (tokens)
      - debtAsset_price (USD/token)
      - liquidated_collateral_amount (ETH)
      - eth_price (USD/ETH)

    The formula used:
      badDebt = max(0, (debt_amount * debtAsset_price)
                         - (liquidated_collateral_amount * eth_price))
    """
    debt_value_usd = debt_amount * debtAsset_price
    collateral_value_usd = liquidated_collateral_amount * eth_price

    return max(0.0, debt_value_usd - collateral_value_usd)

#########################
##### Implementation ####
#########################

# Step 1: Load Data and Merge
def step_1_load_and_merge(liquidations_csv: str, slot0_csv: str) -> pd.DataFrame:
    # (A) Read liquidation CSV
    df_liquidations = pd.read_csv(liquidations_csv)

    # Convert needed columns to numeric
    numeric_cols = ['liquidated_collateral_amount', 'debt_amount', 'debtAsset_price']
    for col in numeric_cols:
        df_liquidations[col] = pd.to_numeric(df_liquidations[col], errors='coerce')

    # (B) Read ETH slot0 data
    df_eth_slot0 = pd.read_csv(slot0_csv)
    df_eth_slot0['sqrtPriceX96'] = pd.to_numeric(df_eth_slot0['sqrtPriceX96'], errors='coerce')
    df_eth_slot0['eth_price'] = df_eth_slot0['sqrtPriceX96'].apply(sqrtPriceX96_to_price)

    # Rename 'blocknumber' to 'block_number'
    df_eth_slot0.rename(columns={'blocknumber': 'block_number'}, inplace=True)

    # Keep only relevant columns
    df_eth_slot0 = df_eth_slot0[['block_number', 'eth_price']]

    # (C) Merge on block_number
    df_merged = pd.merge(
        df_liquidations,
        df_eth_slot0,
        left_on='evt_block_number',
        right_on='block_number',
        how='left'
    )

    return df_merged

# Step 2: Compute Baseline Bad Debt
def step_2_compute_baseline_bad_debt(df_merged: pd.DataFrame) -> pd.DataFrame:

    def _calc_baseline_bd(row):
        return compute_bad_debt_baseline(
            debt_amount=row['debt_amount'],
            debtAsset_price=row['debtAsset_price'],  # <-- Updated variable name
            liquidated_collateral_amount=row['liquidated_collateral_amount'],
            eth_price=row['eth_price']
        )

    df_merged['bad_debt_baseline'] = df_merged.apply(_calc_baseline_bd, axis=1)
    return df_merged

# Call Step 1 and Step 2
if __name__ == "__main__":
    # Example usage:
    liquidations_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/eth_decrease_liquidations.csv"
    slot0_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/ETH_slot0_21755110_21762264.csv"

    df_merged = step_1_load_and_merge(liquidations_file, slot0_file)
    df_with_baseline = step_2_compute_baseline_bad_debt(df_merged)

    # Inspect baseline results
    print(df_with_baseline[['evt_tx_hash', 'evt_block_number',
                            'debt_amount', 'liquidated_collateral_amount',
                            'debtAsset_price', 'eth_price',
                            'bad_debt_baseline']].head(10))

    total_baseline = df_with_baseline['bad_debt_baseline'].sum()
    print(f"\nTotal Baseline Bad Debt: {total_baseline:,.2f} USD")


# Step 3: Oracle Delay & Expected Bad Debt #####


# Progressive ETH price drops due to oracle delay
PROGRESSIVE_DROPS = {
    1: 0.0051,   # 0.51%
    2: 0.0081,   # 0.81% (0.51% + 0.30%)
    3: 0.0132,   # 1.32% (0.51% + 0.30% + 0.51%)
    4: 0.0183,   # 1.83% (0.51% + 0.30% + 0.51% + 0.51%)
    5: 0.0213    # 2.13% (0.51% + 0.30% + 0.51% + 0.51% + 0.30%)
}

def step_3_model_oracle_delay(
    df: pd.DataFrame,
    progressive_drops: dict = PROGRESSIVE_DROPS,
    liquidation_bonus: float = LIQUIDATION_BONUS
) -> pd.DataFrame:

    def _compute_delayed_bd(row, drop_fraction: float):
        """
        Compute bad debt for a single liquidation event given an ETH price drop.
        """
        # 1) Compute new ETH price under delayed scenario
        delayed_eth_price = row['eth_price'] * (1.0 - drop_fraction)

        # 2) Compute new collateral value in USD under delayed price
        collateral_value_new = row['liquidated_collateral_amount'] * delayed_eth_price

        # 3) Compute required debt coverage (including liquidation bonus)
        debt_value = row['debt_amount'] * row['debtAsset_price'] * (1 + liquidation_bonus)

        # 4) Calculate bad debt
        return max(0.0, debt_value - collateral_value_new)

    # For each n in progressive_drops, compute the delayed BD column
    for n, drop_frac in progressive_drops.items():
        col_name = f'bad_debt_delayed_{n}'
        df[col_name] = df.apply(lambda row: _compute_delayed_bd(row, drop_frac), axis=1)

    return df


# Call Step 3:
df_with_baseline = step_2_compute_baseline_bad_debt(df_merged)
df_with_delays = step_3_model_oracle_delay(df_with_baseline)

# Ensure full DataFrame is printed
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Adjust width to prevent wrapping
pd.set_option('display.colheader_justify', 'center')  # Align column headers

# Print a sample of results
print(df_with_delays[['evt_tx_hash', 'bad_debt_baseline',
                      'bad_debt_delayed_1', 'bad_debt_delayed_2',
                      'bad_debt_delayed_3', 'bad_debt_delayed_4', 'bad_debt_delayed_5']].head(10))

# Summarize bad debt totals
print("\n=== Summary of Bad Debt with Oracle Delay ===")
baseline_total = df_with_delays['bad_debt_baseline'].sum()
delay_1_total = df_with_delays['bad_debt_delayed_1'].sum()
delay_2_total = df_with_delays['bad_debt_delayed_2'].sum()
delay_3_total = df_with_delays['bad_debt_delayed_3'].sum()
delay_4_total = df_with_delays['bad_debt_delayed_4'].sum()
delay_5_total = df_with_delays['bad_debt_delayed_5'].sum()

print(f"Total baseline bad debt: {baseline_total:,.2f} USD")
print(f"Total worst-case bad debt (1-block delay): {delay_1_total:,.2f} USD")
print(f"Total worst-case bad debt (2-block delay): {delay_2_total:,.2f} USD")
print(f"Total worst-case bad debt (3-block delay): {delay_3_total:,.2f} USD")
print(f"Total worst-case bad debt (4-block delay): {delay_4_total:,.2f} USD")
print(f"Total worst-case bad debt (5-block delay): {delay_5_total:,.2f} USD")


#########################
##### Step 4: Compute Expected Bad Debt #####
#########################

# Oracle delay probabilities (Normal & Worse cases)
DELAY_PROB_NORMAL = {
    1: 0.10,
    2: 0.01,
    3: 0.001,
    4: 0.0001,
    5: 0.00001
}

DELAY_PROB_WORSE = {
    1: 0.14,
    2: 0.0196,
    3: 0.002744,
    4: 0.00038416,
    5: 0.0000537824
}

def step_4_compute_expected_bad_debt(
    df: pd.DataFrame,
    delay_prob_normal: dict = DELAY_PROB_NORMAL,
    delay_prob_worse: dict = DELAY_PROB_WORSE
) -> pd.DataFrame:
    """
    STEP 4:
      Compute the expected bad debt by weighting each delayed bad debt
      with the probability of an oracle delay for [1..5] blocks.

    Arguments:
      - df: DataFrame with bad debt calculations from Step 3.
      - delay_prob_normal: Probability of delay for normal-case scenario.
      - delay_prob_worse: Probability of delay for worse-case scenario.

    Returns:
      The same DataFrame with added columns:
        - expected_bad_debt_normal
        - expected_bad_debt_worse
    """

    # Compute expected bad debt per delay block
    for n in range(1, 6):
        df[f'expected_bad_debt_n{n}_normal'] = df[f'bad_debt_delayed_{n}'] * delay_prob_normal[n]
        df[f'expected_bad_debt_n{n}_worse'] = df[f'bad_debt_delayed_{n}'] * delay_prob_worse[n]

    # Compute total expected bad debt across all delay cases
    df['expected_bad_debt_total_normal'] = sum(df[f'expected_bad_debt_n{n}_normal'] for n in range(1, 6))
    df['expected_bad_debt_total_worse'] = sum(df[f'expected_bad_debt_n{n}_worse'] for n in range(1, 6))

    return df


# Step 4 usage (after step 3 is completed)
df_with_delays = step_3_model_oracle_delay(df_with_baseline)
df_with_expected_bd = step_4_compute_expected_bad_debt(df_with_delays)

# Print a sample of results with expected bad debt for each delay block
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Prevent line breaks

print(df_with_expected_bd[['evt_tx_hash', 'bad_debt_baseline',
                           'expected_bad_debt_n1_normal', 'expected_bad_debt_n2_normal',
                           'expected_bad_debt_n3_normal', 'expected_bad_debt_n4_normal', 'expected_bad_debt_n5_normal',
                           'expected_bad_debt_n1_worse', 'expected_bad_debt_n2_worse',
                           'expected_bad_debt_n3_worse', 'expected_bad_debt_n4_worse', 'expected_bad_debt_n5_worse',
                           'expected_bad_debt_total_normal', 'expected_bad_debt_total_worse']].head(10))

# Summarize expected bad debt across all events
print("\n=== Summary of Expected Bad Debt (Normal Case) ===")
total_expected_n1_normal = df_with_expected_bd['expected_bad_debt_n1_normal'].sum()
total_expected_n2_normal = df_with_expected_bd['expected_bad_debt_n2_normal'].sum()
total_expected_n3_normal = df_with_expected_bd['expected_bad_debt_n3_normal'].sum()
total_expected_n4_normal = df_with_expected_bd['expected_bad_debt_n4_normal'].sum()
total_expected_n5_normal = df_with_expected_bd['expected_bad_debt_n5_normal'].sum()
total_expected_n1_worse = df_with_expected_bd['expected_bad_debt_n1_worse'].sum()
total_expected_n2_worse = df_with_expected_bd['expected_bad_debt_n2_worse'].sum()
total_expected_n3_worse = df_with_expected_bd['expected_bad_debt_n3_worse'].sum()
total_expected_n4_worse = df_with_expected_bd['expected_bad_debt_n4_worse'].sum()
total_expected_n5_worse = df_with_expected_bd['expected_bad_debt_n5_worse'].sum()

print(f"Total Expected Bad Debt (1-block delay, Normal Case): {total_expected_n1_normal:,.2f} USD")
print(f"Total Expected Bad Debt (2-block delay, Normal Case): {total_expected_n2_normal:,.2f} USD")
print(f"Total Expected Bad Debt (3-block delay, Normal Case): {total_expected_n3_normal:,.2f} USD")
print(f"Total Expected Bad Debt (4-block delay, Normal Case): {total_expected_n4_normal:,.2f} USD")
print(f"Total Expected Bad Debt (5-block delay, Normal Case): {total_expected_n5_normal:,.2f} USD")
print("\n=== Summary of Expected Bad Debt (Worse Case) ===")
print(f"Total Expected Bad Debt (1-block delay, Worse Case): {total_expected_n1_worse:,.2f} USD")
print(f"Total Expected Bad Debt (2-block delay, Worse Case): {total_expected_n2_worse:,.2f} USD")
print(f"Total Expected Bad Debt (3-block delay, Worse Case): {total_expected_n3_worse:,.2f} USD")
print(f"Total Expected Bad Debt (4-block delay, Worse Case): {total_expected_n4_worse:,.2f} USD")
print(f"Total Expected Bad Debt (5-block delay, Worse Case): {total_expected_n5_worse:,.2f} USD")


print(f"\nTotal Expected Bad Debt (All Delays, Normal Case): {df_with_expected_bd['expected_bad_debt_total_normal'].sum():,.2f} USD")
print(f"Total Expected Bad Debt (All Delays, Worse Case): {df_with_expected_bd['expected_bad_debt_total_worse'].sum():,.2f} USD")


#####################
### Visualization ###
#####################
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib import font_manager as fm
import os

# === Replace with actual computed values ===
total_expected_normal = [total_expected_n1_normal, total_expected_n2_normal, total_expected_n3_normal, total_expected_n4_normal, total_expected_n5_normal]
total_expected_worse = [total_expected_n1_worse, total_expected_n2_worse, total_expected_n3_worse, total_expected_n4_worse, total_expected_n5_worse]
total_delayed_bad_debt = [delay_1_total, delay_2_total, delay_3_total, delay_4_total, delay_5_total]

# Falling blocks (X-axis)
falling_blocks = [1, 2, 3, 4, 5]

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
ax.plot(falling_blocks, total_expected_normal, color='#C6FF8A', linewidth=2, marker='o', label='Expected Bad Debt (Normal Case)', zorder=2)
ax.plot(falling_blocks, total_expected_worse, color='#FF6B6B', linewidth=2, marker='s', label='Expected Bad Debt (Worse Case)', zorder=2)
# ax.plot(falling_blocks, total_delayed_bad_debt, color='#4A90E2', linewidth=2, marker='d', label='Total Bad Debt (Delayed)', zorder=2)

# === X and Y Axis Formatting ===
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# === Set Labels and Title ===
ax.set_xlabel('Number of Falling Blocks', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Bad Debt (USD)', color='#ffffff', fontproperties=aeonik_font, fontsize=14, fontweight='bold', labelpad=15)
fig.suptitle('Expected Bad Debt under Oracle Delay Scenarios', color='#fbfbfc', x=0.08, y=0.95,
             fontproperties=at_aero_font, fontsize=30, fontweight='bold', ha='left')

# === Tick Customization ===
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, pad=15)
plt.setp(ax.get_xticklabels(), fontproperties=aeonik_font, fontsize=10, color='#98a2b3')
plt.setp(ax.get_yticklabels(), fontproperties=aeonik_font, fontsize=10, color='#98a2b3')

# === Add Custom Legend ===
lines = [
    mlines.Line2D([], [], color='#C6FF8A', marker='o', label='Expected Bad Debt (Normal Case)'),
    mlines.Line2D([], [], color='#FF6B6B', marker='s', label='Expected Bad Debt (Extreme Case)'),
    # mlines.Line2D([], [], color='#4A90E2', marker='d', label='Total Bad Debt (Delayed)')
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

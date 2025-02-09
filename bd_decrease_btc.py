import pandas as pd
import numpy as np

#########################
##### Configuration #####
#########################

LIQUIDATION_BONUS = 0.05

def sqrtPriceX96_to_price(sqrt_price_x96: float) -> float:
    return (sqrt_price_x96 / 2**96) ** 2 * 1e2

def compute_bad_debt_baseline(
        debt_amount: float,
        debtAsset_price: float,  # <-- Updated variable name
        liquidated_collateral_amount: float,
        btc_price: float
) -> float:
    """
    Compute the baseline bad debt for a single liquidation event, given:
      - debt_amount (tokens)
      - debtAsset_price (USD/token)
      - liquidated_collateral_amount (btc)
      - btc_price (USD/btc)

    The formula used:
      badDebt = max(0, (debt_amount * debtAsset_price)
                         - (liquidated_collateral_amount * btc_price))
    """
    debt_value_usd = debt_amount * debtAsset_price
    collateral_value_usd = liquidated_collateral_amount * btc_price

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

    # (B) Read BTC slot0 data
    df_btc_slot0 = pd.read_csv(slot0_csv)
    df_btc_slot0['sqrtPriceX96'] = pd.to_numeric(df_btc_slot0['sqrtPriceX96'], errors='coerce')
    df_btc_slot0['btc_price'] = df_btc_slot0['sqrtPriceX96'].apply(sqrtPriceX96_to_price)

    # Rename 'blocknumber' to 'block_number'
    df_btc_slot0.rename(columns={'blocknumber': 'block_number'}, inplace=True)

    # Keep only relevant columns
    df_btc_slot0 = df_btc_slot0[['block_number', 'btc_price']]

    # (C) Merge on block_number
    df_merged = pd.merge(
        df_liquidations,
        df_btc_slot0,
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
            btc_price=row['btc_price']
        )

    df_merged['bad_debt_baseline'] = df_merged.apply(_calc_baseline_bd, axis=1)
    return df_merged

# Call Step 1 and Step 2
if __name__ == "__main__":
    # Example usage:
    liquidations_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/btc_decrease_liquidations.csv"
    slot0_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/BTC_slot0_19465095_19472214 copy.csv"

    df_merged = step_1_load_and_merge(liquidations_file, slot0_file)
    df_with_baseline = step_2_compute_baseline_bad_debt(df_merged)

    # Inspect baseline results
    print(df_with_baseline[['evt_tx_hash', 'evt_block_number',
                            'debt_amount', 'liquidated_collateral_amount',
                            'debtAsset_price', 'btc_price',
                            'bad_debt_baseline']].head(10))

    total_baseline = df_with_baseline['bad_debt_baseline'].sum()
    print(f"\nTotal Baseline Bad Debt: {total_baseline:,.2f} USD")


# Step 3: Oracle Delay & Expected Bad Debt #####


# Progressive btc price drops due to oracle delay
PROGRESSIVE_DROPS = {
    1: 0.0033,   # 0.33%
    2: 0.0072,   # 0.72% (0.33% + 0.39%)
    3: 0.0105,   # 1.05% (0.33% + 0.39% + 0.33%)
    4: 0.0120,   # 1.20% (0.33% + 0.39% + 0.33% + 0.15%)
    5: 0.0153    # 1.53% (0.33% + 0.39% + 0.33% + 0.15% + 0.33%)
}

def step_3_model_oracle_delay(
    df: pd.DataFrame,
    progressive_drops: dict = PROGRESSIVE_DROPS,
    liquidation_bonus: float = LIQUIDATION_BONUS
) -> pd.DataFrame:

    def _compute_delayed_bd(row, drop_fraction: float):
        """
        Compute bad debt for a single liquidation event given an btc price drop.
        """
        # 1) Compute new btc price under delayed scenario
        delayed_btc_price = row['btc_price'] * (1.0 - drop_fraction)

        # 2) Compute new collateral value in USD under delayed price
        collateral_value_new = row['liquidated_collateral_amount'] * delayed_btc_price

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

# Step 4: Compute Expected Bad Debt #####

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

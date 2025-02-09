import pandas as pd
import numpy as np

#########################
##### Configuration #####
#########################

LIQUIDATION_BONUS = 0.083

def sqrtPriceX96_to_price(sqrt_price_x96: float) -> float:
    """Convert sqrtPriceX96 to token ratio (CRV per ETH)."""
    return (sqrt_price_x96 / 2**96) ** 2

def sqrt_price_to_token_ratio_eth(sqrt_price_x96):
    """Convert sqrtPriceX96 to ETH price in USD."""
    return ((sqrt_price_x96 / (2 ** 96)) ** 2) * 1e12

def compute_bad_debt_baseline(
        debt_amount: float,
        debtAsset_price: float,
        liquidated_collateral_amount: float,
        crv_usd_price: float
) -> float:
    """
    Compute the baseline bad debt for a single liquidation event.
    """
    debt_value_usd = debt_amount * debtAsset_price
    collateral_value_usd = liquidated_collateral_amount * crv_usd_price

    return max(0.0, debt_value_usd - collateral_value_usd)

#########################
##### Implementation ####
#########################

# Step 1: Load Data and Merge
def step_1_load_and_merge(liquidations_csv: str, crv_slot0_csv: str, eth_slot0_csv: str) -> pd.DataFrame:
    # Load liquidations dataset
    df_liquidations = pd.read_csv(liquidations_csv)
    df_liquidations[['liquidated_collateral_amount', 'debt_amount', 'debtAsset_price']] = \
        df_liquidations[['liquidated_collateral_amount', 'debt_amount', 'debtAsset_price']].apply(pd.to_numeric, errors='coerce')

    # Load CRV slot0 dataset
    df_crv_slot0 = pd.read_csv(crv_slot0_csv)
    df_crv_slot0['sqrtPriceX96'] = pd.to_numeric(df_crv_slot0['sqrtPriceX96'], errors='coerce')
    df_crv_slot0['crv_per_eth'] = df_crv_slot0['sqrtPriceX96'].apply(sqrtPriceX96_to_price)

    # Load ETH slot0 dataset (for ETH/USD price)
    df_eth_slot0 = pd.read_csv(eth_slot0_csv)
    df_eth_slot0['sqrtPriceX96'] = pd.to_numeric(df_eth_slot0['sqrtPriceX96'], errors='coerce')
    df_eth_slot0['eth_usd_price'] = df_eth_slot0['sqrtPriceX96'].apply(sqrt_price_to_token_ratio_eth)

    # Rename 'blocknumber' to 'block_number' for CRV and ETH datasets
    df_crv_slot0.rename(columns={'blocknumber': 'block_number'}, inplace=True)
    df_eth_slot0.rename(columns={'blocknumber': 'block_number'}, inplace=True)

    # Merge CRV price data
    df_merged = pd.merge(df_liquidations, df_crv_slot0[['block_number', 'crv_per_eth']],
                         left_on='evt_block_number', right_on='block_number', how='left')

    # Merge ETH price data
    df_merged = pd.merge(df_merged, df_eth_slot0[['block_number', 'eth_usd_price']],
                         on='block_number', how='left')

    # Compute CRV price in USD
    df_merged['crv_usd_price'] = df_merged['eth_usd_price'] / df_merged['crv_per_eth']

    return df_merged

# Step 2: Compute Baseline Bad Debt
def step_2_compute_baseline_bad_debt(df_merged: pd.DataFrame) -> pd.DataFrame:

    def _calc_baseline_bd(row):
        return compute_bad_debt_baseline(
            debt_amount=row['debt_amount'],
            debtAsset_price=row['debtAsset_price'],
            liquidated_collateral_amount=row['liquidated_collateral_amount'],
            crv_usd_price=row['crv_usd_price']
        )

    df_merged['bad_debt_baseline'] = df_merged.apply(_calc_baseline_bd, axis=1)
    return df_merged

# Call Step 1 and Step 2
if __name__ == "__main__":
    # Example usage:
    liquidations_file = "//Users/hanson_zhang/PycharmProjects/svr_research/data/crv_decrease_liquidations.csv"
    crv_slot0_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/CRV_slot0_20079184_20086327 copy.csv"
    eth_slot0_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/ETH_slot0_20079184_20086327 copy.csv"

    df_merged = step_1_load_and_merge(liquidations_file, crv_slot0_file, eth_slot0_file)
    df_with_baseline = step_2_compute_baseline_bad_debt(df_merged)

    # Ensure full DataFrame is printed
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Adjust width to prevent wrapping
    pd.set_option('display.colheader_justify', 'center')  # Align column headers

    # Inspect results
    print(df_with_baseline[['evt_tx_hash', 'evt_block_number',
                            'debt_amount', 'liquidated_collateral_amount',
                            'debtAsset_price', 'eth_usd_price', 'crv_per_eth', 'crv_usd_price',
                            'bad_debt_baseline']].head(10))

    total_baseline = df_with_baseline['bad_debt_baseline'].sum()
    print(f"\nTotal Baseline Bad Debt: {total_baseline:,.2f} USD")



# Step 3: Oracle Delay & Expected Bad Debt #####


# Progressive crv price drops due to oracle delay
PROGRESSIVE_DROPS = {
    1: 0.1403,   # 14.03%
    2: 0.2673,   # 26.73% (14.03% + 12.70%)
    3: 0.3581,   # 35.81% (14.03% + 12.70% + 9.08%)
    4: 0.4506,   # 45.06% (14.03% + 12.70% + 9.08% + 9.25%)
    5: 0.5414    # 54.14% (14.03% + 12.70% + 9.08% + 9.25% + 9.08%)
}

def step_3_model_oracle_delay(
    df: pd.DataFrame,
    progressive_drops: dict = PROGRESSIVE_DROPS,
    liquidation_bonus: float = LIQUIDATION_BONUS
) -> pd.DataFrame:

    def _compute_delayed_bd(row, drop_fraction: float):
        """
        Compute bad debt for a single liquidation event given an crv price drop.
        """
        # 1) Compute new crv price under delayed scenario
        delayed_crv_price = row['crv_usd_price'] * (1.0 - drop_fraction)

        # 2) Compute new collateral value in USD under delayed price
        collateral_value_new = row['liquidated_collateral_amount'] * delayed_crv_price

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

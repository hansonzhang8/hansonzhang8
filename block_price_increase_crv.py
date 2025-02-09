import pandas as pd

def sqrt_price_to_token_ratio(sqrt_price_x96):
    """Convert sqrtPriceX96 to the price ratio"""
    return (sqrt_price_x96 / (2 ** 96)) ** 2

def sqrt_price_to_token_ratio_eth(sqrt_price_x96):
    """Convert sqrtPriceX96 to the price ratio"""
    return ((sqrt_price_x96 / (2 ** 96)) ** 2) * 1e12

def percentage_increase(start, end):
    """Safely compute percentage increase from start to end."""
    if start != 0:
        return (end - start) / start * 100
    return 0

def find_largest_increase(df, group_size):
    """
    Identify the largest final-step percentage increase in CRV/USD price.
    """
    df_sorted = df.sort_values(by='blocknumber').reset_index(drop=True)

    max_increase = None
    best_blocks = None

    # Step in increments of 'group_size' to avoid overlap
    for i in range(0, len(df_sorted) - group_size + 1, group_size):
        subset = df_sorted.iloc[i : i + group_size]
        blocks = subset['blocknumber'].tolist()
        prices = subset['crv_usd_price'].astype(float).tolist()

        # Check if the entire window is non-decreasing
        if all(prices[j] <= prices[j+1] for j in range(len(prices) - 1)):
            # Measure the final-step increase: (second-to-last) -> (last)
            inc = percentage_increase(prices[-2], prices[-1])
            if max_increase is None or inc > max_increase:
                max_increase = inc
                best_blocks = blocks

    if max_increase is not None:
        print(f"[group_size={group_size}] Largest final-step increase: {max_increase:.2f}% "
              f"(blocks {best_blocks[-2]} -> {best_blocks[-1]}) in window {best_blocks}")
    else:
        print(f"[group_size={group_size}] No valid non-decreasing group found.")

# ===============================
# Load Data
# ===============================
eth_crv_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/CRV_slot0_21296777_21303933 copy.csv"
eth_usd_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/ETH_slot0_21296777_21303933 copy.csv"

df_eth_crv = pd.read_csv(eth_crv_file)
df_eth_usd = pd.read_csv(eth_usd_file)

# Convert sqrtPriceX96 to real price values
df_eth_crv["sqrtPriceX96"] = pd.to_numeric(df_eth_crv["sqrtPriceX96"], errors="coerce")  # Convert to float
df_eth_crv["eth_per_crv"] = df_eth_crv["sqrtPriceX96"].apply(sqrt_price_to_token_ratio)
df_eth_crv["crv_per_eth"] = 1 / df_eth_crv["eth_per_crv"]  # Convert to CRV per ETH

df_eth_usd["sqrtPriceX96"] = pd.to_numeric(df_eth_usd["sqrtPriceX96"], errors="coerce")  # Convert to float
df_eth_usd["eth_usd_price"] = df_eth_usd["sqrtPriceX96"].apply(sqrt_price_to_token_ratio_eth)  # ETH/USD price

# Merge datasets based on blocknumber
df_merged = pd.merge(df_eth_crv, df_eth_usd, on="blocknumber", suffixes=("_crv", "_usd"))

# Compute CRV price in USD
df_merged["crv_usd_price"] = df_merged["eth_usd_price"] / df_merged["eth_per_crv"]

# ===============================
# Run Analysis
# ===============================
for size in range(2, 7):
    find_largest_increase(df_merged, size)

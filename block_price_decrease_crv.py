import pandas as pd

def sqrt_price_to_token_ratio(sqrt_price_x96):
    """Convert sqrtPriceX96 to the price ratio"""
    return (sqrt_price_x96 / (2 ** 96)) ** 2

def sqrt_price_to_token_ratio_eth(sqrt_price_x96):
    """Convert sqrtPriceX96 to the price ratio"""
    return ((sqrt_price_x96 / (2 ** 96)) ** 2) * 1e12

def percentage_decrease(start, end):
    """Safely compute percentage decrease from start to end."""
    if start != 0:
        return (start - end) / start * 100
    return 0

def find_largest_decrease(df, group_size):
    """
    Identify the largest final-step percentage decrease in CRV/USD price.
    """
    df_sorted = df.sort_values(by='blocknumber').reset_index(drop=True)

    max_decrease = None
    best_blocks = None

    # Step in increments of 'group_size' to avoid overlap
    for i in range(0, len(df_sorted) - group_size + 1, group_size):
        subset = df_sorted.iloc[i : i + group_size]
        blocks = subset['blocknumber'].tolist()
        prices = subset['crv_usd_price'].astype(float).tolist()

        # Check if the entire window is non-increasing
        if all(prices[j] >= prices[j+1] for j in range(len(prices) - 1)):
            # Measure the final-step decrease: (second-to-last) -> (last)
            dec = percentage_decrease(prices[-2], prices[-1])
            if max_decrease is None or dec > max_decrease:
                max_decrease = dec
                best_blocks = blocks

    if max_decrease is not None:
        print(f"[group_size={group_size}] Largest final-step decrease: {max_decrease:.2f}% "
              f"(blocks {best_blocks[-2]} -> {best_blocks[-1]}) in window {best_blocks}")
    else:
        print(f"[group_size={group_size}] No valid non-increasing group found.")

# ===============================
# Load Data
# ===============================
eth_crv_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/CRV_slot0_20079184_20086327 copy.csv"
eth_usd_file = "/Users/hanson_zhang/PycharmProjects/svr_research/data/ETH_slot0_20079184_20086327 copy.csv"

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
    find_largest_decrease(df_merged, size)

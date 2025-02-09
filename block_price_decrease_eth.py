import pandas as pd

def percentage_decrease(start, end):
    """Safely compute percentage decrease from start to end."""
    if start != 0:
        return (start - end) / start * 100
    return 0

def find_largest_decrease(df, group_size):
    """
    For each non-overlapping window of size 'group_size':
      1. Check if it's non-increasing.
      2. Calculate the price movement from the second-to-last block to the last block.
      3. Track and finally print the single largest (final-step) decrease across all valid windows.
    """
    # Sort just to be safe
    df_sorted = df.sort_values(by='blocknumber').reset_index(drop=True)

    max_decrease = None
    best_blocks = None

    # Step in increments of 'group_size' to avoid overlap
    for i in range(0, len(df_sorted) - group_size + 1, group_size):
        subset = df_sorted.iloc[i : i + group_size]
        blocks = subset['blocknumber'].tolist()
        prices = subset['sqrtPriceX96'].astype(float).tolist()

        # Check if the entire window is non-increasing
        if all(prices[j] >= prices[j+1] for j in range(len(prices) - 1)):
            # Measure the final-step decrease: (second-to-last) -> (last)
            dec = percentage_decrease(prices[-2], prices[-1])
            # Keep track of the single largest final-step decrease
            if max_decrease is None or dec > max_decrease:
                max_decrease = dec
                best_blocks = blocks

    # Report the best (largest) found, if any
    if max_decrease is not None:
        print(f"[group_size={group_size}] Largest final-step decrease: {max_decrease:.2f}% "
              f"(blocks {best_blocks[-2]} -> {best_blocks[-1]}) in window {best_blocks}")
    else:
        print(f"[group_size={group_size}] No valid non-increasing group found.")

# ===============================
# Usage
# ===============================
file_path = "/Users/hanson_zhang/PycharmProjects/svr_research/data/ETH_slot0_21755110_21762264.csv"
df = pd.read_csv(file_path)

# Compute for window sizes 2 through 6
for size in range(2, 7):
    find_largest_decrease(df, size)

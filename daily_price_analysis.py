import pandas as pd

# Load CSV file (Make sure the file path is correct)
file_path = "/Users/hanson_zhang/PycharmProjects/svr_research/data/link-usd-max.csv"
df = pd.read_csv(file_path)

# Convert date column to datetime
df["snapped_at"] = pd.to_datetime(df["snapped_at"])

# Filter for dates only after 2022-01-01
df = df[df["snapped_at"] > "2023-01-01"]

# Calculate daily price changes
df["price_change"] = df["price"].pct_change()

# Find the max increase and max decrease
max_increase_idx = df["price_change"].idxmax()
max_decrease_idx = df["price_change"].idxmin()

# Ensure the price change is attributed to the correct date (previous day)
max_increase_date = df.loc[max_increase_idx - 1, "snapped_at"].date()
max_increase_pct = df.loc[max_increase_idx, "price_change"] * 100
max_increase_old_price = df.loc[max_increase_idx - 1, "price"]
max_increase_new_price = df.loc[max_increase_idx, "price"]

max_decrease_date = df.loc[max_decrease_idx - 1, "snapped_at"].date()
max_decrease_pct = df.loc[max_decrease_idx, "price_change"] * 100
max_decrease_old_price = df.loc[max_decrease_idx - 1, "price"]
max_decrease_new_price = df.loc[max_decrease_idx, "price"]

# Print results
print(f"Largest price increase: On {max_increase_date}, price jumped from ${max_increase_old_price:.2f} → ${max_increase_new_price:.2f} (+{max_increase_pct:.2f}%)")
print(f"Largest price drop: On {max_decrease_date}, price fell from ${max_decrease_old_price:.2f} → ${max_decrease_new_price:.2f} (-{max_decrease_pct:.2f}%)")

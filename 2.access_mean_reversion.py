#############################
## Load necessary packages ##
#############################
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os

####################
## Importing Data ##
####################
# Construct the file path
desktop_path = os.path.expanduser('~/Desktop')
file_path = os.path.join(desktop_path, 'mev_research', 'mev_data', 'mev_2023_q1_cleaned.csv')
# Load the CSV file into a pandas DataFrame
mev_2023_q1_cleaned = pd.read_csv(file_path)
print(mev_2023_q1_cleaned.head())
print(mev_2023_q1_cleaned.columns)

#####################
## Access ADF Test ##
#####################
# Define the log return
log_return_block_level = mev_2023_q1_cleaned['log_return']

# Check for stationary:
result = adfuller(log_return_block_level)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print('Critical Value ({}): {}'.format(key, value))

# ADF Statistic: -119.68041226566089
# p-value: 0.0
# Critical Value (1%): -3.430360225877967
# Critical Value (5%): -2.8615445197198714
# Critical Value (10%): -2.5667724056814087
# Since the ADF statistic is much lower than the critical values at all significance levels
# And the p-value is 0.0 (which is less than 0.05)
# We can reject the null hypothesis of the ADF test.

##############################
## Access Log_Return Visual ##
##############################

# Plot log returns
plt.figure(figsize=(12, 6))
plt.plot(log_return_block_level, label='Log Return')
plt.title('Time Series Plot of Log Returns')
plt.xlabel('Time')
plt.ylabel('Log Return')
plt.legend()
plt.show()

# Plot Rolling Variance 16 Slots
rolling_variance_16slots = log_return_block_level.rolling(window=16).var()
plt.figure(figsize=(12, 6))
plt.plot(rolling_variance_16slots, color='g', label='Rolling Variance')
plt.title('Rolling Variance 16 Slots')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot Rolling Mean and Variance
rolling_variance_32slots = log_return_block_level.rolling(window=32).var()
plt.figure(figsize=(12, 6))
plt.plot(rolling_variance_32slots, color='g', label='Rolling Variance')
plt.title('Rolling Variance 32 Slots')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
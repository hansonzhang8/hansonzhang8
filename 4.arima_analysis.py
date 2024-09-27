#############################
## Load necessary packages ##
#############################
#############################
## Load necessary packages ##
#############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
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
# Check the 'log_return' series
mev_2023_q1_cleaned['log_return'].plot(title='Log Returns Over Time')
plt.show()

######################################
## Variance ARIMA Analysis 16 Slots ##
######################################
# Define variable
log_returns = mev_2023_q1_cleaned['log_return'].values
actual_prices = mev_2023_q1_cleaned['proposer_total_reward_in_eth'].values
log_prices = np.log(actual_prices)

# Define window size
window_size = 16
# Calculate block variance 16 slots
block_variances_16blocks = [log_returns[i:i+window_size].var() for i in range(0, len(log_returns), window_size) if i+window_size <= len(log_returns)]
# Convert to Series for further analysis
rolling_variance_16slots = pd.Series(block_variances_16blocks)

# Plotting ACF for 16 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_acf(rolling_variance_16slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Autocorrelation Function (ACF) for Rolling Variance 16slots')
plt.show()

# Plotting PACF for 16 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_pacf(rolling_variance_16slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Partial Autocorrelation Function (PACF) for Rolling Variance 16slots')
plt.show()

######################################
## Variance ARIMA Analysis 32 Slots ##
######################################
# Define window size
window_size = 32
# Calculate block variance 32 slots
block_variances_32blocks = [log_returns[i:i+window_size].var() for i in range(0, len(log_returns), window_size) if i+window_size <= len(log_returns)]
# Convert to Series for further analysis
rolling_variance_32slots = pd.Series(block_variances_32blocks)

# Plotting ACF for 32 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_acf(rolling_variance_32slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Autocorrelation Function (ACF) for Rolling Variance 32slots')
plt.show()

# Plotting PACF for 32 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_pacf(rolling_variance_32slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Partial Autocorrelation Function (PACF) for Rolling Variance 32slots')
plt.show()

##########################
## Check for Stationary ##
##########################
result = adfuller(rolling_variance_32slots)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

######################################
## Check Lag Signifiance with ARIMA ##
######################################
# Define the ARIMA model
model = ARIMA(rolling_variance_32slots, order=(1, 0, 5))
# Fit the model
model_fit = model.fit()
# Print model summary
print(model_fit.summary())

######################################
## Variance ARIMA Analysis 150 Slots ##
######################################
# Define window size
window_size = 150
# Calculate block variance 150 slots
block_variances_150blocks = [log_returns[i:i+window_size].var() for i in range(0, len(log_returns), window_size) if i+window_size <= len(log_returns)]
# Convert to Series for further analysis
rolling_variance_150slots = pd.Series(block_variances_150blocks)

# Plotting ACF for 150 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_acf(rolling_variance_150slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Autocorrelation Function (ACF) for Rolling Variance 150slots')
plt.show()

# Plotting PACF for 150 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_pacf(rolling_variance_150slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Partial Autocorrelation Function (PACF) for Rolling Variance 150slots')
plt.show()

##########################
## Check for Stationary ##
##########################
result = adfuller(rolling_variance_150slots)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

######################################
## Check Lag Signifiance with ARIMA ##
######################################
# Define the ARIMA model
model = ARIMA(rolling_variance_150slots, order=(10, 0, 5))
# Fit the model
model_fit = model.fit()
# Print model summary
print(model_fit.summary())

######################################
##Variance ARIMA Analysis 300 Slots ##
######################################
# Define window size
window_size = 300
# Calculate block variance
block_variances_300blocks = [log_returns[i:i+window_size].var() for i in range(0, len(log_returns), window_size) if i+window_size <= len(log_returns)]
# Convert to Series for further analysis
rolling_variance_300slots = pd.Series(block_variances_300blocks)

# Plotting ACF for 300 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_acf(rolling_variance_300slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Autocorrelation Function (ACF) for Rolling Variance 300slots')
plt.show()

# Plotting PACF for 300 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_pacf(rolling_variance_300slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Partial Autocorrelation Function (PACF) for Rolling Variance 300slots')
plt.show()

##########################
## Check for Stationary ##
##########################
result = adfuller(rolling_variance_300slots)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

######################################
## Check Lag Signifiance with ARIMA ##
######################################
# Define the ARIMA model
model = ARIMA(rolling_variance_300slots, order=(5, 0, 5))
# Fit the model
model_fit = model.fit()
# Print model summary
print(model_fit.summary())

######################################
##Variance ARIMA Analysis 900 Slots ##
######################################
# Define window size
window_size = 900
# Calculate block variance 900 slots
block_variances_900blocks = [log_returns[i:i+window_size].var() for i in range(0, len(log_returns), window_size) if i+window_size <= len(log_returns)]
# Convert to Series for further analysis
rolling_variance_900slots = pd.Series(block_variances_900blocks)
print(len(rolling_variance_900slots))
# Plotting ACF for 900 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_acf(rolling_variance_900slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Autocorrelation Function (ACF) for Rolling Variance 900slots')
plt.show()

# Plotting PACF for 900 Slots rolling variance
plt.figure(figsize=(12, 6))
plot_pacf(rolling_variance_900slots, lags=40, alpha=0.05)  # Change lags as needed
plt.title('Partial Autocorrelation Function (PACF) for Rolling Variance 900slots')
plt.show()

##########################
## Check for Stationary ##
##########################
result = adfuller(rolling_variance_900slots)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

######################################
## Check Lag Signifiance with ARIMA ##
######################################
# Define the ARIMA model
model = ARIMA(rolling_variance_900slots, order=(2, 0, 3))
# Fit the model
model_fit = model.fit()
# Print model summary
print(model_fit.summary())
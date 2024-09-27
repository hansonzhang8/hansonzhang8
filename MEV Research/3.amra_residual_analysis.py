#############################
## Load necessary packages ##
#############################
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import shapiro
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller
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

##############################
## Determine the ARMA Order ##
##############################
# Plot the ACF and PACF
# lags = 20 means to display the correlation between the series and its lagged values up to 20 periods back
# acf will give q, if q = 10, it means the model includes the previous 10 forecasting errors in its calculations.
plot_acf(mev_2023_q1_cleaned['log_return'], lags=20)
# pacf will give p, if p = 10, it means the current value of the series is potentially influenced by the previous 10 values
plot_pacf(mev_2023_q1_cleaned['log_return'], lags=20)
plt.show()

########################
## Fit the ARMA Model ##
########################
# Fit the model
q = 1  # From our previous step
p = 3  # From our previous step
model = SARIMAX(mev_2023_q1_cleaned['log_return'], order=(p, 0, q), seasonal_order=(0, 0, 0, 0))
results = model.fit()
print(results.summary())

########################
## Residual Analysis ##
########################
# Extract residuals
residuals = results.resid
# Plot residuals
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('Residuals of MEV Log Return from ARMA model')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Histogram
plt.hist(residuals, bins=30, alpha=0.7, color='b', density=True)
plt.title('Histogram of Residuals')
plt.show()

# Tests for Normality
# Shapiro-Wilk Test
stat, p = shapiro(residuals)
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Residuals seem to follow a normal distribution (fail to reject H0)')
else:
    print('Residuals do not follow a normal distribution (reject H0)')

# Test for Skewness
res_skew = skew(residuals)
print('Skewness of residuals: ', res_skew)

# Test for Kurtosis
res_kurtosis = kurtosis(residuals)
print('Kurtosis of residuals: ', res_kurtosis)

# Conduct ADF Test
adf_test = adfuller(residuals)
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])
print('Critical Values:')
for key, value in adf_test[4].items():
    print('\t%s: %.3f' % (key, value))
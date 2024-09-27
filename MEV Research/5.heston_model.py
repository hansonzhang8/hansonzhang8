#############################
## Load necessary packages ##
#############################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.optimize as opt
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

###############################################
## Split data into training and testing sets ##
###############################################
# Determine the split index
split_index = int(len(mev_2023_q1_cleaned) * 0.8)

# Split the data into training and testing sets based on the sequence
train_data = mev_2023_q1_cleaned[:split_index]
test_data = mev_2023_q1_cleaned[split_index:]

# Display the shapes of the resulting datasets
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Save the splits to new CSV files
train_data.to_csv('mev_2023_q1_train.csv', index=False)
test_data.to_csv('mev_2023_q1_test.csv', index=False)

# Display the first few rows of the dataset
# Read the training dataset
mev_2023_q1_train = pd.read_csv('../mev_2023_q1_train.csv')
# Read the testing dataset
mev_2023_q1_test = pd.read_csv('../mev_2023_q1_test.csv')

# Print the training dataset
print(mev_2023_q1_train.head())
# Print the testing dataset
print(mev_2023_q1_test.head())

# Load the train data
log_returns_train = mev_2023_q1_train['log_return'].values
actual_prices_train = mev_2023_q1_train['proposer_total_reward_in_eth'].values
log_prices_train = np.log(actual_prices_train)

# Load the test data
log_returns_test = mev_2023_q1_test['log_return'].values
actual_prices_test = mev_2023_q1_test['proposer_total_reward_in_eth'].values
log_prices_test = np.log(actual_prices_test)

# Check Log Transition Train
print("len of train_actual_price:", len(actual_prices_train))  # Number of train actual price
print("len of train_log_price:", len(log_prices_train))  # Number of train log price
print("mean of train_actual_price:", np.mean(actual_prices_train))  # mean of train actual price
print("mean of train_log_price:", np.mean(log_prices_train))  # mean of train log price

# Check Log Transition Test
print("len of test_actual_price:", len(actual_prices_test))  # Number of test actual price
print("len of test_log_price:", len(log_prices_test))  # Number of test log price
print("mean of test_actual_price:", np.mean(actual_prices_test))  # mean of test actual price
print("mean of test_log_price:", np.mean(log_prices_test))  # mean of test log price

#############################################
## Find Initial Parameters Value 32 Slots ##
#############################################
# Define window size
window_size = 32
# Calculate block variance
block_variances_32blocks = [log_returns_train[i:i + window_size].var() for i in
                            range(0, len(log_returns_train), window_size) if i + window_size <= len(log_returns_train)]
# Convert to Series for further analysis
rolling_variance_32slots = pd.Series(block_variances_32blocks)


# Estimate initial kappa and sigma
def estimate_kappa_sigma(rolling_variance, dt):
    # Calculate the lagged series
    variance_t = rolling_variance[:-1]
    variance_t1 = rolling_variance[1:]

    # Estimate kappa
    covariance = np.cov(variance_t, variance_t1)[0, 1]
    variance = np.var(variance_t)
    kappa = -np.log(covariance / variance) / dt

    # Estimate sigma
    diff_variance = np.diff(rolling_variance)
    sigma = np.sqrt(np.var(diff_variance) / dt)

    return kappa, sigma


# Find Kappa and Sigma
dt = 1 / 82125
initial_kappa, initial_sigma = estimate_kappa_sigma(rolling_variance_32slots, dt)
print("Initial kappa is: ", initial_kappa)
print("Initial sigma is: ", initial_sigma)

# Estimate initial Mu
initial_mu = np.mean(log_returns_train)
print("Initial mu is: ", initial_mu)

# Estimate initial theta
initial_theta = np.var(log_returns_train)
print("Initial theta is: ", initial_theta)


#########################################
## Log Likelihood Estimation 32 Slots ##
#########################################
def heston_log_likelihood(params, log_prices):
    mu, kappa, theta, sigma, rho = params
    n = len(log_prices)
    dt = 1 / 82125

    # Initialize the variance and log-likelihood
    v = np.var(np.diff(log_prices))  # Initial variance estimate
    log_likelihood = 0

    # Generate correlated Brownian motions
    dW = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n - 1) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, n):
        v_next = max(0, v) + kappa * (theta - max(0, v)) * dt + sigma * np.sqrt(max(0, v)) * dW2[i - 1]
        dY = (mu - 0.5 * max(0, v)) * dt + np.sqrt(max(0, v)) * dW1[i - 1]
        log_likelihood += -0.5 * (np.log(2 * np.pi * max(0.0001, v)) + (log_prices[i - 1] - dY) ** 2 / max(0.0001, v))
        v = v_next

    return -log_likelihood


def optimize_parameters(log_prices):
    # Define bounds and initial guesses
    bounds = [(None, None), (0, None), (0, None), (0, None), (-1, 1)]
    initial_guess = [initial_mu, initial_kappa, initial_theta, initial_sigma, 0]

    # Minimize the negative log likelihood
    result = opt.minimize(heston_log_likelihood, initial_guess, args=(log_prices), bounds=bounds)

    return result.x


optimized_params = optimize_parameters(log_prices_train)
print("Adjusted parameters:", optimized_params)

estimated_mu, estimated_kappa, estimated_theta, estimated_sigma, estimated_rho = optimized_params

#####################################
## Monte Carlo Simulation 32 Slots ##
#####################################
T = 18 / 365  # Total time in years for simulation (18 days)
dt = 1 / 82125  # Delta t in years per epoch
num_paths = 1000  # Number of Monte Carlo paths
N = 4048  # Total number of time steps (Per Epoch)

# Initial conditions
S0 = np.median(log_prices_train)  # Start from the median log price in the train dataset
v0 = np.var(log_returns_train)  # Initial variance, estimated from the train log returns

# Arrays to store simulations
S = np.zeros((N, num_paths))
v = np.zeros((N, num_paths))

S[0, :] = S0
v[0, :] = v0

# Generate correlated random shocks for each path
for path in range(num_paths):
    dW = np.random.multivariate_normal([0, 0], [[1, estimated_rho], [estimated_rho, 1]], N) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, N):
        v_prev = v[i - 1, path]
        v_drift = estimated_kappa * (estimated_theta - np.abs(v_prev)) * dt
        v_diffusion = estimated_sigma * np.sqrt(np.abs(v_prev)) * dW2[i - 1]
        v[i, path] = np.abs(v_prev + v_drift + v_diffusion)
        S[i, path] = S[i - 1, path] + (estimated_mu - 0.5 * max(v[i, path], 0)) * dt + np.sqrt(max(v[i, path], 0)) * dW1[i - 1]

print("Shape of v:", v.shape)

# Plotting the results for all the paths
plt.figure(figsize=(12, 6))
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
plt.title('Simulated Log Price Dynamics Over Test Period Update Per Epoch')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.show()

# Average log prices over fixed intervals
epoch_prices = np.array([np.mean(log_prices_test[i:i+32]) for i in range(0, len(log_prices_test), 32)])
total_epochs = len(epoch_prices)

# Mean of all paths
mean_S = np.mean(S, axis=1)
median_S = np.median(S, axis=1)
plt.figure(figsize=(12, 6))

# Time arrays for plotting
simulated_times = np.linspace(0, T, N)
actual_times = np.linspace(0, T, total_epochs)

# Plot simulated data
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
plt.plot(actual_times, epoch_prices, label='Average Log Prices Per Epoch', color='green')
plt.title('Comparison of Actual and Simulated Log Prices Per Epoch')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.legend()
plt.show()

# Calculate RMSE & MAE between the actual epoch prices and the simulated mean prices
mean_S = np.mean(S, axis=1)
simulated_epoch_prices = mean_S[:total_epochs]
# Calculate RMSE between the actual epoch prices and the simulated mean prices
rmse = np.sqrt(mean_squared_error(epoch_prices, simulated_epoch_prices))
print(f"RMSE between actual and simulated log prices: {rmse}")
# Calculate MAE between the actual epoch prices and the simulated mean prices
mae = np.mean(np.abs(epoch_prices - simulated_epoch_prices))
print(f"MAE between actual and simulated log prices: {mae}")


##################################
## Volatility Analysis 32 Slots ##
##################################
# Simulation parameters
T = 18 / 365  # Total simulation time in years (18 days)
num_paths = 1000  # Number of Monte Carlo paths
N = 4048  # Expected number of time steps (18 days, per epoch)

# Rolling variance calculation parameters
window_size = 32

# Calculate rolling variance using windows of 32 time steps
block_variances_32blocks = [
    log_returns_test[i:i+window_size].var()
    for i in range(0, len(log_returns_test), window_size)
    if i + window_size <= len(log_returns_test)
]

# Convert variance data to a pandas Series for analysis
actual_variance = pd.Series(block_variances_32blocks)
actual_variance_diff = actual_variance.diff().dropna()

# Truncate the simulated variance data to match the actual data length
v_truncated = v[:len(actual_variance), :]
simulated_variance_diff = np.diff(v_truncated, axis=0)

# Define jump threshold as a multiple of standard deviation
jump_threshold_multiplier = 3
actual_jump_threshold = jump_threshold_multiplier * np.std(actual_variance_diff)
simulated_jump_threshold = jump_threshold_multiplier * np.std(simulated_variance_diff, axis=0)

# Identify and count significant jumps in actual and simulated data
actual_jumps = (np.abs(actual_variance_diff) > actual_jump_threshold).astype(int)
actual_jump_count = actual_jumps.sum()
simulated_jumps = (np.abs(simulated_variance_diff) > simulated_jump_threshold).astype(int)
simulated_jump_counts = simulated_jumps.sum(axis=0)

# Visualization of variance dynamics
plt.figure(figsize=(12, 8))
for path in range(min(3, num_paths)):  # Display a few paths for clarity
    plt.plot(v_truncated[:, path], lw=0.5, alpha=0.2, color='blue')

# Plot mean and median of the simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
median_v_truncated = np.median(v_truncated, axis=1)
plt.plot(mean_v_truncated, 'r', label='Mean Simulated Variance', linewidth=2, zorder=3)
plt.plot(median_v_truncated, 'g', label='Median Simulated Variance', linewidth=2, zorder=3)

# Plot actual variance for comparison
plt.plot(actual_variance.values, 'k', label='Actual Variance (Rolling 32)', linewidth=1)
plt.title('Comparison of Actual and Simulated Variance Dynamics Per Epoch')
plt.xlabel('Time (32-slot blocks)')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.show()


# Print the jump analysis results
print(f"Actual number of jumps: {actual_jump_count}")
print(f"Simulated average number of jumps across paths: {np.mean(simulated_jump_counts)}")
print(f"Simulated median number of jumps across paths: {np.median(simulated_jump_counts)}")

# Calculate RMSE between actual variance and mean simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
rmse_variance = np.sqrt(mean_squared_error(actual_variance, mean_v_truncated))
print(f"RMSE between actual and simulated variance: {rmse_variance}")
mae = np.mean(np.abs(actual_variance - mean_v_truncated))
print(f"MAE between actual and simulated variance: {mae}")

#############################################
## Find Initial Parameters Value 64 Slots ##
#############################################
# Define window size
window_size = 64
# Calculate block variance
block_variances_64blocks = [log_returns_train[i:i + window_size].var() for i in
                            range(0, len(log_returns_train), window_size) if i + window_size <= len(log_returns_train)]
# Convert to Series for further analysis
block_variances_64blocks = pd.Series(block_variances_64blocks)


# Estimate initial kappa and sigma
def estimate_kappa_sigma(rolling_variance, dt):
    # Calculate the lagged series
    variance_t = rolling_variance[:-1]
    variance_t1 = rolling_variance[1:]

    # Estimate kappa
    covariance = np.cov(variance_t, variance_t1)[0, 1]
    variance = np.var(variance_t)
    kappa = -np.log(covariance / variance) / dt

    # Estimate sigma
    diff_variance = np.diff(rolling_variance)
    sigma = np.sqrt(np.var(diff_variance) / dt)

    return kappa, sigma


# Find Kappa and Sigma
dt = 1 / (82125 / 2)
initial_kappa, initial_sigma = estimate_kappa_sigma(block_variances_64blocks, dt)
print("Initial kappa is: ", initial_kappa)
print("Initial sigma is: ", initial_sigma)

# Estimate initial Mu
initial_mu = np.mean(log_returns_train)
print("Initial mu is: ", initial_mu)

# Estimate initial theta
initial_theta = np.var(log_returns_train)
print("Initial theta is: ", initial_theta)

#########################################
## Log Likelihood Estimation 64 Slots ##
#########################################
def heston_log_likelihood(params, log_prices):
    mu, kappa, theta, sigma, rho = params
    n = len(log_prices)
    dt = 1 / (82125 / 2)

    # Initialize the variance and log-likelihood
    v = np.var(np.diff(log_prices))  # Initial variance estimate
    log_likelihood = 0

    # Generate correlated Brownian motions
    dW = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n - 1) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, n):
        v_next = max(0, v) + kappa * (theta - max(0, v)) * dt + sigma * np.sqrt(max(0, v)) * dW2[i - 1]
        dY = (mu - 0.5 * max(0, v)) * dt + np.sqrt(max(0, v)) * dW1[i - 1]
        log_likelihood += -0.5 * (np.log(2 * np.pi * max(0.0001, v)) + (log_prices[i - 1] - dY) ** 2 / max(0.0001, v))
        v = v_next

    return -log_likelihood


def optimize_parameters(log_prices):
    # Define bounds and initial guesses
    bounds = [(None, None), (0, None), (0, None), (0, None), (-1, 1)]
    initial_guess = [initial_mu, initial_kappa, initial_theta, initial_sigma, 0]

    # Minimize the negative log likelihood
    result = opt.minimize(heston_log_likelihood, initial_guess, args=(log_prices), bounds=bounds)

    return result.x


optimized_params = optimize_parameters(log_prices_train)
print("Adjusted parameters:", optimized_params)

estimated_mu, estimated_kappa, estimated_theta, estimated_sigma, estimated_rho = optimized_params

#####################################
## Monte Carlo Simulation 64 Slots ##
#####################################
T = 18 / 365  # Total time in years for simulation (18 days)
dt = 1/(82125/2)  # Delta t in years per 2 epochs
num_paths = 1000  # Number of Monte Carlo paths
N = 2024  # Total number of time steps (Per 2 Epoch)

# Initial conditions
S0 = np.median(log_prices_train)  # Start from the median log price in the train dataset
v0 = np.var(log_returns_train)  # Initial variance, estimated from the train log returns

# Arrays to store simulations
S = np.zeros((N, num_paths))
v = np.zeros((N, num_paths))

S[0, :] = S0
v[0, :] = v0

# Generate correlated random shocks for each path
for path in range(num_paths):
    dW = np.random.multivariate_normal([0, 0], [[1, estimated_rho], [estimated_rho, 1]], N) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, N):
        v_prev = v[i - 1, path]
        v_drift = estimated_kappa * (estimated_theta - max(v_prev, 0)) * dt
        v_diffusion = estimated_sigma * np.sqrt(max(v_prev, 0)) * dW2[i - 1]
        v[i, path] = max(v_prev + v_drift + v_diffusion, 0)

        S[i, path] = S[i - 1, path] + (estimated_mu - 0.5 * max(v[i, path], 0)) * dt + np.sqrt(max(v[i, path], 0)) * dW1[i - 1]

print("Shape of v:", v.shape)

# Plotting the results for all the paths
plt.figure(figsize=(12, 6))
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
plt.title('Simulated Log Price Dynamics Over Test Period Per 2 Epoch')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.show()

# Average log prices over fixed intervals
two_epoch_prices = np.array([np.mean(log_prices_test[i:i+64]) for i in range(0, len(log_prices_test), 64)])
total_2epochs = len(two_epoch_prices)

# Mean of all paths
mean_S = np.mean(S, axis=1)
median_S = np.median(S, axis=1)
plt.figure(figsize=(12, 6))

# Time arrays for plotting
simulated_times = np.linspace(0, T, N)
actual_times = np.linspace(0, T, total_2epochs)

# Plot simulated data
plt.plot(simulated_times, mean_S, label='Mean Simulated Log Prices', color='red')
plt.plot(simulated_times, median_S, label='Median Simulated Log Prices', color='purple')
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
# Plot 2 Epochly
plt.plot(actual_times, two_epoch_prices, label='Average Log Prices Per 2 Epochs', color='green')
# Finalizing the plot
plt.title('Comparison of Actual and Simulated Log Prices Per 2 Epochs')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.legend()
plt.show()

# Calculate RMSE & MAE between the actual epoch prices and the simulated mean prices
mean_S = np.mean(S, axis=1)
simulated_epoch_prices = mean_S[:total_2epochs]
# Calculate RMSE between the actual epoch prices and the simulated mean prices
rmse = np.sqrt(mean_squared_error(two_epoch_prices, simulated_epoch_prices))
print(f"RMSE between actual and simulated log prices: {rmse}")
# Calculate MAE between the actual epoch prices and the simulated mean prices
mae = np.mean(np.abs(two_epoch_prices - simulated_epoch_prices))
print(f"MAE between actual and simulated log prices: {mae}")

###################################################
## Variance Dynamics Analysis for 64-Slot Blocks ##
###################################################

# Simulation parameters
T = 18 / 365  # Total simulation time in years (18 days)
num_paths = 1000  # Number of Monte Carlo paths
N = 2024  # Expected number of time steps (18 days, per 2 epochs)

# Define the window size for block variance analysis
window_size = 64

# Calculate block variances using rolling windows of 64 time steps
block_variances_64blocks = [
    log_returns_test[i:i + window_size].var()
    for i in range(0, len(log_returns_test), window_size)
    if i + window_size <= len(log_returns_test)
]

# Convert block variances to a pandas Series for detailed analysis
actual_variance = pd.Series(block_variances_64blocks)
actual_variance_diff = actual_variance.diff().dropna()

# Adjust the simulated variance to match the actual data's length
v_truncated = v[:len(actual_variance), :]
simulated_variance_diff = np.diff(v_truncated, axis=0)

# Set the jump threshold as a multiple of the standard deviation
jump_threshold_multiplier = 3
actual_jump_threshold = jump_threshold_multiplier * np.std(actual_variance_diff)
simulated_jump_threshold = jump_threshold_multiplier * np.std(simulated_variance_diff, axis=0)

# Detect significant jumps in actual and simulated variance differences
actual_jumps = (np.abs(actual_variance_diff) > actual_jump_threshold).astype(int)
actual_jump_count = actual_jumps.sum()
simulated_jumps = (np.abs(simulated_variance_diff) > simulated_jump_threshold).astype(int)
simulated_jump_counts = simulated_jumps.sum(axis=0)

# Plot the dynamics of simulated variance for a selection of paths
plt.figure(figsize=(12, 8))
for path in range(min(5, num_paths)):  # Limit the number of paths plotted for better visibility
    plt.plot(v_truncated[:, path], lw=0.5, alpha=0.2, color='blue')

# Include mean and median lines of the simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
median_v_truncated = np.median(v_truncated, axis=1)
plt.plot(mean_v_truncated, 'r', label='Mean Simulated Variance', linewidth=2, zorder=3)
plt.plot(median_v_truncated, 'g', label='Median Simulated Variance', linewidth=2, zorder=3)

# Plot the actual variance computed over 64 slots
plt.plot(actual_variance.values, 'k', label='Actual Variance (Rolling 64)', linewidth=1)
plt.title('Comparison of Actual and Simulated Variance Dynamics Per 2 Epoch')
plt.xlabel('Time (64-slot blocks)')
plt.ylabel('Variance')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Output the jump analysis results
print(f"Actual number of jumps: {actual_jump_count}")
print(f"Simulated average number of jumps across paths: {np.mean(simulated_jump_counts)}")
print(f"Simulated median number of jumps across paths: {np.median(simulated_jump_counts)}")

# Calculate RMSE between actual variance and mean simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
rmse_variance = np.sqrt(mean_squared_error(actual_variance, mean_v_truncated))
print(f"RMSE between actual and simulated variance: {rmse_variance}")
mae = np.mean(np.abs(actual_variance - mean_v_truncated))
print(f"MAE between actual and simulated variance: {mae}")

#############################################
## Find Initial Parameters Value 150 Slots ##
#############################################
# Define window size
window_size = 150
# Calculate block variance
block_variances_150blocks = [log_returns_train[i:i + window_size].var() for i in
                             range(0, len(log_returns_train), window_size) if i + window_size <= len(log_returns_train)]
# Convert to Series for further analysis
rolling_variance_150slots = pd.Series(block_variances_150blocks)


# Estimate initial kappa and sigma
def estimate_kappa_sigma(rolling_variance, dt):
    # Calculate the lagged series
    variance_t = rolling_variance[:-1]
    variance_t1 = rolling_variance[1:]

    # Estimate kappa
    covariance = np.cov(variance_t, variance_t1)[0, 1]
    variance = np.var(variance_t)
    kappa = -np.log(covariance / variance) / dt

    # Estimate sigma
    diff_variance = np.diff(rolling_variance)
    sigma = np.sqrt(np.var(diff_variance) / dt)

    return kappa, sigma


# Find Kappa and Sigma
dt = 1 / (365 * 24 * 2)
initial_kappa, initial_sigma = estimate_kappa_sigma(rolling_variance_150slots, dt)
print("Initial kappa is: ", initial_kappa)
print("Initial sigma is: ", initial_sigma)

# Estimate initial Mu
initial_mu = np.mean(log_returns_train)
print("Initial mu is: ", initial_mu)

# Estimate initial theta
initial_theta = np.var(log_returns_train)
print("Initial theta is: ", initial_theta)


#########################################
## Log Likelihood Estimation 150 Slots ##
#########################################
def heston_log_likelihood(params, log_prices):
    mu, kappa, theta, sigma, rho = params
    n = len(log_prices)
    dt = 1 / (365 * 24 * 2)

    # Initialize the variance and log-likelihood
    v = np.var(np.diff(log_prices))  # Initial variance estimate
    log_likelihood = 0

    # Generate correlated Brownian motions
    dW = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n - 1) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, n):
        v_next = max(0, v) + kappa * (theta - max(0, v)) * dt + sigma * np.sqrt(max(0, v)) * dW2[i - 1]
        dY = (mu - 0.5 * max(0, v)) * dt + np.sqrt(max(0, v)) * dW1[i - 1]
        log_likelihood += -0.5 * (np.log(2 * np.pi * max(0.0001, v)) + (log_prices[i - 1] - dY) ** 2 / max(0.0001, v))
        v = v_next

    return -log_likelihood


def optimize_parameters(log_prices):
    # Define bounds and initial guesses
    bounds = [(None, None), (0, None), (0, None), (0, None), (-1, 1)]
    initial_guess = [initial_mu, initial_kappa, initial_theta, initial_sigma, 0]

    # Minimize the negative log likelihood
    result = opt.minimize(heston_log_likelihood, initial_guess, args=(log_prices), bounds=bounds)

    return result.x


optimized_params = optimize_parameters(log_prices_train)
print("Adjusted parameters:", optimized_params)

estimated_mu, estimated_kappa, estimated_theta, estimated_sigma, estimated_rho = optimized_params

######################################
## Monte Carlo Simulation 150 Slots ##
######################################
T = 18 / 365  # Total time in years for simulation (18 days)
dt = 1 / (365 * 24 * 2)  # Delta t in years per hour
num_paths = 1000  # Number of Monte Carlo paths
N = 18 * 24 * 2  # Total number of time steps (18 days, Per Half Hour)

# Initial conditions
S0 = np.median(log_prices_train)  # Start from the median log price in the train dataset
v0 = np.var(log_returns_train)  # Initial variance, estimated from the train log returns

# Arrays to store simulations
S = np.zeros((N, num_paths))
v = np.zeros((N, num_paths))

S[0, :] = S0
v[0, :] = v0

# Generate correlated random shocks for each path
for path in range(num_paths):
    dW = np.random.multivariate_normal([0, 0], [[1, estimated_rho], [estimated_rho, 1]], N) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, N):
        v_prev = v[i - 1, path]
        v_drift = estimated_kappa * (estimated_theta - max(v_prev, 0)) * dt
        v_diffusion = estimated_sigma * np.sqrt(max(v_prev, 0)) * dW2[i - 1]
        v[i, path] = max(v_prev + v_drift + v_diffusion, 0)

        S[i, path] = S[i - 1, path] + (estimated_mu - 0.5 * max(v[i, path], 0)) * dt + np.sqrt(max(v[i, path], 0)) * \
                     dW1[i - 1]

print("Shape of v:", v.shape)

# Plotting the results for all the paths
plt.figure(figsize=(12, 6))
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
plt.title('Simulated Log Price Dynamics Over Test Period Per Half Hour')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Assuming log_prices_test is organized in a flat array format
halfhourly_prices = np.array([np.mean(log_prices_test[i:i + 150]) for i in range(0, len(log_prices_test), 150)])
total_halfhours = len(halfhourly_prices)
# Mean of all paths
mean_S = np.mean(S, axis=1)
median_S = np.median(S, axis=1)
plt.figure(figsize=(12, 6))
# Time arrays for plotting
simulated_times = np.linspace(0, T, N)
actual_times = np.linspace(0, T, total_halfhours)
# Plot simulated data
plt.plot(simulated_times, mean_S, label='Mean Simulated Log Prices', color='red')
plt.plot(simulated_times, median_S, label='Median Simulated Log Prices', color='purple')
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
# Plot half hour prices
plt.plot(actual_times, halfhourly_prices, label='Average Log Prices Per Half Hour', color='green')
# Finalizing the plot
plt.title('Comparison of Actual and Simulated Log Prices Per Half Hour')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

###################################################
## Variance Dynamics Analysis for 150-Slot Blocks ##
###################################################

# Simulation parameters
T = 18 / 365  # Total simulation time in years (18 days)
num_paths = 1000  # Number of Monte Carlo paths
N = 2024  # Expected number of time steps (18 days, per 2 epochs)

# Define the window size for block variance analysis
window_size = 150

# Calculate block variances using rolling windows of 150 time steps
block_variances_150blocks = [
    log_returns_test[i:i + window_size].var()
    for i in range(0, len(log_returns_test), window_size)
    if i + window_size <= len(log_returns_test)
]

# Convert block variances to a pandas Series for detailed analysis
actual_variance = pd.Series(block_variances_150blocks)
actual_variance_diff = actual_variance.diff().dropna()

# Adjust the simulated variance to match the actual data's length
v_truncated = v[:len(actual_variance), :]
simulated_variance_diff = np.diff(v_truncated, axis=0)

# Set the jump threshold as a multiple of the standard deviation
jump_threshold_multiplier = 2.5
actual_jump_threshold = jump_threshold_multiplier * np.std(actual_variance_diff)
simulated_jump_threshold = jump_threshold_multiplier * np.std(simulated_variance_diff, axis=0)

# Detect significant jumps in actual and simulated variance differences
actual_jumps = (np.abs(actual_variance_diff) > actual_jump_threshold).astype(int)
actual_jump_count = actual_jumps.sum()
simulated_jumps = (np.abs(simulated_variance_diff) > simulated_jump_threshold).astype(int)
simulated_jump_counts = simulated_jumps.sum(axis=0)

# Plot the dynamics of simulated variance for a selection of paths
plt.figure(figsize=(12, 8))
for path in range(min(5, num_paths)):  # Limit the number of paths plotted for better visibility
    plt.plot(v_truncated[:, path], lw=0.5, alpha=0.2, color='blue')

# Include mean and median lines of the simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
median_v_truncated = np.median(v_truncated, axis=1)
plt.plot(mean_v_truncated, 'r', label='Mean Simulated Variance', linewidth=2, zorder=3)
plt.plot(median_v_truncated, 'g', label='Median Simulated Variance', linewidth=2, zorder=3)

# Plot the actual variance computed over 150 slots
plt.plot(actual_variance.values, 'k', label='Actual Variance (Rolling 150)', linewidth=1)
plt.title('Comparison of Actual and Simulated Variance Dynamics Per Half Hour')
plt.xlabel('Time (150-slot blocks)')
plt.ylabel('Variance')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Output the jump analysis results
print(f"Actual number of jumps: {actual_jump_count}")
print(f"Simulated average number of jumps across paths: {np.mean(simulated_jump_counts)}")
print(f"Simulated median number of jumps across paths: {np.median(simulated_jump_counts)}")

# Calculate RMSE between actual variance and mean simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
rmse_variance = np.sqrt(mean_squared_error(actual_variance, mean_v_truncated))
print(f"RMSE between actual and simulated variance: {rmse_variance}")
mae = np.mean(np.abs(actual_variance - mean_v_truncated))
print(f"MAE between actual and simulated variance: {mae}")

#############################################
## Find Initial Parameters Value 300 Slots ##
#############################################
# Define window size
window_size = 300
# Calculate block variance
block_variances_300blocks = [log_returns_train[i:i + window_size].var() for i in
                             range(0, len(log_returns_train), window_size) if i + window_size <= len(log_returns_train)]
# Convert to Series for further analysis
rolling_variance_300slots = pd.Series(block_variances_300blocks)


# Estimate initial kappa and sigma
def estimate_kappa_sigma(rolling_variance, dt):
    # Calculate the lagged series
    variance_t = rolling_variance[:-1]
    variance_t1 = rolling_variance[1:]

    # Estimate kappa
    covariance = np.cov(variance_t, variance_t1)[0, 1]
    variance = np.var(variance_t)
    kappa = -np.log(covariance / variance) / dt

    # Estimate sigma
    diff_variance = np.diff(rolling_variance)
    sigma = np.sqrt(np.var(diff_variance) / dt)

    return kappa, sigma


# Find Kappa and Sigma
dt = 1 / (365 * 24)
initial_kappa, initial_sigma = estimate_kappa_sigma(rolling_variance_300slots, dt)
print("Initial kappa is: ", initial_kappa)
print("Initial sigma is: ", initial_sigma)

# Estimate initial Mu
initial_mu = np.mean(log_returns_train)
print("Initial mu is: ", initial_mu)

# Estimate initial theta
initial_theta = np.var(log_returns_train)
print("Initial theta is: ", initial_theta)


#########################################
## Log Likelihood Estimation 300 Slots ##
#########################################
def heston_log_likelihood(params, log_prices):
    mu, kappa, theta, sigma, rho = params
    n = len(log_prices)
    dt = 1 / (365 * 24)

    # Initialize the variance and log-likelihood
    v = np.var(np.diff(log_prices))  # Initial variance estimate
    log_likelihood = 0

    # Generate correlated Brownian motions
    dW = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n - 1) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, n):
        v_next = max(0, v) + kappa * (theta - max(0, v)) * dt + sigma * np.sqrt(max(0, v)) * dW2[i - 1]
        dY = (mu - 0.5 * max(0, v)) * dt + np.sqrt(max(0, v)) * dW1[i - 1]
        log_likelihood += -0.5 * (np.log(2 * np.pi * max(0.0001, v)) + (log_prices[i - 1] - dY) ** 2 / max(0.0001, v))
        v = v_next

    return -log_likelihood


def optimize_parameters(log_prices):
    # Define bounds and initial guesses
    bounds = [(None, None), (0, None), (0, None), (0, None), (-1, 1)]
    initial_guess = [initial_mu, initial_kappa, initial_theta, initial_sigma, 0]

    # Minimize the negative log likelihood
    result = opt.minimize(heston_log_likelihood, initial_guess, args=(log_prices), bounds=bounds)

    return result.x


optimized_params = optimize_parameters(log_prices_train)
print("Adjusted parameters:", optimized_params)

estimated_mu, estimated_kappa, estimated_theta, estimated_sigma, estimated_rho = optimized_params

######################################
## Monte Carlo Simulation 300 Slots ##
######################################
T = 18 / 365  # Total time in years for simulation (18 days)
dt = 1 / (365 * 24)  # Delta t in years per hour
num_paths = 1000  # Number of Monte Carlo paths
N = 18 * 24  # Total number of time steps (18 days, hourly)

# Initial conditions
S0 = np.median(log_prices_train)  # Start from the median log price in the train dataset
v0 = np.var(log_returns_train)  # Initial variance, estimated from the train log returns

# Arrays to store simulations
S = np.zeros((N, num_paths))
v = np.zeros((N, num_paths))

S[0, :] = S0
v[0, :] = v0

# Generate correlated random shocks for each path
for path in range(num_paths):
    dW = np.random.multivariate_normal([0, 0], [[1, estimated_rho], [estimated_rho, 1]], N) * np.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]

    for i in range(1, N):
        v_prev = v[i - 1, path]
        v_drift = estimated_kappa * (estimated_theta - max(v_prev, 0)) * dt
        v_diffusion = estimated_sigma * np.sqrt(max(v_prev, 0)) * dW2[i - 1]
        v[i, path] = max(v_prev + v_drift + v_diffusion, 0)

        S[i, path] = S[i - 1, path] + (estimated_mu - 0.5 * max(v[i, path], 0)) * dt + np.sqrt(max(v[i, path], 0)) * dW1[i - 1]

print("Shape of v:", v.shape)

# Plotting the results for all the paths
plt.figure(figsize=(12, 6))
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
plt.title('Simulated Log Price Dynamics Over Test Period Per Hour')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.show()

# Average log prices over fixed intervals
hourly_prices = np.array([np.mean(log_prices_test[i:i+300]) for i in range(0, len(log_prices_test), 300)])
total_hours = len(hourly_prices)

# Mean of all paths
mean_S = np.mean(S, axis=1)
median_S = np.median(S, axis=1)
plt.figure(figsize=(12, 6))

# Time arrays for plotting
simulated_times = np.linspace(0, T, N)
actual_times = np.linspace(0, T, total_hours)

# Plot simulated data
plt.plot(simulated_times, mean_S, label='Mean Simulated Log Prices', color='red')
plt.plot(simulated_times, median_S, label='Median Simulated Log Prices', color='purple')
for path in range(S.shape[1]):  # S.shape[1] gives the number of paths
    plt.plot(np.linspace(0, T, N), S[:, path], lw=0.5, alpha=0.3, color='blue')
# Plot hourly prices
plt.plot(actual_times, hourly_prices, label='Average Log Prices Per Hour', color='green')
# Finalizing the plot
plt.title('Comparison of Actual and Simulated Log Prices Per Hour')
plt.xlabel('Time (years)')
plt.ylabel('Log Price')
plt.grid(True)
plt.legend()
plt.show()

# Calculate RMSE & MAE between the actual epoch prices and the simulated mean prices
mean_S = np.mean(S, axis=1)
simulated_epoch_prices = mean_S[:total_hours]
# Calculate RMSE between the actual epoch prices and the simulated mean prices
rmse = np.sqrt(mean_squared_error(hourly_prices, simulated_epoch_prices))
print(f"RMSE between actual and simulated log prices: {rmse}")
# Calculate MAE between the actual epoch prices and the simulated mean prices
mae = np.mean(np.abs(hourly_prices - simulated_epoch_prices))
print(f"MAE between actual and simulated log prices: {mae}")

###################################################
## Variance Dynamics Analysis for 300-Slot Blocks ##
###################################################

# Simulation parameters
T = 18 / 365  # Total simulation time in years (18 days)
num_paths = 1000  # Number of Monte Carlo paths
N = 2024  # Expected number of time steps (18 days, per 2 epochs)

# Define the window size for block variance analysis
window_size = 300

# Calculate block variances using rolling windows of 300 time steps
block_variances_300blocks = [
    log_returns_test[i:i + window_size].var()
    for i in range(0, len(log_returns_test), window_size)
    if i + window_size <= len(log_returns_test)
]

# Convert block variances to a pandas Series for detailed analysis
actual_variance = pd.Series(block_variances_300blocks)
actual_variance_diff = actual_variance.diff().dropna()

# Adjust the simulated variance to match the actual data's length
v_truncated = v[:len(actual_variance), :]
simulated_variance_diff = np.diff(v_truncated, axis=0)

# Set the jump threshold as a multiple of the standard deviation
jump_threshold_multiplier = 2.5
actual_jump_threshold = jump_threshold_multiplier * np.std(actual_variance_diff)
simulated_jump_threshold = jump_threshold_multiplier * np.std(simulated_variance_diff, axis=0)

# Detect significant jumps in actual and simulated variance differences
actual_jumps = (np.abs(actual_variance_diff) > actual_jump_threshold).astype(int)
actual_jump_count = actual_jumps.sum()
simulated_jumps = (np.abs(simulated_variance_diff) > simulated_jump_threshold).astype(int)
simulated_jump_counts = simulated_jumps.sum(axis=0)

# Plot the dynamics of simulated variance for a selection of paths
plt.figure(figsize=(12, 8))
for path in range(min(5, num_paths)):  # Limit the number of paths plotted for better visibility
    plt.plot(v_truncated[:, path], lw=0.5, alpha=0.2, color='blue')

# Include mean and median lines of the simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
median_v_truncated = np.median(v_truncated, axis=1)
plt.plot(mean_v_truncated, 'r', label='Mean Simulated Variance', linewidth=2, zorder=3)
plt.plot(median_v_truncated, 'g', label='Median Simulated Variance', linewidth=2, zorder=3)

# Plot the actual variance computed over 300 slots
plt.plot(actual_variance.values, 'k', label='Actual Variance (Rolling 300)', linewidth=1)
plt.title('Comparison of Actual and Simulated Variance Dynamics Per Hour')
plt.xlabel('Time (300-slot blocks)')
plt.ylabel('Variance')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Output the jump analysis results
print(f"Actual number of jumps: {actual_jump_count}")
print(f"Simulated average number of jumps across paths: {np.mean(simulated_jump_counts)}")
print(f"Simulated median number of jumps across paths: {np.median(simulated_jump_counts)}")

# Calculate RMSE between actual variance and mean simulated variance
mean_v_truncated = np.mean(v_truncated, axis=1)
rmse_variance = np.sqrt(mean_squared_error(actual_variance, mean_v_truncated))
print(f"RMSE between actual and simulated variance: {rmse_variance}")
mae = np.mean(np.abs(actual_variance - mean_v_truncated))
print(f"MAE between actual and simulated variance: {mae}")
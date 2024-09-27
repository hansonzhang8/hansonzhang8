#############################
## Load necessary packages ##
#############################
import pandas as pd
import numpy as np
import os

####################
## Importing Data ##
####################
# Construct the file path based on your folder structure and username
file_path = r'/Users/hansonzhang/Desktop/mev_research/mev_data/mev_2023_q1_cleaned.csv'
# Load the CSV file into a pandas DataFrame
mev_2023_q1_cleaned = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(mev_2023_q1_cleaned)
# Display the columns of the dataset
print(mev_2023_q1_cleaned.columns)

###################
## Define Return ##
###################

# Calculate simple return
mev_2023_q1_cleaned['simple_return'] = mev_2023_q1_cleaned['proposer_total_reward_in_eth'].pct_change()

# Calculate log return
mev_2023_q1_cleaned['log_return'] = np.log(mev_2023_q1_cleaned['proposer_total_reward_in_eth']).diff()

# Drop the first and last rows
mev_2023_q1_cleaned = mev_2023_q1_cleaned.iloc[1:-1]

# Check for infinite values
infinite_values = mev_2023_q1_cleaned[['simple_return', 'log_return']].replace([np.inf, -np.inf], np.nan).isnull().sum()

# Check for zero values
zero_values = (mev_2023_q1_cleaned[['simple_return', 'log_return']] == 0).sum()

# Check for null values
null_values = mev_2023_q1_cleaned[['simple_return', 'log_return']].isnull().sum()

# Display the results
print("Number of infinite returns: ",infinite_values)
print("Number of zero returns: ",zero_values)
print("Number of null returns: ",null_values)

# Save cleaned-file
mev_2023_q1_cleaned.to_csv('mev_2023_q1_cleaned.csv', index=False)
# Path to the original data file
original_file_path = '/Users/hansonzhang/Desktop/MEV Research/MEV Data/2023_Q1_Data.csv'
# Extract the directory path from the original file path
save_directory = os.path.dirname(original_file_path)
save_path = os.path.join(save_directory, 'mev_2023_q1_cleaned.csv')
# Save the cleaned DataFrame to a CSV file in the same directory
mev_2023_q1_cleaned.to_csv(save_path, index=False)
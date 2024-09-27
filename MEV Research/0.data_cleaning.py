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
file_path = r'/Users/hansonzhang/Desktop/mev_research/mev_data/2023_Q1_Data.csv'
# Load the CSV file into a pandas DataFrame
mev_2023_q1 = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(mev_2023_q1)
print(mev_2023_q1.columns)


#############################
## Check for missing block ##
#############################

def check_missing_blocks(data):
    # Ensure the data is sorted by 'block_number'
    sorted_data = data.sort_values(by=['block_number'], ascending=True)

    # Create a DataFrame to check for missing blocks
    check = pd.DataFrame({'block_number': sorted_data['block_number'],
                          'interval': sorted_data['block_number'].shift(-1) - sorted_data['block_number']})

    # Identify missing blocks where the interval is not equal to 1
    missing_blocks = check[check['interval'] != 1]

    return missing_blocks


missing_blocks = check_missing_blocks(mev_2023_q1)

# Display the missing blocks
print("Missing blocks:")
print(missing_blocks)
# Block Number 16444583 has two same lines
# Block 16889384, 16898786, 16931417, 16931918 are missing;

# Remove the duplicate line
mev_2023_q1 = mev_2023_q1.drop_duplicates(subset=['block_number'], keep='first')

# Re-check the dataset
missing_blocks = check_missing_blocks(mev_2023_q1)
# Display the missing blocks
print("Missing blocks:")
print(missing_blocks)

#################################
## Check for missing timestamp ##
#################################

# Count the number of 0 'block_timestamp' in 'block_timestamp'
missing_timestamp = mev_2023_q1[mev_2023_q1['block_timestamp'] == 0]
print("Rows with 'block_timestamp' == 0:", len(missing_timestamp))
# There is no row with 0 Timestamp

# Count the number of missing (NaN) values in 'block_timestamp'
na_counts = (mev_2023_q1['block_timestamp'].isna()).sum()
print("Count of missing (NaN) values in 'block_timestamp':", na_counts)
# # There is no row with NaN Timestamp

###########################
## Check for 0 MEV block ##
###########################
# Filter rows where 'proposer_total_reward_in_eth' is 0
proposer_reward_zero = mev_2023_q1[mev_2023_q1['proposer_total_reward_in_eth'] == 0]
num_proposer_reward_zero = len(proposer_reward_zero)
print("Number of rows with 'proposer_total_reward_in_eth' == 0:", num_proposer_reward_zero)

# Check if 'header_fee_recipient_balance_change_in_eth' is also 0 for these rows
header_fee_zero = proposer_reward_zero[proposer_reward_zero['header_fee_recipient_balance_change_in_eth'] == 0]
num_header_fee_zero = len(header_fee_zero)
print("Number of rows with 'proposer_total_reward_in_eth' == 0 and 'header_fee_recipient_balance_change_in_eth' == 0:", num_header_fee_zero)

# After we double-checked with Etherscan, we realized these blocks have no transactions. So we will drop them.

# Check for rows with negative 'proposer_total_reward_in_eth'
negative_reward = mev_2023_q1[mev_2023_q1['proposer_total_reward_in_eth'] < 0]
num_negative_reward = len(negative_reward)
print("Number of rows with negative 'proposer_total_reward_in_eth':", num_negative_reward)
print("Rows with negative 'proposer_total_reward_in_eth':")
print(negative_reward)

# Check for rows with infinite 'proposer_total_reward_in_eth'
infinite_reward = mev_2023_q1[~np.isfinite(mev_2023_q1['proposer_total_reward_in_eth'])]
num_infinite_reward = len(infinite_reward)
print("Number of rows with infinite 'proposer_total_reward_in_eth':", num_infinite_reward)

# Drop negative & 0 MEV blocks
mev_2023_q1_cleaned = mev_2023_q1[mev_2023_q1['proposer_total_reward_in_eth'] > 0]

# Verify the rows are removed
print("Data after removing rows with 0 or negative 'proposer_total_reward_in_eth':")
print(mev_2023_q1_cleaned)

# Save cleaned-file
mev_2023_q1_cleaned.to_csv('mev_2023_q1_cleaned.csv', index=False)
# Path to the original data file
original_file_path = '/Users/hansonzhang/Desktop/MEV Research/MEV Data/2023_Q1_Data.csv'
# Extract the directory path from the original file path
save_directory = os.path.dirname(original_file_path)
save_path = os.path.join(save_directory, 'mev_2023_q1_cleaned.csv')
# Save the cleaned DataFrame to a CSV file in the same directory
mev_2023_q1_cleaned.to_csv(save_path, index=False)
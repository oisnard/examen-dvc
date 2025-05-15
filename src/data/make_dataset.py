import pandas as pd 
from sklearn.model_selection import train_test_split 
import os 

input_file = 'data/raw_data/raw.csv'
output_dir = 'data/processed_data/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Check if the directory exists, if not create it
if os.path.isfile('data/raw_data/raw.csv'):
    data = pd.read_csv('data/raw_data/raw.csv', index_col=0)
else:
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    data = pd.read_csv(url, index_col=0)
    data.to_csv('data/raw_data/raw.csv')

# Extract the features and target variable
# The target variable is 'silica_concentrate' and the features are all other columns
data_features = data.drop(columns=['silica_concentrate'])
data_target = data['silica_concentrate']

# Define the ratio for the train-test split
# 70% of the data will be used for training and 30% for testing
RATIO_TEST = 0.3


# Split the data into train and test sets   
X_train, X_test, y_train, y_test = train_test_split(data_features, 
                                                    data_target, 
                                                    test_size=RATIO_TEST, 
                                                    shuffle=True,
                                                    random_state=42)

# Save the train and test sets to CSV files 
X_train.to_csv('data/processed_data/X_train.csv')
y_train.to_csv('data/processed_data/y_train.csv')
X_test.to_csv('data/processed_data/X_test.csv')
y_test.to_csv('data/processed_data/y_test.csv')




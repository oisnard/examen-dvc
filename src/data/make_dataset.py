import pandas as pd 
from sklearn.model_selection import train_test_split 
import os 
import yaml 

input_file = 'data/raw_data/raw.csv'
input_dir = 'data/raw_data/'
output_dir = 'data/processed_data/'

# Create the input directory if it doesn't exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Check if the directory exists, if not create it
if os.path.isfile(input_file):
    data = pd.read_csv(input_file, index_col=0)
else:
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    data = pd.read_csv(url, index_col=0)
    data.to_csv('data/raw_data/raw.csv')

# Extract the features and target variable
# The target variable is 'silica_concentrate' and the features are all other columns
data_features = data.drop(columns=['silica_concentrate'])
data_target = data['silica_concentrate']

# Define the ratio for the train-test split and random state
with open('params.yaml') as file:
    params = yaml.safe_load(file)
    RATIO_TEST = params['split']['test_size']
    RANDOM_STATE = params['split']['random_state']
# If the ratio is not defined in the params.yaml file, use a default value
if RATIO_TEST is None:
    # Default value for the test size
    # 30% of the data will be used for testing and 70% for training
    RATIO_TEST = 0.3
# If the random state is not defined in the params.yaml file, use a default value
if RANDOM_STATE is None:
    # Default value for the random state
    RANDOM_STATE = 42



# Split the data into train and test sets   
X_train, X_test, y_train, y_test = train_test_split(data_features, 
                                                    data_target, 
                                                    test_size=RATIO_TEST, 
                                                    shuffle=True,
                                                    random_state=RANDOM_STATE)

# Save the train and test sets to CSV files 
X_train.to_csv('data/processed_data/X_train.csv')
y_train.to_csv('data/processed_data/y_train.csv')
X_test.to_csv('data/processed_data/X_test.csv')
y_test.to_csv('data/processed_data/y_test.csv')




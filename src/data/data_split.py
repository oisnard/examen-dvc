import pandas as pd 
from sklearn.model_selection import train_test_split 

data = pd.read_csv('data/raw_data/raw.csv', index_col=0)

data_features = data.drop(columns=['silica_concentrate'])
data_target = data['silica_concentrate']
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




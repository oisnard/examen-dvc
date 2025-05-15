import pandas as pd 
from sklearn.preprocessing import StandardScaler 

# Get the data
X_train = pd.read_csv('data/processed_data/X_train.csv', index_col=0)
X_test = pd.read_csv('data/processed_data/X_test.csv', index_col=0) 

# Scaling the features
# Initialize the StandardScaler 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaled data to CSV files
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)  
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv')
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv') 


import pandas as pd 
import joblib
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Load the training data    
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('data/processed_data/y_train.csv', index_col=0) 


# Load the best parameters
with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)


# Define the model with the best parameters 
model = ElasticNet(**best_params)
# Fit the model to the training data
model.fit(X_train_scaled, y_train.values.ravel())
# Save the model to a file
joblib.dump(model, 'models/model.pkl')

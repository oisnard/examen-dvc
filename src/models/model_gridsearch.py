import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle 
import yaml

# Load the training data
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('data/processed_data/y_train.csv', index_col=0)
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv', index_col=0)
y_test = pd.read_csv('data/processed_data/y_test.csv', index_col=0)

# Define the model
model = RandomForestRegressor()

# Define the hyperparameters and their values to be tested
with open('params.yaml') as file:
    params = yaml.safe_load(file)
param_grid = params['gridsearch']['param_grid']
# If the parameters are not defined in the params.yaml file, use default values
if param_grid is None:
    # Default value for the parameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

# Define the scoring method
SCORING = params['gridsearch']['scoring']
if SCORING is None:
    # Default value for the scoring
    SCORING = 'neg_mean_squared_error'

# Define the number of folds for cross-validation
CV = params['gridsearch']['cv']
# If the number of folds is not defined in the params.yaml file, use a default value
if CV is None:
    # Default value for the number of folds
    CV = 5


# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=model,
                             param_grid=param_grid, 
                             scoring=SCORING, 
                             cv=CV, 
                             n_jobs=-1,
                             verbose=2)



# Fit the model to the training data
grid_search.fit(X_train_scaled, y_train.values.ravel())

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
# Print the best parameters and the best score
print("Best parameters: ", best_params)
print("Best score: ", best_score)

# Save the best parameters and the best score to a CSV file
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

   
import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle 

# Load the training data
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('data/processed_data/y_train.csv', index_col=0)
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv', index_col=0)
y_test = pd.read_csv('data/processed_data/y_test.csv', index_col=0)
# Define the model
base_models = [
    ('ridge', Ridge()),
    ('rf', RandomForestRegressor()),
    ('gbr', GradientBoostingRegressor())
]

final_estimator = LinearRegression()
model = StackingRegressor(estimators=base_models, final_estimator=final_estimator, passthrough=True, cv=5)

# Define the hyperparameters and their values to be tested      
param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0],
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20],
    'gbr__n_estimators': [50, 100, 200],
    'gbr__learning_rate': [0.01, 0.1, 1.0],
    'gbr__max_depth': [3, 5, 7],
    'gbr__min_samples_split': [2, 5, 10],
    'gbr__min_samples_leaf': [1, 2, 4],
    'final_estimator__fit_intercept': [True, False]
    }
# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=model, 
                             param_grid=param_grid, 
                             scoring='neg_mean_squared_error', 
                             cv=5, 
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

    
#best_params_df = pd.DataFrame([best_params])
#best_params_df.to_csv('models/best_params.csv', index=False)

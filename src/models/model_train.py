import pandas as pd 
import joblib
import pickle
from sklearn.linear_model import Ridge, LinearRegression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error

# Load the training data    
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('data/processed_data/y_train.csv', index_col=0) 


# Load the best parameters
with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

rf_params = {
    'n_estimators': best_params['rf__n_estimators'],
    'max_depth': best_params['rf__max_depth'],
}

gbr_params = {
    'n_estimators': best_params['gbr__n_estimators'],
    'learning_rate': best_params['gbr__learning_rate'],
    'max_depth': best_params['gbr__max_depth'],
    'min_samples_split': best_params['gbr__min_samples_split'],
    'min_samples_leaf': best_params['gbr__min_samples_leaf']
}

ridge_params = {
    'alpha': best_params['ridge__alpha']
}

estimators = [
    ('ridge', Ridge(**ridge_params)),
    ('rf', RandomForestRegressor(**rf_params)),
    ('gbr', GradientBoostingRegressor(**gbr_params))
]
final_estimator = LinearRegression(fit_intercept=best_params['final_estimator__fit_intercept'])

model = StackingRegressor(estimators=estimators,
                          final_estimator=final_estimator, 
                          passthrough=True, 
                          cv=5)

# Fit the model to the training data
model.fit(X_train_scaled, y_train.values.ravel())
# Save the model to a file
joblib.dump(model, 'models/model.pkl')

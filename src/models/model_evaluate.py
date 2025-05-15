import pandas as pd 
import joblib 
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
 
# Load the test data
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv', index_col=0)
y_test = pd.read_csv('data/processed_data/y_test.csv', index_col=0)
# Load the model
model = joblib.load('models/model.pkl')
# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Calculate the metrics
results = {}
results["Mean Squared Error"] = mean_squared_error(y_test, y_pred)
results["Mean Absolute Error"] = mean_absolute_error(y_test, y_pred)
results["R2 Score"] = r2_score(y_test, y_pred)
results["Mean Absolute Percentage Error"] = mean_absolute_percentage_error(y_test, y_pred)

# Export the results to a JSON file
with open('metrics/scores.json', 'w') as f:
    json.dump(results, f, indent=4)

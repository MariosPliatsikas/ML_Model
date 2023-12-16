import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# Load the data from the CSV file
data = pd.read_csv('cycling_data.csv')

# Select features (X) and target variable (y)
X = data[['Distance', 'CyclingTime']]  # Updated column names
y = data['Speed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = metrics.mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the model
joblib.dump(model, 'speed_prediction_model.joblib')


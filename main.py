# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('Chennai.csv')

# Data Preprocessing
# Dropping unnecessary columns (like TV, DiningTable, etc., which might not be very relevant)
data = data.drop(['TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator', 'BED'], axis=1)

# Handling missing values (if any)
data = data.dropna()

# Encoding categorical data (Location)
data = pd.get_dummies(data, columns=['Location'], drop_first=True)

# Splitting the data into features (X) and target (y)
X = data.drop('Price', axis=1)  # Features
y = data['Price']  # Target variable (Price)

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting house prices for the test set
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Example: Predicting price for a single test case
sample_data = X_test.iloc[0].values.reshape(1, -1)
predicted_price = model.predict(sample_data)
print(f"Predicted Price for the sample data: {predicted_price[0]}")

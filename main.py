import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('Chennai.csv')

data = data.drop(['TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator', 'BED'], axis=1)
data = data.dropna()

data = pd.get_dummies(data, columns=['Location'], drop_first=True)

X = data.drop('Price', axis=1)  
y = data['Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

sample_data = X_test.iloc[0].values.reshape(1, -1)
predicted_price = model.predict(sample_data)
print(f"Predicted Price for the sample data: {predicted_price[0]}")

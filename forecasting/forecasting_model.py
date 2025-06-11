import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt  # for RMSE calculation

# Load dataset
df = pd.read_csv("data/sales_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

X = df[['store_id', 'product_id', 'month', 'day', 'weekday']]
y = df['sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Save model
os.makedirs("forecasting/models", exist_ok=True)
joblib.dump(model, "forecasting/models/xgb_forecast_model.pkl")

# Visualization
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("XGBoost Forecasting Performance")
plt.grid(True)
plt.show()

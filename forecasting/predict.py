import joblib
import numpy as np

# Load the saved model
model = joblib.load("forecasting/models/xgb_forecast_model.pkl")

def predict_sales(store_id, product_id, month, day, weekday):
    """
    Predict sales given input features.
    
    Parameters:
    - store_id (int): Store identifier
    - product_id (int): Product identifier
    - month (int): Month of the date (1-12)
    - day (int): Day of the month (1-31)
    - weekday (int): Day of the week (0=Monday, 6=Sunday)
    
    Returns:
    - float: Predicted sales quantity
    """
    features = np.array([[store_id, product_id, month, day, weekday]])
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    # Example usage: predict sales for store 1, product 101, Jan 4, Thursday (weekday=3)
    result = predict_sales(1, 101, 1, 4, 3)
    print(f"Predicted sales: {result:.2f}")

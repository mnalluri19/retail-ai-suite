from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("forecasting/models/xgb_forecast_model.pkl")

# Input schema using Pydantic
class SalesInput(BaseModel):
    store_id: int
    product_id: int
    month: int
    day: int
    weekday: int

# Define prediction endpoint
@app.post("/predict")
def predict_sales(data: SalesInput):
    features = np.array([[data.store_id, data.product_id, data.month, data.day, data.weekday]])
    prediction = model.predict(features)
    return {"predicted_sales": float(prediction[0])}

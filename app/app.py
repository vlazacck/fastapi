from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

# Define the input data model with meaningful feature names
class InputFeatures(BaseModel):
    customers: float
    store_type: float
    competition_distance: float
    promo_indicator: float
    day_of_week: float
    holiday_indicator: float

# Load the trained model
model = joblib.load("random_forest_model.pkl")  # Replace with your actual path

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: InputFeatures):
    try:
        # Prepare input data for prediction
        input_data = np.array([[data.customers, data.store_type, data.competition_distance, 
                                data.promo_indicator, data.day_of_week, data.holiday_indicator]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]  # Probability for class 1

        return {"prediction": int(prediction[0]), "probability": float(probability[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred during prediction: {str(e)}")
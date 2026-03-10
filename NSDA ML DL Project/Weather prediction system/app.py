from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# create app
app = FastAPI(title="Weather Prediction API")

# load model and scaler
model = joblib.load("weather_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
EXPECTED_FEATURES = int(getattr(scaler, "n_features_in_", 0))


# input schema
class WeatherInput(BaseModel):
    data: list[float]


# home route
@app.get("/")
def home():
    return {"message": "Weather Prediction API is running"}


# prediction route
@app.post("/predict")
def predict(input_data: WeatherInput):
    if EXPECTED_FEATURES and len(input_data.data) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid input length: expected {EXPECTED_FEATURES} feature values, "
                f"got {len(input_data.data)}"
            ),
        )

    # convert list to numpy array
    features = np.array(input_data.data).reshape(1, -1)

    # apply scaling
    features_scaled = scaler.transform(features)

    # prediction
    prediction = model.predict(features_scaled)

    return {
        "prediction": int(prediction[0])
    }
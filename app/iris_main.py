import joblib
import pandas as pd

from fastapi import FastAPI
from app.iris_schema import IrisInput

# Load trained model
model = joblib.load("app/iris_model.pkl")

app = FastAPI(
    title="Iris Prediction API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Iris Prediction API Running"
    }


@app.post("/predict")
def predict(request: IrisInput):
    input_df = pd.DataFrame([request.dict()])

    prediction = model.predict(input_df)[0]

    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return {
        "prediction": species[prediction]
    }
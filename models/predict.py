import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("models/wine_model.pkl")

app = FastAPI(
    title="Wine Prediction API",
    version="1.0.0"
)


class WineRequest(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.get("/")
def home():
    return {
        "message": "Wine Prediction API Running"
    }


@app.post("/predict")
def predict(request: WineRequest):
    input_df = pd.DataFrame([request.dict()])

    input_df.rename(
        columns={
            "od280_od315_of_diluted_wines": "od280/od315_of_diluted_wines"
        },
        inplace=True,
    )

    print("\nColumns received:")
    print(input_df.columns.tolist())

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction)
    }


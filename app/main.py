import os

print("========== EXPERIMENT 12 MAIN LOADED ==========")
print("Loaded from:", __file__)
print("Current Working Directory:", os.getcwd())

from fastapi import FastAPI, HTTPException
import pandas as pd

from app.model import load_model, train_model
from app.validation import validate_dataset, validate_quality
from app.schemas import PredictionRequest

app = FastAPI(
    title="Advanced Testing Pipeline API",
    version="2.0.0"
)

# Load model (train if model doesn't exist)
try:
    model = load_model()
except Exception:
    model, metrics = train_model()

    df = pd.read_csv("Data/breast_cancer.csv")
    validate_dataset(df)
    validate_quality(metrics)

    model = load_model()


@app.get("/")
def home():
    return {
        "message": "Advanced Testing Pipeline is running"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }

@app.get("/deploy")
def deploy():
    return {
        "deployment": "Application deployed successfully"
    }


@app.get("/metrics")
def get_metrics():
    """
    Returns the model evaluation metrics.
    """
    _, metrics = train_model()

    validate_quality(metrics)

    return metrics


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert request into DataFrame
        input_df = pd.DataFrame([request.dict()])

        # Rename columns to match training dataset
        feature_mapping = {
            "mean_radius": "mean radius",
            "mean_texture": "mean texture",
            "mean_perimeter": "mean perimeter",
            "mean_area": "mean area",
            "mean_smoothness": "mean smoothness",
            "mean_compactness": "mean compactness",
            "mean_concavity": "mean concavity",
            "mean_concave_points": "mean concave points",
            "mean_symmetry": "mean symmetry",
            "mean_fractal_dimension": "mean fractal dimension",
            "radius_error": "radius error",
            "texture_error": "texture error",
            "perimeter_error": "perimeter error",
            "area_error": "area error",
            "smoothness_error": "smoothness error",
            "compactness_error": "compactness error",
            "concavity_error": "concavity error",
            "concave_points_error": "concave points error",
            "symmetry_error": "symmetry error",
            "fractal_dimension_error": "fractal dimension error",
            "worst_radius": "worst radius",
            "worst_texture": "worst texture",
            "worst_perimeter": "worst perimeter",
            "worst_area": "worst area",
            "worst_smoothness": "worst smoothness",
            "worst_compactness": "worst compactness",
            "worst_concavity": "worst concavity",
            "worst_concave_points": "worst concave points",
            "worst_symmetry": "worst symmetry",
            "worst_fractal_dimension": "worst fractal dimension"
        }

        input_df.rename(columns=feature_mapping, inplace=True)

        prediction = model.predict(input_df)[0]

        result = "Malignant" if prediction == 0 else "Benign"

        return {
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Debug: Print all registered routes
print("\n========== REGISTERED ROUTES ==========")
for route in app.routes:
    print(route.path)
print("=======================================\n")
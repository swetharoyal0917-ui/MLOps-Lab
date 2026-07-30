import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "app", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Data", "breast_cancer.csv")


def train_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

    os.makedirs("app", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, metrics


def load_model():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    model, metrics = train_model()

    print("=" * 50)
    print("MODEL TRAINED SUCCESSFULLY")
    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key.capitalize():<10}: {value:.4f}")
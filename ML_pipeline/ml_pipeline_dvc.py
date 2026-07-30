import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import subprocess
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# Get DVC / Git Version
# ---------------------------------------
def get_dvc_version():
    try:
        version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()

        return version

    except Exception:
        return "Unknown"


# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("../Data/Telecom Customer Churn.csv")

df.columns = df.columns.str.strip()

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

df = df.dropna()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df.drop("customerID", axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

# ---------------------------------------
# Train Test Split
# ---------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# Feature Scaling
# ---------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------
# MLflow Experiment
# ---------------------------------------
mlflow.set_experiment("Churn_Model_Comparison")

with mlflow.start_run(run_name="Logistic Regression"):

    dvc_version = get_dvc_version()

    mlflow.log_param("data_version", dvc_version)

    model = LogisticRegression(max_iter=2000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", "Logistic Regression")

    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "model.pkl")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    dvc_file = "../Data/Telecom Customer Churn.csv.dvc"

    if os.path.exists(dvc_file):
        mlflow.log_artifact(dvc_file)

    print(f"Accuracy : {acc:.4f}")
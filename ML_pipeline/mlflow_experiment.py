import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("../Data/Telecom Customer Churn.csv")
df.columns = df.columns.str.strip()

# -------------------------------
# Data Preprocessing
# -------------------------------
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df.drop("customerID", axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# Train-Test Split
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# MLflow Experiment
# -------------------------------
mlflow.set_experiment("Churn_Experiment")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_accuracy = 0
best_model = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("Model", name)
        mlflow.log_metric("Accuracy", accuracy)

        mlflow.log_param("dataset", "Telecom Customer Churn.csv")
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy)

        print(f"{name} Accuracy : {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

print("\n==============================")
print("Best Model :", best_model)
print(f"Best Accuracy : {best_accuracy:.4f}")
print("==============================")
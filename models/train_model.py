import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Data/wine.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/wine_model.pkl")

print("=" * 50)
print("Model Trained Successfully")
print(f"Accuracy : {accuracy:.4f}")
print("Model saved as models/wine_model.pkl")
print("=" * 50)
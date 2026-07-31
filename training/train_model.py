from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create app folder if it doesn't exist
os.makedirs("app", exist_ok=True)

# Save model
joblib.dump(model, "app/iris_model.pkl")

print("=" * 50)
print("Model Trained Successfully!")
print("Model saved as iris_model.pkl")
print("=" * 50)
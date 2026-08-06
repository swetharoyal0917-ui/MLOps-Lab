from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, "iris_model.pkl")

print("=" * 50)
print("Model Trained Successfully!")
print("Model saved as iris_model.pkl")
print("=" * 50)
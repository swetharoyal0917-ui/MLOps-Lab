from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("ML_pipeline/model.pkl")


@app.route("/")
def home():
    return "Telecom Churn Prediction API is Running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    return jsonify({
        "Prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
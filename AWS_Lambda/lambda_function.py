import json
import joblib
import pandas as pd

# Load model only once
model = joblib.load("iris_model.pkl")

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])

        input_df = pd.DataFrame([{
            "sepal length (cm)": body["sepal_length"],
            "sepal width (cm)": body["sepal_width"],
            "petal length (cm)": body["petal_length"],
            "petal width (cm)": body["petal_width"]
        }])

        prediction = model.predict(input_df)[0]

        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": species[int(prediction)]
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
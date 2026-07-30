from data_loading import load_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model_training import split_data
from model_training import train_model
from evaluation import evaluate_model

def run_pipeline(file_path):

    df = load_data(file_path)

    df = preprocess_data(df)

    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    run_pipeline("../Data/Telecom Customer Churn.csv")
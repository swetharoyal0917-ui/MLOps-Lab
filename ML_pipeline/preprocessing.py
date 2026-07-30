import numpy as np
import pandas as pd

def preprocess_data(df):

    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    df = df.drop("customerID", axis=1)

    print("Preprocessing Completed")

    return df
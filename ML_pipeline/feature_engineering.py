import pandas as pd

def feature_engineering(df):

    def tenure_group(x):

        if x <= 12:
            return "0-1 Year"

        elif x <= 24:
            return "1-2 Years"

        elif x <= 48:
            return "2-4 Years"

        else:
            return "4+ Years"

    df["tenure_group"] = df["tenure"].apply(tenure_group)

    df = pd.get_dummies(df, drop_first=True)

    print("Feature Engineering Completed")

    return df
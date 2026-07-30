import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    print("Data Loaded Successfully")

    return df
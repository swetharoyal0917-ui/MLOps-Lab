import pandas as pd

from app.validation import validate_dataset

df = pd.read_csv("Data/breast_cancer.csv")

print(validate_dataset(df))
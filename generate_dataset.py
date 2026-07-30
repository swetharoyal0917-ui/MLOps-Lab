from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Load dataset
data = load_breast_cancer(as_frame=True)

# Convert to DataFrame
df = data.frame

# Rename target column
df.rename(columns={"target": "diagnosis"}, inplace=True)

# Create Data folder if it doesn't exist
os.makedirs("Data", exist_ok=True)

# Save CSV
df.to_csv("Data/breast_cancer.csv", index=False)

print("Dataset saved successfully!")
print(df.head())
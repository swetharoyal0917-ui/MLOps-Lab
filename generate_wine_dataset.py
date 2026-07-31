from sklearn.datasets import load_wine
import pandas as pd
import os

# Load dataset
wine = load_wine(as_frame=True)

df = wine.frame

# Create Data folder if not exists
os.makedirs("Data", exist_ok=True)

# Save dataset
df.to_csv("Data/wine.csv", index=False)

print("=" * 50)
print("Wine Dataset Generated Successfully")
print(f"Dataset Shape : {df.shape}")
print("Saved to : Data/wine.csv")
print("=" * 50)
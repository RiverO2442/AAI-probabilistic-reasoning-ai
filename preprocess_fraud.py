import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("./data/synthetic_fraud_dataset.csv")

# Define columns to discretize
numeric_cols = [
    "Transaction_Amount", "Risk_Score", "Account_Balance",
    "Transaction_Distance", "Card_Age", "Avg_Transaction_Amount_7d"
]

# Discretize into 4 bins using quantiles
for col in numeric_cols:
    df[col + "_bin"] = pd.qcut(df[col], q=4, labels=["low", "mid", "high", "vhigh"])

# Drop IDs and timestamp
df = df.drop(columns=["Transaction_ID", "User_ID", "Timestamp"])

# Save processed version
df.to_csv("data/fraud_data_processed.csv", index=False)
print("✅ Preprocessed fraud dataset saved as data/fraud_data_processed.csv")

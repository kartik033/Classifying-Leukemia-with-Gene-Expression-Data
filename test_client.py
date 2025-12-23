import requests

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
df = pd.read_csv("gene_merged.csv")
# Features and labels
X = df.drop(columns=["gsm_id", "cancer"])
y = df["cancer"]

# Encode ALL/AML -> 0/1
le = LabelEncoder()
y_enc = le.fit_transform(y)      # e.g. ALL=0, AML=1

# Train/test split (we'll use train for CV and model selection)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

import requests

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

# Loop through first 10 samples
for i in range(10):
    sample = X_test.iloc[i].to_dict()  # convert row to dict
    resp = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"features": sample}
    )
    true_label = le.inverse_transform([y_test[i]])[0]
    print(f"Sample {i+1}")
    print("True label:", true_label)
    print("Response :", resp.json())
    print("-" * 40)

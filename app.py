from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
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

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

app = Flask(__name__)

scaler = joblib.load(r"C:\Users\karti\Desktop\i\algoma\computing\scaler_gene.pkl")
model  = joblib.load(r"C:\Users\karti\Desktop\i\algoma\computing\svm_gene_model.pkl")

# Keep the same gene order as during training
feature_names = list(X_train.columns)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    feats_dict = data.get("features", {})
    # Build feature vector in correct order
    x_vec = [feats_dict.get(g, 0.0) for g in feature_names]
    x_vec = np.array(x_vec).reshape(1, -1)
    x_vec_s = scaler.transform(x_vec)

    prob_aml = model.predict_proba(x_vec_s)[0, 1]
    pred = int(prob_aml >= 0.5)
    label = le.inverse_transform([pred])[0]  # back to "ALL"/"AML"

    return jsonify({
        "prediction": label,
        "probability_AML": float(prob_aml)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

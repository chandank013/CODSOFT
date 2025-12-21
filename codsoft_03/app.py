"""
Customer Churn Prediction Application
CodSoft ML Internship - Task 3
Author: Chandan Kumar
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# Flask App Configuration

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend"
)


# Churn Predictor Class

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder_gender = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load("models/churn_prediction_model.pkl")
            self.scaler = joblib.load("artifacts/scaler.pkl")
            self.label_encoder_gender = joblib.load(
                "artifacts/label_encoder_gender.pkl"
            )
            print("✅ Churn prediction model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")

    def preprocess_customer(self, customer_data):
        df = pd.DataFrame([customer_data])

        # Encode Gender
        df["Gender"] = self.label_encoder_gender.transform(df["Gender"])

        # Geography One-Hot Encoding
        df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
        df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)
        df.drop("Geography", axis=1, inplace=True)

        # Scale numeric features
        scale_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
        df[scale_cols] = self.scaler.transform(df[scale_cols])

        # Column order
        ordered_cols = [
            "CreditScore", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "Geography_Germany", "Geography_Spain"
        ]

        return df[ordered_cols]

    def predict(self, customer_data):
        X = self.preprocess_customer(customer_data)
        pred = int(self.model.predict(X)[0])

        prob = None
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0][1])

        if prob is not None:
            if prob >= 0.8:
                risk = ("CRITICAL", "🔴")
            elif prob >= 0.6:
                risk = ("HIGH", "🟠")
            elif prob >= 0.4:
                risk = ("MEDIUM", "🟡")
            else:
                risk = ("LOW", "🟢")
        else:
            risk = ("UNKNOWN", "⚪")

        return {
            "will_churn": bool(pred),
            "churn_probability": prob,
            "risk_level": risk[0],
            "risk_color": risk[1],
            "timestamp": datetime.now().isoformat()
        }


predictor = ChurnPredictor()


# Routes

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        result = predictor.predict({
            "CreditScore": int(data["CreditScore"]),
            "Geography": data["Geography"],
            "Gender": data["Gender"],
            "Age": int(data["Age"]),
            "Tenure": int(data["Tenure"]),
            "Balance": float(data["Balance"]),
            "NumOfProducts": int(data["NumOfProducts"]),
            "HasCrCard": int(data["HasCrCard"]),
            "IsActiveMember": int(data["IsActiveMember"]),
            "EstimatedSalary": float(data["EstimatedSalary"])
        })

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": predictor.model is not None
    })



# Run Server

if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)

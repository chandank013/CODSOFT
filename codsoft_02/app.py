"""
Credit Card Fraud Detection Application
CodSoft ML Internship - Task 2
Author: Chandan Kumar

Real-time fraud detection system
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import os


# Flask App Configuration

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend"
)
 
# Fraud Detection Class

class FraudDetector:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load("models/fraud_detection_model.pkl")
            print("✅ Fraud detection model loaded successfully!")
        except Exception as e:
            print("❌ Failed to load model:", e)
            self.model = None

    def preprocess_transaction(self, data):
        """
        Expected features:
        Time, V1-V28, Amount (total 30)
        """
        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        features = np.array([float(data.get(f, 0)) for f in feature_order])
        return features.reshape(1, -1)

    def predict(self, data):
        if self.model is None:
            return {"error": "Model not loaded"}

        X = self.preprocess_transaction(data)

        pred = int(self.model.predict(X)[0])

        prob = None
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0][1])

        # Risk level
        if prob is None:
            risk = "UNKNOWN"
        elif prob >= 0.8:
            risk = "CRITICAL"
        elif prob >= 0.6:
            risk = "HIGH"
        elif prob >= 0.4:
            risk = "MEDIUM"
        elif prob >= 0.2:
            risk = "LOW"
        else:
            risk = "MINIMAL"

        return {
            "is_fraud": bool(pred),
            "fraud_probability": prob,
            "risk_level": risk,
            "timestamp": datetime.now().isoformat()
        }


detector = FraudDetector()

# Routes

@app.route("/")
def home():
    """Render index.html"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Fraud prediction API"""
    if detector.model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500

    try:
        data = request.get_json()

        result = detector.predict(data)

        if "error" in result:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500

        return jsonify({
            "success": True,
            "is_fraud": result["is_fraud"],
            "fraud_probability": round(result["fraud_probability"], 4) if result["fraud_probability"] else None,
            "risk_level": result["risk_level"],
            "message": "🚨 Fraudulent Transaction" if result["is_fraud"] else "✅ Legitimate Transaction"
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
        "model_loaded": detector.model is not None
    })


# Run Server
 
if __name__ == "__main__":
    print("🚀 Credit Card Fraud Detection Server Running")
    print("🌐 Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
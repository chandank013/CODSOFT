"""
Movie Genre Classification Web App
CodSoft ML Internship - Task 1
Author: Chandan Kumar
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
from datetime import datetime


# Flask App Config

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend"
)


# Movie Genre Classifier
 
class MovieGenreClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            with open("models/model.pkl", "rb") as f:
                self.model = pickle.load(f)

            with open("artifacts/vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)

            with open("artifacts/label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            print("✅ Movie Genre model loaded successfully!")
        except Exception as e:
            print("❌ Error loading artifacts:", e)
            self.model = None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(self, description):
        if not description:
            return {"error": "Empty description"}

        cleaned = self.preprocess_text(description)
        vectorized = self.vectorizer.transform([cleaned])

        prediction = self.model.predict(vectorized)[0]

        confidence = None
        if hasattr(self.model, "predict_proba"):
            confidence = float(max(self.model.predict_proba(vectorized)[0]))

        genre = self.label_encoder.inverse_transform([prediction])[0]

        return {
            "genre": genre,
            "confidence": confidence,
            "cleaned_text": cleaned,
            "timestamp": datetime.now().isoformat()
        }


classifier = MovieGenreClassifier()

 
# Routes
 
@app.route("/")
def home():
    """Render index.html"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict movie genre"""
    if classifier.model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500

    try:
        data = request.get_json()
        description = data.get("description", "").strip()

        if not description:
            return jsonify({
                "success": False,
                "error": "Please enter a movie description"
            }), 400

        result = classifier.predict(description)

        return jsonify({
            "success": True,
            "genre": result["genre"],
            "confidence": round(result["confidence"], 4) if result["confidence"] else None
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
        "model_loaded": classifier.model is not None
    })


 
# Run Server
 
if __name__ == "__main__":
    print("🚀 Movie Genre Classification Server Running")
    print("🌐 Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
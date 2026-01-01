"""
Spam SMS Detection Application
CodSoft ML Internship - Task 4
Author: Chandan Kumar

Classify SMS messages as Spam or Ham using ML
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# NLTK SETUP

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# FLASK APP CONFIG

app = Flask(
    __name__,
    template_folder='frontend',
    static_folder='frontend'
)


# LOAD MODEL & VECTORIZER

try:
    model = joblib.load('models/spam_detector_model.pkl')
    vectorizer = joblib.load('artifacts/tfidf_vectorizer.pkl')
    print("✅ Spam detection model & vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model = None
    vectorizer = None


# TEXT PREPROCESSING

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess SMS text
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)     # URLs
    text = re.sub(r'\S+@\S+', '', text)            # Emails
    text = re.sub(r'\d+', '', text)                # Numbers
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)


# ROUTES


@app.route('/')
def home():
    """Render frontend UI"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict Spam / Ham"""
    if model is None or vectorizer is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Train the model first.'
        }), 500

    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({
                'success': False,
                'error': 'Please enter a valid message.'
            }), 400

        # Preprocess & vectorize
        processed_text = preprocess_text(message)
        vectorized_text = vectorizer.transform([processed_text])

        # Prediction
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]

        spam_prob = round(probabilities[1] * 100, 2)
        ham_prob = round(probabilities[0] * 100, 2)

        return jsonify({
            'success': True,
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'confidence': max(spam_prob, ham_prob),
            'original_length': len(message),
            'processed_length': len(processed_text)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


# MAIN

if __name__ == '__main__':
    print("🚀 Starting Spam SMS Detection Server...")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# Load models and mappings
print("🔄 Loading models and mappings...")

try:
    # Load character mappings
    with open('artifacts/char_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    char_to_idx = mappings['char_to_idx']
    idx_to_char = mappings['idx_to_char']
    n_chars = mappings['n_chars']
    seq_length = mappings['seq_length']
    
    # Load models
    model_rnn = keras.models.load_model('models/rnn_model_best.keras')
    model_lstm = keras.models.load_model('models/lstm_model_best.keras')
    model_gru = keras.models.load_model('models/gru_model_best.keras')
    
    models = {
        'rnn': model_rnn,
        'lstm': model_lstm,
        'gru': model_gru
    }
    
    print("✅ Models and mappings loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️  Please run notebooks/preprocessing.ipynb and notebooks/model_training.ipynb first")
    models = None
    char_to_idx = None
    idx_to_char = None
    n_chars = None
    seq_length = None

def sample_with_temperature(preds, temperature=1.0):
    """Sample from probability distribution with temperature"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, length=200, temperature=1.0):
    """
    Generate text using trained model
    
    Args:
        model: Trained model
        seed_text: Starting text
        length: Number of characters to generate
        temperature: Sampling temperature (0.5-1.5)
    
    Returns:
        Generated text
    """
    if models is None:
        return "Error: Models not loaded"
    
    generated = seed_text.lower()
    seed = seed_text.lower()
    
    # Pad seed if too short
    if len(seed) < seq_length:
        seed = ' ' * (seq_length - len(seed)) + seed
    
    for _ in range(length):
        # Prepare input
        x_pred = np.zeros((1, seq_length, n_chars))
        for t, char in enumerate(seed[-seq_length:]):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
            else:
                # Use space for unknown characters
                x_pred[0, t, char_to_idx.get(' ', 0)] = 1
        
        # Predict
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        seed += next_char
    
    return generated

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text endpoint"""
    if models is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please run training notebooks first.'
        }), 500
    
    try:
        data = request.get_json()
        
        seed_text = data.get('seed_text', 'deep learning')
        model_type = data.get('model', 'lstm').lower()
        length = int(data.get('length', 200))
        temperature = float(data.get('temperature', 0.8))
        
        # Validate inputs
        if not seed_text or len(seed_text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Please provide seed text'
            }), 400
        
        if model_type not in models:
            return jsonify({
                'success': False,
                'error': f'Invalid model type. Choose from: {list(models.keys())}'
            }), 400
        
        if length < 50 or length > 1000:
            return jsonify({
                'success': False,
                'error': 'Length must be between 50 and 1000'
            }), 400
        
        if temperature < 0.1 or temperature > 2.0:
            return jsonify({
                'success': False,
                'error': 'Temperature must be between 0.1 and 2.0'
            }), 400
        
        # Generate text
        model = models[model_type]
        generated_text = generate_text(model, seed_text, length, temperature)
        
        result = {
            'success': True,
            'generated_text': generated_text,
            'seed_text': seed_text,
            'model': model_type,
            'length': len(generated_text) - len(seed_text),
            'total_length': len(generated_text),
            'temperature': temperature
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Generation error: {str(e)}'
        }), 500

@app.route('/models_info')
def models_info():
    """Get information about loaded models"""
    if models is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded'
        })
    
    info = {
        'success': True,
        'vocabulary_size': n_chars,
        'sequence_length': seq_length,
        'available_models': list(models.keys()),
        'model_details': {
            'rnn': {
                'name': 'Simple RNN',
                'parameters': int(model_rnn.count_params()),
                'layers': len(model_rnn.layers)
            },
            'lstm': {
                'name': 'LSTM',
                'parameters': int(model_lstm.count_params()),
                'layers': len(model_lstm.layers)
            },
            'gru': {
                'name': 'GRU',
                'parameters': int(model_gru.count_params()),
                'layers': len(model_gru.layers)
            }
        }
    }
    
    return jsonify(info)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'models_loaded': models is not None
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 HANDWRITTEN TEXT GENERATION SERVER")
    print("="*80)
    print("📱 Open http://127.0.0.1:5000 in your browser")
    print("⚙️  Press CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
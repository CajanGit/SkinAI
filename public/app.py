from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from predict import SkinClassifier
from glob import glob

app = Flask(__name__, static_folder='static')

# Configuration
MODELS_DIR = '/Users/anthonyjirano/Desktop/CSUF/CSUF Spring 2025/AI/Project Testing/skincare/models'
model_paths = glob(os.path.join(MODELS_DIR, '*.keras'))
classifiers = {os.path.basename(p): SkinClassifier(p) for p in model_paths}

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(temp_path)
        
        # Process with all models
        results = {}
        for model_name, classifier in classifiers.items():
            percentages, (final_label, confidence) = classifier.predict(temp_path)
            results[model_name] = {
                'final_label': final_label,
                'confidence': float(confidence),
                'percentages': percentages
            }
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
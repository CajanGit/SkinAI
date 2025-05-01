from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from predict import analyze_image, get_all_models  # Import your code

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MODELS_DIR = "/your/model/path"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
models = get_all_models(MODELS_DIR)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Run the model
        classifier = SkinClassifier(models[0])  # or loop like your CLI
        percentages, (label, conf) = classifier.predict(filepath)
        os.remove(filepath)  # Clean up

        return jsonify({
            'label': label.replace('_', ' ').title(),
            'confidence': f"{conf:.2%}",
            'percentages': {k.replace('_', ' ').title(): f"{v:.2%}" for k, v in percentages.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

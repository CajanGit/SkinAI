import os
import tensorflow as tf
import numpy as np
from PIL import Image

class SkinClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Define labels in ALPHABETICAL ORDER to match TensorFlow's default
        self.labels = ["clear_skin", "pimple", "rash"]  # Sorted alphabetically!
    
    def predict(self, image_path):
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        pred = self.model.predict(img_array)[0]
        
        # Pair labels with percentages in correct order
        percentages = {label: float(percent) for label, percent in zip(self.labels, pred)}
        final_label = max(percentages.items(), key=lambda x: x[1])
        return percentages, final_label

def get_all_models(models_dir):
    models = sorted(
        [f for f in os.listdir(models_dir) 
         if f.startswith('skin_model_') and f.endswith('.keras')],
        key=lambda x: int(x.split('_')[2].split('.')[0])
    )
    if not models:
        print(f" No models found in {models_dir}")
        exit()
    return [os.path.join(models_dir, m) for m in models]

def analyze_image(image_path, models):
    print("\n" + "="*50)
    print(f"Analyzing: {os.path.basename(image_path)}")
    print("="*50)
    
    for model_path in models:
        try:
            classifier = SkinClassifier(model_path)
            percentages, (final_label, final_conf) = classifier.predict(image_path)
            print(f"\nModel: {os.path.basename(model_path)}")
            
            # Print all percentages
            for label, conf in percentages.items():
                print(f"  {label.replace('_', ' ').title():<12}: {conf:.1%}")
            
            # Print final verdict
            print("\n  → Final Verdict:")
            print(f"  {final_label.replace('_', ' ').title()} ({final_conf:.1%} confidence)")
            print("-"*45)
        except Exception as e:
            print(f" Error with {os.path.basename(model_path)}: {str(e)}")

if __name__ == "__main__":
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    MODELS_DIR = "/Users/anthonyjirano/Desktop/CSUF/CSUF Spring 2025/AI/Project Testing/skincare/models"
    models = get_all_models(MODELS_DIR)
    
    print("  Skin Classifier - Multi-Model Comparison")
    print("Type 'exit' to quit\n")
    
    while True:
        image_path = input("Drag & drop image or enter path: ").strip(' \'"')
        
        if image_path.lower() in ['exit', 'quit', 'end']:
            break
            
        if not os.path.exists(image_path):
            print(f" File not found: {image_path}")
            continue
            
        analyze_image(image_path, models)
        
        print("\n" + "="*50)
        print("Test another image or type 'exit'")
        print("="*50 + "\n")

    print("\n Session ended")
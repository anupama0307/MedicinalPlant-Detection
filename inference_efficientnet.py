import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import sys

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
RESULTS_DIR = 'results/efficientnet'
MODEL_PATH = os.path.join(RESULTS_DIR, 'best_efficientnet_model.h5')
INDICES_PATH = os.path.join(RESULTS_DIR, 'class_indices.json')

def predict_image(image_path):
    """
    Predicts the class of an input image using the trained EfficientNetB0 model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        return

    if not os.path.exists(INDICES_PATH):
        print(f"Error: Class indices not found at {INDICES_PATH}.")
        return

    print(f"Loading EfficientNetB0 model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: If this is an EfficientNet loading error, try upgrading/reinstalling Keras or check compatibility.")
        return
    
    print(f"Loading class indices from {INDICES_PATH}...")
    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Invert the indices: {0: 'ClassA', 1: 'ClassB'}
    label_map = {v: k for k, v in class_indices.items()}

    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load and preprocess image
    # Note: EfficientNet usually expects inputs in range [0, 255] if using its preprocess_input
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # IMPORTANT: Use EfficientNet preprocess_input!
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = label_map[predicted_class_idx]
    
    print("-" * 30)
    print(f"Prediction Result (EfficientNetB0):")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)
    
    # Show image
    plt.imshow(image.load_img(image_path))
    plt.title(f"EfficientNet: {predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path)
    else:
        # Default path
        default_path = r"C:\Users\roshn\OneDrive\Desktop\FINAL_NNDL\BJ-S-004.jpg"
        
        print(f"Enter the path to the image (Default: {os.path.basename(default_path)}):")
        user_input = input("Image Path: ").strip().strip('"').strip("'")
        
        if user_input:
            predict_image(user_input)
        else:
            print(f"Using default image: {default_path}")
            predict_image(default_path)

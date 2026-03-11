import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
RESULTS_DIR = 'results/resnet'
MODEL_PATH = os.path.join(RESULTS_DIR, 'best_resnet_model.h5')
INDICES_PATH = os.path.join(RESULTS_DIR, 'class_indices.json')

def predict_image(image_path):
    """
    Predicts the class of an input image using the trained model and saved indices.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        return

    if not os.path.exists(INDICES_PATH):
        print(f"Error: Class indices not found at {INDICES_PATH}.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print(f"Loading class indices from {INDICES_PATH}...")
    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Invert the indices: {0: 'ClassA', 1: 'ClassB'}
    label_map = {v: k for k, v in class_indices.items()}

    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return

    # Load and preprocess image
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # IMPORTANT: Use ResNet50 preprocess_input!
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = label_map[predicted_class_idx]
    
    print("-" * 30)
    print(f"Prediction Result:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)
    
    # Show image with prediction (Need to undo preprocessing for display)
    # Undo mean subtraction roughly for visualization
    # Or just reload original image
    display_img = image.load_img(image_path)
    plt.imshow(display_img)
    plt.title(f"{predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import sys
    
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path)
    else:
        # User requested specific default path
        default_path = r"C:\Users\roshn\OneDrive\Desktop\FINAL_NNDL\BJ-S-004.jpg"
        
        # If no argument, ask user for input or use default
        print(f"Enter the path to the image (Default: {os.path.basename(default_path)}):")
        user_input = input("Image Path: ").strip().strip('"').strip("'")
        
        if user_input:
            predict_image(user_input)
        else:
            print(f"Using default image: {default_path}")
            predict_image(default_path)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
import matplotlib.pyplot as plt

# Configuration
# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../../data/Segmented Medicinal Leaf Images')
MODEL_PATH = os.path.join(BASE_DIR, 'vgg16_best.h5')
IMG_SIZE = (160, 160)

def load_class_names(data_dir):
    """
    Returns a sorted list of class names from the data directory.
    """
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def predict_image(model, img_path, class_names):
    """
    Predicts the class of an image.
    """
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Use VGG16 specific preprocessing
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = class_names[predicted_class_idx]

        return predicted_class, confidence, img
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, None

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    print("Loading class names...")
    if os.path.exists(DATA_DIR):
        class_names = load_class_names(DATA_DIR)
    else:
        # Fallback if running from a different directory or data not present
        print("Warning: Data directory not found. Using dummy classes or please ensure data is present.")
        return

    print("\n--- Medicinal Plant Predictor ---")
    while True:
        img_path = input("\nEnter path to image (or 'q' to quit): ").strip()
        if img_path.lower() == 'q':
            break # type: ignore
        
        # Remove quotes if user dragged and dropped file
        img_path = img_path.replace("'", "").replace('"', "")
        
        if not os.path.exists(img_path):
            print("File not found. Please try again.")
            continue
            
        predicted_class, confidence, img = predict_image(model, img_path, class_names)
        
        if predicted_class:
            print(f"\nPrediction: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            
            # Show image with prediction
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"{predicted_class} ({confidence:.2%})")
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    main()

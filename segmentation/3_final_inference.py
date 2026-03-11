import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Assuming your training folders map exactly to your class names
TRAIN_DIR = r'c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split/train'
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]) if os.path.exists(TRAIN_DIR) else []

print("Loading models...")
# 1. Load U-Net
if os.path.exists('best_unet_model.h5'):
    try:
        unet = tf.keras.models.load_model('best_unet_model.h5')
    except Exception as e:
        print(f"Error loading U-Net: {e}")
        unet = None
else:
    print("U-Net model not found.")
    unet = None

# 2. Choose and Load Your Classifier (DenseNet, ResNet, or EfficientNet)
# Change this path to whichever model you want to use!
#CLASSIFIER_PATH = '../results/densenet/best_densenet_model.h5' 
#CLASSIFIER_PATH = '../results/resnet/best_resnet_model.h5'
CLASSIFIER_PATH = '../results/efficientnet/best_efficientnet_model.h5'

if os.path.exists(CLASSIFIER_PATH):
    try:
        classifier_model = tf.keras.models.load_model(CLASSIFIER_PATH)
        print(f"Loaded Classifier from {CLASSIFIER_PATH}")
    except Exception as e:
        print(f"Error loading Classifier: {e}")
        classifier_model = None
else:
    print(f"Classifier model not found at {CLASSIFIER_PATH}. Ensure the path is correct.")
    classifier_model = None

def classify_plant(img_path):
    if not unet or not classifier_model:
        print("Models are not fully loaded. Exiting inference.")
        return

    print(f"Processing Image: {img_path}")
    # Preprocess Image
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    
    # Stage 1: Segmentation (Predict the mask using U-Net)
    mask = unet.predict(np.expand_dims(img_arr, axis=0), verbose=0)[0]
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # Stage 2: Masking (Multiply the image by the mask to delete the background)
    segmented_img = img_arr * binary_mask
    
    # Stage 3: Classification (Pass the cleaned image to the chosen classifier)
    pred = classifier_model.predict(np.expand_dims(segmented_img, axis=0), verbose=0)
    class_index = np.argmax(pred[0])
    
    if len(CLASS_NAMES) > class_index:
        plant_name = CLASS_NAMES[class_index]
    else:
        plant_name = f"Class Index {class_index}"
        
    confidence = pred[0][class_index] * 100
    
    # Final Result
    print(f"\n--- Result ---")
    print(f"Predicted Diagnosis: {plant_name}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the flow visually
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_arr); plt.title("Input Image"); plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask.squeeze(), cmap='gray'); plt.title("U-Net Mask"); plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_img); plt.title(f"Result: {plant_name}"); plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test on any random image path
    import sys
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # fallback demo image
        test_image_path = r'C:\Users\roshn\OneDrive\Desktop\FINAL_NNDL\mint.jpg'
        if not os.path.exists(test_image_path):
            print("Fallback image not found, please provide an image path via CLI.")
            sys.exit(1)
            
    if os.path.exists(test_image_path):
        classify_plant(test_image_path)
    else:
        print(f"Test image {test_image_path} does not exist!")

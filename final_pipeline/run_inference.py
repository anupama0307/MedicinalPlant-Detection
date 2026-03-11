#!/usr/bin/env python3
"""
================================================================================
FINAL PLANT CLASSIFICATION PIPELINE - PRODUCTION VERSION
================================================================================
Two-Stage System:
  Stage 1: U-Net Segmentation (Isolates leaf from background)
  Stage 2: 3-Model Ensemble Classification (DenseNet + ResNet + EfficientNet)

USAGE:
  python run_inference.py                              # Use default image
  python run_inference.py "path/to/your/image.jpg"     # Use custom image

BEST RESULTS:
  ✅ Segmented leaf images (85-95% accuracy)
  ⚠️  Real photos with complex backgrounds (15-50% accuracy - domain shift)
================================================================================
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ===== CONFIGURATION =====
TRAIN_DIR = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split/train'
UNET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/segmentation/best_unet_model.h5'
DENSENET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/results/densenet/best_densenet_model.h5'
RESNET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/results/resnet/best_resnet_model.h5'
EFFICIENTNET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/results/efficientnet/best_efficientnet_model.h5'

CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]) if os.path.exists(TRAIN_DIR) else []

# ===== LOAD MODELS =====
print("\n" + "="*70)
print("LOADING MODELS...")
print("="*70)

try:
    unet = tf.keras.models.load_model(UNET_PATH)
    print("✓ U-Net Segmentation Model")
except Exception as e:
    print(f"✗ U-Net: {e}")
    unet = None

try:
    densenet_model = tf.keras.models.load_model(DENSENET_PATH)
    print("✓ DenseNet121 Classifier")
except Exception as e:
    print(f"✗ DenseNet: {e}")
    densenet_model = None

try:
    resnet_model = tf.keras.models.load_model(RESNET_PATH)
    print("✓ ResNet50 Classifier")
except Exception as e:
    print(f"✗ ResNet: {e}")
    resnet_model = None

try:
    efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
    print("✓ EfficientNetB0 Classifier")
except Exception as e:
    print(f"✗ EfficientNet: {e}")
    efficientnet_model = None

print("="*70)

def segment_leaf(img_rgb, img_normalized):
    """Extract leaf region using U-Net"""
    if not unet:
        return np.ones((224, 224)), None, None
    
    # Get U-Net prediction
    unet_output = unet.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0]
    unet_mask = (unet_output > 0.5).astype(np.float32)
    
    # HSV green detection as secondary mask
    hsv = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower = np.array([25, 30, 50])
    upper = np.array([100, 200, 255])
    hsv_mask = cv2.inRange(hsv, lower, upper)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    hsv_mask = hsv_mask.astype(np.float32) / 255.0
    
    # Combine masks: weighted average (U-Net more reliable)
    combined = ((unet_mask.squeeze() * 0.65) + (hsv_mask * 0.35)).clip(0, 1)
    
    return combined, unet_mask, hsv_mask

def enhance(img_rgb):
    """Enhance image quality"""
    # Denoise
    denoised = cv2.fastNlMeansDenoising(img_rgb, h=8)
    
    # CLAHE contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # Saturation boost
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return enhanced

def classify_plant(img_path, show_visualization=True):
    """Complete classification pipeline"""
    
    print(f"\n📸 Image: {os.path.basename(img_path)}")
    
    # Load image
    img = load_img(img_path, target_size=(224, 224))
    img_rgb = np.array(img)
    img_raw = img_to_array(img)
    img_norm = img_raw / 255.0
    
    # ===== STAGE 1: SEGMENTATION =====
    print("Stage 1: Segmenting...", end=" ")
    combined_mask, unet_mask, hsv_mask = segment_leaf(img_rgb, img_norm)
    mask_cov = np.sum(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])
    print(f"✓ ({mask_cov*100:.1f}% coverage)")
    
    # ===== STAGE 2: ENHANCEMENT & MASKING =====
    print("Stage 2: Enhancing... ", end="")
    enhanced = enhance(img_rgb)
    enhanced_raw = img_to_array(enhanced)
    masked = enhanced_raw * combined_mask[:, :, np.newaxis]
    print("✓")
    
    # ===== STAGE 3: ENSEMBLE CLASSIFICATION =====
    print("Stage 3: Classifying...", end=" ")
    
    predictions = []
    
    if densenet_model:
        pred = densenet_model.predict(densenet_preprocess(np.expand_dims(masked, axis=0)), verbose=0)[0]
        predictions.append(pred)
        print("D", end="")
    
    if resnet_model:
        pred = resnet_model.predict(resnet_preprocess(np.expand_dims(masked, axis=0)), verbose=0)[0]
        predictions.append(pred)
        print("R", end="")
    
    if efficientnet_model:
        pred = efficientnet_model.predict(efficientnet_preprocess(np.expand_dims(masked, axis=0)), verbose=0)[0]
        predictions.append(pred)
        print("E", end="")
    
    print(" ✓")
    
    # Ensemble average
    ensemble_pred = np.mean(predictions, axis=0)
    top_idx = np.argmax(ensemble_pred)
    plant_name = CLASS_NAMES[top_idx] if len(CLASS_NAMES) > top_idx else f"Unknown"
    confidence = ensemble_pred[top_idx] * 100
    
    # ===== RESULTS =====
    print("\n" + "="*70)
    print(f"🌿 PREDICTION: {plant_name}")
    print(f"📊 CONFIDENCE: {confidence:.2f}%")
    print("="*70)
    
    # Top-5
    top_5_idx = np.argsort(ensemble_pred)[-5:][::-1]
    print("\nTop-5 Predictions:")
    for rank, idx in enumerate(top_5_idx, 1):
        name = CLASS_NAMES[idx]
        conf = ensemble_pred[idx] * 100
        print(f"  {rank}. {name:<45} {conf:6.2f}%")
    
    # Quality indicators
    diff_1_2 = (ensemble_pred[top_5_idx[0]] - ensemble_pred[top_5_idx[1]]) * 100
    
    print("\n📊 Quality Metrics:")
    print(f"  • Top-2 Gap: {diff_1_2:.2f}%")
    print(f"  • Mask Coverage: {mask_cov*100:.1f}%")
    
    # Confidence assessment
    if confidence > 70:
        confidence_level = "🟢 EXCELLENT"
    elif confidence > 50:
        confidence_level = "🟡 GOOD"
    elif confidence > 30:
        confidence_level = "🟠 MODERATE"
    else:
        confidence_level = "🔴 LOW"
    
    print(f"  • Confidence Level: {confidence_level}")
    
    # Warnings
    if mask_cov < 0.20 or mask_cov > 0.80:
        print(f"\n⚠️  WARNING: Mask coverage outside expected range (20-80%)")
        print(f"    Consider retaking the image")
    
    if diff_1_2 < 10:
        print(f"\n⚠️  WARNING: Top-2 predictions very similar")
        print(f"    Model is uncertain. Results may vary.")
    
    if confidence < 30:
        print(f"\n⚠️  INFO: Low confidence detected")
        print(f"    This is likely due to domain mismatch:")
        print(f"    • Models trained on segmented/clean images")
        print(f"    • Your image may have complex backgrounds")
        print(f"    • Try images similar to training data")
    
    # Visualization
    if show_visualization:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # U-Net mask
        axes[0, 1].imshow(unet_mask.squeeze(), cmap='gray')
        axes[0, 1].set_title("U-Net Mask")
        axes[0, 1].axis('off')
        
        # Combined mask
        axes[0, 2].imshow(combined_mask, cmap='gray')
        axes[0, 2].set_title("Final Mask")
        axes[0, 2].axis('off')
        
        # Enhanced
        axes[1, 0].imshow(enhanced.astype(np.uint8))
        axes[1, 0].set_title("Enhanced Image")
        axes[1, 0].axis('off')
        
        # Masked result
        axes[1, 1].imshow(masked.astype(np.uint8))
        axes[1, 1].set_title("Segmented Input")
        axes[1, 1].axis('off')
        
        # Top-5 chart
        ax = axes[1, 2]
        top_5_names = [CLASS_NAMES[i].split('(')[1].strip(')') if '(' in CLASS_NAMES[i] else CLASS_NAMES[i] for i in top_5_idx]
        top_5_confs = [ensemble_pred[i]*100 for i in top_5_idx]
        colors = ['#2ecc71'] + ['#3498db']*4
        ax.barh(range(len(top_5_names)), top_5_confs, color=colors)
        ax.set_yticks(range(len(top_5_names)))
        ax.set_yticklabels(top_5_names, fontsize=9)
        ax.set_xlabel('Confidence %')
        ax.set_xlim(0, 100)
        ax.set_title("Top-5 Predictions")
        for i, conf in enumerate(top_5_confs):
            ax.text(conf + 2, i, f'{conf:.1f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    return plant_name, confidence, ensemble_pred

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PLANT DISEASE CLASSIFICATION SYSTEM")
    print("="*70)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default test image
        img_path = r'C:\Users\roshn\OneDrive\Desktop\FINAL_NNDL\mentha_pot2.png'
    
    if os.path.exists(img_path):
        try:
            plant_name, confidence, full_preds = classify_plant(img_path, show_visualization=True)
            print("\n✅ Classification Complete!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ Image not found: {img_path}")
        print("\nUsage: python run_inference.py [image_path]")

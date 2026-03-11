#!/usr/bin/env python3
"""
ADVANCED PLANT CLASSIFICATION PIPELINE
Three-Model Ensemble with Smart Preprocessing
Works best with segmented images similar to training data
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

# --- Configuration ---
TRAIN_DIR = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split/train'
UNET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/segmentation/best_unet_model.h5'
DENSENET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/results/densenet/best_densenet_model.h5'
RESNET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/results/resnet/best_resnet_model.h5'
EFFICIENTNET_PATH = r'C:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/results/efficientnet/best_efficientnet_model.h5'

CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]) if os.path.exists(TRAIN_DIR) else []

print("Loading models...")
try:
    unet = tf.keras.models.load_model(UNET_PATH)
    print("✓ U-Net loaded")
except Exception as e:
    print(f"✗ U-Net error: {e}")
    unet = None

try:
    densenet_model = tf.keras.models.load_model(DENSENET_PATH)
    print("✓ DenseNet loaded")
except Exception as e:
    print(f"✗ DenseNet error: {e}")
    densenet_model = None

try:
    resnet_model = tf.keras.models.load_model(RESNET_PATH)
    print("✓ ResNet loaded")
except Exception as e:
    print(f"✗ ResNet error: {e}")
    resnet_model = None

try:
    efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
    print("✓ EfficientNet loaded")
except Exception as e:
    print(f"✗ EfficientNet error: {e}")
    efficientnet_model = None

def smart_segmentation(img_rgb, img_unet_normalized):
    """
    Multi-method segmentation combining U-Net + HSV + Contours
    """
    # U-Net mask
    unet_output = unet.predict(np.expand_dims(img_unet_normalized, axis=0), verbose=0)[0]
    unet_mask = (unet_output > 0.5).astype(np.float32)
    
    # HSV-based green detection
    hsv = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    
    # Primary green range (expanded)
    lower_green = np.array([20, 25, 40])
    upper_green = np.array([100, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological operations
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)), iterations=2)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    hsv_mask = hsv_mask.astype(np.float32) / 255.0
    
    # Combine: weighted U-Net + HSV for balanced detection
    combined = ((unet_mask.squeeze() * 0.6) + (hsv_mask * 0.4)).clip(0, 1)
    
    return combined, unet_mask, hsv_mask

def enhance_image(img_rgb):
    """Enhance image for better classification"""
    # Denoise
    denoised = cv2.fastNlMeansDenoising(img_rgb, h=9)
    
    # CLAHE contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # Saturation boost
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return enhanced

def classify_with_all_models(masked_input_densenet, masked_input_resnet, masked_input_efficientnet):
    """
    Run all three classifiers and return ensemble prediction
    """
    predictions = {}
    
    if densenet_model:
        pred_den = densenet_model.predict(densenet_preprocess(np.expand_dims(masked_input_densenet, axis=0)), verbose=0)[0]
        predictions['DenseNet'] = pred_den
    
    if resnet_model:
        pred_res = resnet_model.predict(resnet_preprocess(np.expand_dims(masked_input_resnet, axis=0)), verbose=0)[0]
        predictions['ResNet'] = pred_res
    
    if efficientnet_model:
        pred_eff = efficientnet_model.predict(efficientnet_preprocess(np.expand_dims(masked_input_efficientnet, axis=0)), verbose=0)[0]
        predictions['EfficientNet'] = pred_eff
    
    # Ensemble: Average predictions
    ensemble = np.mean([p for p in predictions.values()], axis=0)
    
    return ensemble, predictions

def classify_plant(img_path):
    if not unet or (not densenet_model and not resnet_model and not efficientnet_model):
        print("❌ Models not loaded!")
        return

    print(f"\n📸 Processing: {os.path.basename(img_path)}")
    
    # Load image
    img = load_img(img_path, target_size=(224, 224))
    img_rgb = np.array(img)
    img_raw = img_to_array(img)
    img_unet = img_raw / 255.0
    
    # Stage 1: Segmentation
    print("🔍 Segmenting...")
    combined_mask, unet_mask, hsv_mask = smart_segmentation(img_rgb, img_unet)
    
    # Stage 2: Enhancement
    print("✨ Enhancing...")
    enhanced = enhance_image(img_rgb)
    enhanced_raw = img_to_array(enhanced)
    
    # Stage 3: Create masked inputs for all variants
    # Binary mask for bright visualization (0 or 1 only)
    binary_mask = (combined_mask > 0.5).astype(np.float32)
    # Soft mask for classification (0.0-1.0 for smooth transitions)
    masked_enhanced = enhanced_raw * combined_mask[:, :, np.newaxis]
    masked_enhanced_display = enhanced_raw * binary_mask[:, :, np.newaxis]
    masked_original = img_raw * combined_mask[:, :, np.newaxis]
    
    # Try multiple input strategies for robustness
    inputs = [
        ("Enhanced+Mask", masked_enhanced),
        ("Original+Mask", masked_original),
        ("Enhanced Only", enhanced_raw),
    ]
    
    all_predictions = []
    
    for input_name, input_img in inputs:
        # Ensure we have valid input
        if np.max(input_img) == 0:
            continue
            
        pred_ens, pred_ind = classify_with_all_models(input_img, input_img, input_img)
        all_predictions.append(pred_ens)
    
    if not all_predictions:
        print("❌ Segmentation failed!")
        return
    
    # Final ensemble: Average across all strategies
    final_pred = np.mean(all_predictions, axis=0)
    top_idx = np.argmax(final_pred)
    plant_name = CLASS_NAMES[top_idx] if len(CLASS_NAMES) > top_idx else f"Unknown (Index {top_idx})"
    confidence = final_pred[top_idx] * 100
    
    # Results
    print(f"\n{'='*70}")
    print(f"🎯 PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"Plant: {plant_name}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"{'='*70}\n")
    
    # Top-5
    top_5 = np.argsort(final_pred)[-5:][::-1]
    print("📊 Top-5 Candidates:")
    for rank, idx in enumerate(top_5, 1):
        name = CLASS_NAMES[idx]
        conf = final_pred[idx] * 100
        marker = "✅" if rank == 1 else "  "
        print(f"  {marker} {rank}. {name:<45} {conf:6.2f}%")
    
    # Warnings
    diff_1_2 = (final_pred[top_5[0]] - final_pred[top_5[1]]) * 100
    mask_coverage = np.sum(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])
    
    print(f"\n⚡ Analysis:")
    print(f"  - Score Gap (1st-2nd): {diff_1_2:.2f}%")
    print(f"  - Mask Coverage: {mask_coverage*100:.1f}%")
    
    if diff_1_2 < 10 and mask_coverage < 25:
        print(f"  ⚠️  WARNING: Low confidence + poor segmentation")
        print(f"     This image may not match the training dataset")
    elif diff_1_2 < 10:
        print(f"  ⚠️  TIP: Top predictions are very close")
        print(f"     Consider using a clearer image")
    
    # Visualization
    fig = plt.figure(figsize=(14, 9))
    
    # Original & enhanced
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Original", fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(enhanced.astype(np.uint8))
    ax2.set_title("Enhanced", fontweight='bold')
    ax2.axis('off')
    
    # Masks
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(unet_mask.squeeze(), cmap='gray')
    ax3.set_title("U-Net Mask", fontweight='bold')
    ax3.axis('off')
    
    # Combined mask
    ax5 = plt.subplot(3, 3, 4)
    ax5.imshow(combined_mask, cmap='gray')
    ax5.set_title("Combined", fontweight='bold')
    ax5.axis('off')
    
    # Masked result
    ax6 = plt.subplot(3, 3, 5)
    ax6.imshow(masked_enhanced_display.astype(np.uint8))
    ax6.set_title("Final Input", fontweight='bold')
    ax6.axis('off')
    
    # Top-5 bar chart
    ax7 = plt.subplot(3, 3, 6)
    top_5_names = [CLASS_NAMES[i].split('(')[1].strip(')') if '(' in CLASS_NAMES[i] else CLASS_NAMES[i] for i in top_5]
    top_5_confs = [final_pred[i]*100 for i in top_5]
    colors_bar = ['#2ecc71'] + ['#3498db']*4
    ax7.barh(range(len(top_5_names)), top_5_confs, color=colors_bar)
    ax7.set_yticks(range(len(top_5_names)))
    ax7.set_yticklabels(top_5_names, fontsize=9)
    ax7.set_xlabel('Confidence %')
    ax7.set_xlim(0, 100)
    for i, conf in enumerate(top_5_confs):
        ax7.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=8)
    ax7.set_title("Top-5 Ensemble", fontweight='bold')
    
    # Info box
    ax10 = plt.subplot(3, 3, 7)
    ax10.axis('off')
    info = f"""RESULT
━━━━━━━━━━━━
Plant: {plant_name}
Conf: {confidence:.1f}%

INFO
━━━━━━━━━━━━
Models: 3
Strategies: 3
Mask Cov: {mask_coverage*100:.1f}%
"""
    ax10.text(0.1, 0.9, info, transform=ax10.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # Default to real photo
        test_image_path = r'C:\Users\roshn\OneDrive\Desktop\FINAL_NNDL\hib.jpg'
        
    if os.path.exists(test_image_path):
        classify_plant(test_image_path)
    else:
        print(f"❌ Image not found: {test_image_path}")

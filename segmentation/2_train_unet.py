import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

IMAGE_DIR = r'c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split/train'
MASK_DIR = r'c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/unet_masks'

# Create results directory
RESULTS_DIR = 'results/unet'
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)  # Clean start
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Data Generator ---
# This safely loads images and masks in small batches so your RAM doesn't crash
class SegDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, mask_dir, batch_size, img_size):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.pairs = []
        if os.path.exists(img_dir):
            for category in os.listdir(img_dir):
                cat_img_dir = os.path.join(img_dir, category)
                cat_mask_dir = os.path.join(mask_dir, category)
                if not os.path.isdir(cat_img_dir): continue
                
                if os.path.exists(cat_mask_dir):
                    for f in os.listdir(cat_img_dir):
                        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(cat_img_dir, f)
                            mask_path = os.path.join(cat_mask_dir, f)
                            if os.path.exists(mask_path):
                                self.pairs.append((img_path, mask_path))
        
        np.random.shuffle(self.pairs)
        print(f"Found {len(self.pairs)} image-mask pairs.")
        
    def __len__(self):
        return int(np.ceil(len(self.pairs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        
        for img_path, mask_path in batch_pairs:
            # Load and normalize Image
            img = img_to_array(load_img(img_path, target_size=self.img_size)) / 255.0
            
            # Load Mask, grab one channel, normalize (0.0 or 1.0)
            mask = img_to_array(load_img(mask_path, target_size=self.img_size, color_mode="grayscale"))
            mask = mask / 255.0
            mask[mask > 0.5] = 1.0
            mask[mask <= 0.5] = 0.0
            
            batch_x.append(img)
            batch_y.append(mask)
            
        return np.array(batch_x), np.array(batch_y)


# --- 2. U-Net Model ---
def build_unet(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Bridge
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    
    # Decoder
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.concatenate([u1, c1])
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c2)
    return models.Model(inputs, outputs)

# --- 3. Training ---
if __name__ == "__main__":
    print("Building model...")
    model = build_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Loading data generator...")
    train_gen = SegDataGenerator(IMAGE_DIR, MASK_DIR, BATCH_SIZE, IMG_SIZE)
    
    if len(train_gen.pairs) == 0:
        print("No paired images and masks found! Did you run 1_generate_masks.py?")
    else:
        print("Starting training...")
        # Add early stopping so it stops if it isn't getting better
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        
        # In a real run, workers > 1 might be useful, but for stability cross-platform we use 1 for now
        history = model.fit(train_gen, epochs=EPOCHS, callbacks=callbacks)
        
        model.save(os.path.join(RESULTS_DIR, 'best_unet_model.h5'))
        print(f"U-Net training complete. Model saved to '{RESULTS_DIR}/best_unet_model.h5'")
        
        # Save training history and metrics to text file
        with open(os.path.join(RESULTS_DIR, 'unet_training_report.txt'), 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("U-Net Segmentation Model Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Epochs: {len(history.history['loss'])}\n")
            f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Best Training Loss: {min(history.history['loss']):.4f}\n")
            f.write(f"Best Training Accuracy: {max(history.history['accuracy']):.4f}\n")
            f.write(f"\nTotal Image-Mask Pairs: {len(train_gen.pairs)}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Image Size: {IMG_SIZE}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Training History (Last 5 Epochs)\n")
            f.write("=" * 50 + "\n")
            f.write("Epoch | Loss      | Accuracy\n")
            f.write("-" * 35 + "\n")
            
            start_idx = max(0, len(history.history['loss']) - 5)
            for i in range(start_idx, len(history.history['loss'])):
                loss = history.history['loss'][i]
                acc = history.history['accuracy'][i]
                f.write(f"{i+1:5d} | {loss:9.4f} | {acc:8.4f}\n")
        
        print(f"Training report saved to '{RESULTS_DIR}/unet_training_report.txt'")

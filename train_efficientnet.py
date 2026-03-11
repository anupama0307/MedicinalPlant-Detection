import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import json

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATA_DIR = r"c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split"
RESULTS_DIR = 'results/efficientnet'

if os.path.exists(RESULTS_DIR):
    import shutil
    shutil.rmtree(RESULTS_DIR) # Clean start
os.makedirs(RESULTS_DIR, exist_ok=True)

# URL for EfficientNetB0 No Top Weights (Standard Keras Source)
WEIGHTS_URL = 'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5'

def load_data_local(preprocessing_function=None):
    """
    Dedicated Data Loader for EfficientNet (Internal).
    Enforces RGB color mode and handling.
    """
    print(f"Loading data from {DATA_DIR} [EfficientNet Mode]...")
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    if preprocessing_function:
        rescale_factor = None
    else:
        rescale_factor = 1./255

    # Data Augmentation (Same robust settings)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rescale=rescale_factor,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rescale=rescale_factor
    )
    
    print("Found training images:")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb', # FORCE RGB
        shuffle=True
    )
    
    print("Found validation images:")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb', # FORCE RGB
        shuffle=False
    )
    
    print("Found test images:")
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb', # FORCE RGB
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, train_generator.class_indices

def build_efficientnet_model(num_classes):
    print("Building EfficientNetB0 model...")
    
    # 1. Build model structure WITHOUT weights first
    # This guarantees the input shape is correctly set to (224, 224, 3)
    base_model = EfficientNetB0(
        weights=None, 
        include_top=False, 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # 2. Key Check: Verify Input Shape
    print(f"Base Model Input Shape: {base_model.input_shape}")
    
    # 3. Manually download and load weights
    print("Loading ImageNet weights manually...")
    weights_path = tf.keras.utils.get_file(
        'efficientnetb0_notop.h5',
        WEIGHTS_URL,
        cache_subdir='models'
    )
    base_model.load_weights(weights_path)
    print("Weights loaded successfully.")
    
    # FREEZE the entire base model initially
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Adjusted Regularization: L2 (1e-4) and Dropout (0.2) for better convergence
    # EfficientNet benefits from less dropout when underfitting compared to ResNet
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x) 
    x = Dropout(0.2)(x) 
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def plot_history(history, stage_name=""):
    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'EfficientNetB0 Accuracy ({stage_name})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'efficientnet_acc_{stage_name}.png'))
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'EfficientNetB0 Loss ({stage_name})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'efficientnet_loss_{stage_name}.png'))
    plt.close()

def main():
    # 1. Load Data
    print("Loading data for EfficientNetB0 (Internal Loader)...")
    train_generator, val_generator, test_generator, class_indices = load_data_local(preprocessing_function=preprocess_input)
    
    if train_generator is None:
        return

    num_classes = len(class_indices)
    print(f"Classes: {num_classes}")

    # SAVE CLASS INDICES
    with open(os.path.join(RESULTS_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)
    
    # 2. Build Model
    model, base_model = build_efficientnet_model(num_classes)
    
    # --- STAGE 1: Train Head (Frozen Base) ---
    print("\n[Stage 1] Training generic features (Base Frozen)...")
    # Label Smoothing improves confidence calibration (helps fix 'low confidence' issue)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    
    callbacks_s1 = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(os.path.join(RESULTS_DIR, 'best_efficientnet_model.h5'), monitor='val_loss', save_best_only=True)
    ]
    
    history1 = model.fit(
        train_generator,
        epochs=12,
        validation_data=val_generator,
        callbacks=callbacks_s1
    )
    plot_history(history1, "Stage1")

    # --- STAGE 2: Fine-Tuning ---
    print("\n[Stage 2] Fine-Tuning specific features (Unfreezing top layers)...")
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze bottom layers 
    # EfficientNetB0 has ~237 layers. Freeze ~40% (Unfreeze ~60% layers)
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * 0.4)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
        
    # Compile with low learning rate and label smoothing
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    
    callbacks_s2 = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        ModelCheckpoint(os.path.join(RESULTS_DIR, 'best_efficientnet_model.h5'), monitor='val_loss', save_best_only=True)
    ]
    
    history2 = model.fit(
        train_generator,
        epochs=30, # Increased from 20 to allow better convergence 
        validation_data=val_generator,
        callbacks=callbacks_s2
    )
    plot_history(history2, "Stage2")
    
    # SAVE FINAL TRAINED MODEL
    model.save(os.path.join(RESULTS_DIR, 'final_efficientnet_model.h5'))
    
    # 3. Final Evaluation
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(test_generator)
    
    train_acc = history2.history['accuracy'][-1]
    train_loss = history2.history['loss'][-1]
    
    print(f"Final Train Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # 4. Metrics & Plots
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_indices.keys(), yticklabels=class_indices.keys())
    plt.title('EfficientNetB0 Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'efficientnet_confusion_matrix.png'))
    plt.close()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_indices.keys())
    with open(os.path.join(RESULTS_DIR, 'efficientnet_classification_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\n\nMean Train Accuracy: {train_acc:.4f}")
        f.write(f"\nMean Train Loss: {train_loss:.4f}")
        f.write(f"\nFinal Test Accuracy: {test_acc:.4f}")
        f.write(f"\nFinal Test Loss: {test_loss:.4f}")
        
    # Save Metrics CSV (Combine histories)
    hist1_df = pd.DataFrame(history1.history)
    hist2_df = pd.DataFrame(history2.history)
    hist1_df['epoch'] = range(1, len(hist1_df) + 1)
    hist1_df['stage'] = 1
    hist2_df['epoch'] = range(len(hist1_df) + 1, len(hist1_df) + len(hist2_df) + 1)
    hist2_df['stage'] = 2
    full_metrics = pd.concat([hist1_df, hist2_df])
    full_metrics.to_csv(os.path.join(RESULTS_DIR, 'efficientnet_metrics.csv'), index=False)
    
    # Summary File
    evaluation_summary = {
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss'],
        'Value': [train_acc, test_acc, train_loss, test_loss]
    }
    pd.DataFrame(evaluation_summary).to_csv(os.path.join(RESULTS_DIR, 'evaluation_results.csv'), index=False)
        
    print("Training Complete. Results saved to results/efficientnet/")

if __name__ == "__main__":
    main()

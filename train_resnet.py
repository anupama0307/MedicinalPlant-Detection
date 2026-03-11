import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import json
from data_loader import load_data, IMG_HEIGHT, IMG_WIDTH

# Create results directory
RESULTS_DIR = 'results/resnet'
if os.path.exists(RESULTS_DIR):
    import shutil
    shutil.rmtree(RESULTS_DIR) # Clean start
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_resnet_model(num_classes):
    # Load ResNet50 with ImageNet weights, excluding the top (classification) layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # FREEZE the entire base model initially
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # [MODIFIED] Added L2 Regularization and increased Dropout to prevent "too perfect" results
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x) 
    x = Dropout(0.7)(x) # Increased dropout from 0.5 to 0.7
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def plot_history(history, stage_name=""):
    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'ResNet50 Accuracy ({stage_name})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'resnet_acc_{stage_name}.png'))
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'ResNet50 Loss ({stage_name})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'resnet_loss_{stage_name}.png'))
    plt.close()

def main():
    # 1. Load Data
    print("Loading data...")
    train_generator, val_generator, test_generator, class_indices = load_data(preprocessing_function=preprocess_input)
    
    if train_generator is None:
        return

    num_classes = len(class_indices)
    print(f"Classes: {num_classes}")

    # SAVE CLASS INDICES
    with open(os.path.join(RESULTS_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)
    
    # 2. Build Model
    model, base_model = build_resnet_model(num_classes)
    
    # --- STAGE 1: Train Head (Frozen Base) ---
    print("\n[Stage 1] Training generic features (Base Frozen)...")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks_s1 = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(os.path.join(RESULTS_DIR, 'best_resnet_model.h5'), monitor='val_loss', save_best_only=True)
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
    
    # Freeze bottom layers (Keep first 140 frozen, train top ~35)
    for layer in base_model.layers[:140]:
        layer.trainable = False
        
    # Compile with low learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks_s2 = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        ModelCheckpoint(os.path.join(RESULTS_DIR, 'best_resnet_model.h5'), monitor='val_loss', save_best_only=True)
    ]
    
    history2 = model.fit(
        train_generator,
        epochs=20, # Increased epochs to allow slower convergence with regularization
        validation_data=val_generator,
        callbacks=callbacks_s2
    )
    plot_history(history2, "Stage2")
    
    # SAVE FINAL TRAINED MODEL
    model.save(os.path.join(RESULTS_DIR, 'final_resnet_model.h5'))
    
    # 3. Final Evaluation
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(test_generator)
    
    # Get Train Accuracy (Final)
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
    plt.title('ResNet50 Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'resnet_confusion_matrix.png'))
    plt.close()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_indices.keys())
    with open(os.path.join(RESULTS_DIR, 'resnet_classification_report.txt'), 'w') as f:
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
    full_metrics.to_csv(os.path.join(RESULTS_DIR, 'resnet_metrics.csv'), index=False)
    
    # [NEW] Simple Summary File
    evaluation_summary = {
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss'],
        'Value': [train_acc, test_acc, train_loss, test_loss]
    }
    pd.DataFrame(evaluation_summary).to_csv(os.path.join(RESULTS_DIR, 'evaluation_results.csv'), index=False)
        
    print("Training Complete. Results saved to results/resnet/")

if __name__ == "__main__":
    main()

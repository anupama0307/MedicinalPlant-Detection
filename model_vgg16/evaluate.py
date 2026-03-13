import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.data_utils import load_and_split_data
from tensorflow.keras.applications.vgg16 import preprocess_input

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../../data/Segmented Medicinal Leaf Images')
MODEL_PATH = os.path.join(BASE_DIR, 'vgg16_best.h5')
HISTORY_PATH = os.path.join(BASE_DIR, '../../results/vgg16_history.csv')
RESULTS_DIR = os.path.join(BASE_DIR, '../../results')

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

os.makedirs(os.path.join(RESULTS_DIR, "graphs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrix"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "reports"), exist_ok=True)

# ---------------- PLOT HISTORY ----------------
def plot_history(history_df):

    # Accuracy graph
    plt.figure(figsize=(10,6))
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('VGG16 Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'graphs/vgg16_accuracy.png'))
    plt.close()

    # Loss graph
    plt.figure(figsize=(10,6))
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('VGG16 Loss Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'graphs/vgg16_loss.png'))
    plt.close()

# ---------------- CONFUSION MATRIX ----------------
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("VGG16 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix/vgg16_cm.png'))
    plt.close()

# ---------------- MAIN ----------------
def main():

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first.")
        return

    print("Loading dataset...")
    _, _, test_df = load_and_split_data(DATA_DIR)

    classes = sorted(test_df['label'].unique())

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    # ---------------- TEST ACCURACY ----------------
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # ---------------- PREDICTIONS ----------------
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # ---------------- CLASSIFICATION REPORT ----------------
    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    with open(os.path.join(RESULTS_DIR, 'reports/vgg16_report.txt'), 'w') as f:
        f.write(report)

    # ---------------- TRAIN + VAL ACCURACY ----------------
    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)

        final_train_acc = history['accuracy'].iloc[-1]
        final_val_acc = history['val_accuracy'].iloc[-1]

        print(f"\nFinal Training Accuracy: {final_train_acc*100:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")

        with open(os.path.join(RESULTS_DIR, 'reports/vgg16_accuracy.txt'), 'w') as f:
            f.write(f"Train Accuracy: {final_train_acc*100:.2f}%\n")
            f.write(f"Validation Accuracy: {final_val_acc*100:.2f}%\n")
            f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")

        plot_history(history)
    else:
        print("History file missing")

    # ---------------- CONFUSION MATRIX ----------------
    print("Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, classes)

    print("\n✅ ALL RESULTS SAVED IN results/ FOLDER")
    print("Case study outputs ready.")

if __name__ == "__main__":
    main()

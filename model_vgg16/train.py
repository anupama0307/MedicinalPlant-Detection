import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.data_utils import load_and_split_data, get_data_generators
import pandas as pd
import json

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../../data/Segmented Medicinal Leaf Images')
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20                 # increased epochs
LEARNING_RATE = 0.00001     # low LR for fine tuning

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'vgg16_best.h5')
HISTORY_SAVE_PATH = os.path.join(BASE_DIR, '../../results/vgg16_history.csv')
CLASS_INDEX_PATH = os.path.join(BASE_DIR, 'class_indices.json')

# ---------------- MODEL ----------------
def build_vgg16_model(num_classes):

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))


    for layer in base_model.layers[:-4]:
        layer.trainable = False


    for layer in base_model.layers[-4:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# ---------------- MAIN ----------------
def main():

    print("Loading dataset...")
    if not os.path.exists(DATA_DIR):
        print("Dataset not found:", DATA_DIR)
        return

    train_df, val_df, test_df = load_and_split_data(DATA_DIR)

    num_classes = len(train_df['label'].unique())
    print("Detected classes:", num_classes)

    # ---------------- GENERATORS ----------------
    train_gen, val_gen, test_gen = get_data_generators(
        train_df,
        val_df,
        test_df,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        preprocessing_function=preprocess_input
    )

    print("Train samples:", train_gen.samples)
    print("Val samples:", val_gen.samples)
    print("Test samples:", test_gen.samples)

    # save class indices (VERY IMPORTANT for prediction)
    class_indices = train_gen.class_indices
    print("Class indices:", class_indices)

    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(class_indices, f)

    # ---------------- MODEL ----------------
    print("Building VGG16...")
    model = build_vgg16_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ---------------- CALLBACKS ----------------
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH,
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=1),

        EarlyStopping(monitor='val_loss',
                      patience=7,
                      restore_best_weights=True,
                      verbose=1),

        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.3,
                          patience=3,
                          min_lr=1e-7,
                          verbose=1)
    ]

    # ---------------- TRAIN ----------------
    print("\nTraining started...\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # ---------------- SAVE HISTORY ----------------
    print("Saving history...")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_SAVE_PATH, index=False)

    print("\n VGG16 training complete.")
    print("Model saved at:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()

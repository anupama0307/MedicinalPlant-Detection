import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATA_DIR = r"c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split"

def load_data_efficientnet(preprocessing_function=None):
    """
    Dedicated Data Loader for EfficientNet.
    Enforces RGB color mode and handling.
    """
    print(f"Loading data from {DATA_DIR} [EfficientNet Mode]...")
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # EfficientNet expects 0-255 inputs if using its internal preprocessing, 
    # OR specific scaling if using 'preprocess_input' externally.
    # The 'preprocess_input' passed from efficientnet module usually handles scaling.
    # So we set rescale=None if function is present.
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

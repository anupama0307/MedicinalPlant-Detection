import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATA_DIR = r"c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split"

def load_data(preprocessing_function=None):
    """
    Loads data from train/val/test directories and creates data generators.
    Allows passing a specific preprocessing function (e.g. for ResNet).
    """
    print(f"Loading data from {DATA_DIR}...")
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # Determine rescale factor:
    # If using a specific preprocessing function (like ResNet's), typical rule is NO rescale (function handles it)
    # If NO function, standard is 1./255
    if preprocessing_function:
        rescale_factor = None
    else:
        rescale_factor = 1./255

    # Data Augmentation for Training
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
    
    # Validation/Test Datagen
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
        shuffle=True
    )
    
    print("Found validation images:")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Found test images:")
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, train_generator.class_indices

if __name__ == "__main__":
    load_data()

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
SOURCE_DIR = r"c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/Segmented Medicinal Leaf Images"
DEST_BASE_DIR = r"c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split"

def split_dataset():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_BASE_DIR)

    # 1. CLEANUP: Remove existing split directory to avoid duplicates
    if dest_path.exists():
        print(f"Removing existing directory: {DEST_BASE_DIR}")
        shutil.rmtree(DEST_BASE_DIR)
    
    # Create destination folders
    for split in ['train', 'val', 'test']:
        os.makedirs(dest_path / split, exist_ok=True)

    # Get all classes
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes.")

    for class_name in tqdm(classes, desc="Processing classes"):
        # Create class folders within splits
        for split in ['train', 'val', 'test']:
            os.makedirs(dest_path / split / class_name, exist_ok=True)

        # Get all unique image files
        class_dir = source_path / class_name
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(list(class_dir.glob(ext)))
        
        # Remove duplicates if any (by name) just in case
        images = list(set(images)) 
        
        if not images:
            print(f"Warning: No images found for class {class_name}")
            continue

        # Split images: 80% Train, 10% Val, 10% Test
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

        # Copy images
        for img in train_imgs:
            shutil.copy(img, dest_path / 'train' / class_name / img.name)
        
        for img in val_imgs:
            shutil.copy(img, dest_path / 'val' / class_name / img.name)
            
        for img in test_imgs:
            shutil.copy(img, dest_path / 'test' / class_name / img.name)

    print(f"\nDataset split complete!")
    print(f"Train/Val/Test data stored in: {DEST_BASE_DIR}")

if __name__ == "__main__":
    split_dataset()

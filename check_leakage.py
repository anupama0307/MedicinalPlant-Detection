import os
import hashlib
from pathlib import Path

DATA_DIR = r"c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split"

def get_file_hash(filepath):
    """Calculates MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_leakage():
    print("Checking for data leakage between TRAIN and TEST sets...")
    
    train_dir = Path(DATA_DIR) / 'train'
    test_dir = Path(DATA_DIR) / 'test'
    
    # Get all hashes from train set
    train_hashes = set()
    train_files = list(train_dir.glob('*/*'))
    print(f"Hashing {len(train_files)} training files...")
    for f in train_files:
        if f.is_file():
            train_hashes.add(get_file_hash(f))
            
    # Check test set against train hashes
    test_files = list(test_dir.glob('*/*'))
    print(f"Checking {len(test_files)} testing files...")
    
    leakage_count = 0
    for f in test_files:
        if f.is_file():
            h = get_file_hash(f)
            if h in train_hashes:
                print(f"LEAK FOUND: {f} is also in the training set!")
                leakage_count += 1
                
    if leakage_count == 0:
        print("\nSUCCESS: No identical files found between Train and Test sets.")
        print("The 100% accuracy is likely genuine (or due to very similar but not identical images).")
    else:
        print(f"\nFAILURE: Found {leakage_count} duplicate files.")

if __name__ == "__main__":
    check_leakage()

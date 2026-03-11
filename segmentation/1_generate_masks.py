import cv2
import numpy as np
import os

# Update these to your actual paths
IMAGE_DIR = r'c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/dataset_split/train'
MASK_DIR = r'c:/Users/roshn/OneDrive/Desktop/FINAL_NNDL/unet_masks'

def generate_masks():
    count = 0
    print("Starting mask generation...")

    if not os.path.exists(IMAGE_DIR):
        print(f"Error: IMAGE_DIR '{IMAGE_DIR}' does not exist.")
        return

    for root, dirs, files in os.walk(IMAGE_DIR):
        category = os.path.basename(root)
        
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                
                if img is None: continue

                # Convert to HSV to detect Green/Disease spots
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Refined range (25-95) to catch green and brown/yellow diseased bits
                mask = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
                
                # Morphological closing: Fills tiny holes inside the leaf mask
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Create category folder inside MASK_DIR
                category_mask_dir = os.path.join(MASK_DIR, category)
                os.makedirs(category_mask_dir, exist_ok=True)
                
                # Keep original filename so it maps exactly 1:1 with training images
                save_path = os.path.join(category_mask_dir, file)
                cv2.imwrite(save_path, mask)
                count += 1

    print(f"Mask generation complete! {count} masks created in {MASK_DIR}")

if __name__ == "__main__":
    generate_masks()

"""
Offline Data Augmentation Script for Semantic Segmentation
Generates augmented pairs (Image + Mask) and merges them with original data.

Features:
- Geometric Transforms (Flip) -> Applied to Image AND Mask
- Photometric Transforms (Brightness, Darkness, Blur) -> Applied to Image ONLY
- Preserves Directory Structure
- Multiprocessing for speed
"""

import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
import shutil
import multiprocessing
from functools import partial

# ============================================================================
# Configuration
# ============================================================================

INPUT_DIR = "data_train_test/Offroad_Segmentation_Training_Dataset"  # Your current dataset
OUTPUT_DIR = "data_train_test/Offroad_Segmentation_Training_Augmented_Dataset"            # New dataset location

# Define Augmentation Pipelines using Albumentations
# We define separate pipelines to control specific augmentation logic

def get_transforms():
    """
    Returns a dictionary of transform pipelines.
    Key: Suffix to append to filename
    Value: Albumentations Compose object
    """
    return {
        # 1. Geometric: Horizontal Flip
        "_flip_h": A.Compose([
            A.HorizontalFlip(p=1.0),
        ], is_check_shapes=False),

        # 2. Geometric: Vertical Flip (Useful for texture learning, though unrealistic physics)
        "_flip_v": A.Compose([
            A.VerticalFlip(p=1.0),
        ], is_check_shapes=False),

        # 3. Lighting: Brightness (Sunlight simulation)
        "_bright": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=0.1, p=1.0),
        ], is_check_shapes=False),

        # 4. Lighting: Darkness (Shadow/Dusk simulation)
        "_dark": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), contrast_limit=0.1, p=1.0),
        ], is_check_shapes=False),

        # 5. Blur: Light (Motion blur)
        "_blur_light": A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], is_check_shapes=False),

        # 6. Blur: Strong (Heat haze/Focus loss)
        "_blur_strong": A.Compose([
            A.GaussianBlur(blur_limit=(5, 9), p=1.0),
        ], is_check_shapes=False),
        
        # 7. Noise: Simulates sensor noise (Grain)
        "_noise": A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
        ], is_check_shapes=False)
    }

# ============================================================================
# Processing Logic
# ============================================================================

def process_single_pair(file_info):
    """
    Process one image/mask pair: Copy original, then generate augmentations.
    """
    filename, img_dir, mask_dir, out_img_dir, out_mask_dir, transforms_dict = file_info

    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename) # Assuming mask has same filename

    # Check if files exist
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return

    # 1. Load Data
    # Read Image (RGB)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read Mask (Grayscale/Indexed) - DO NOT CHANGE VALUES
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # 2. Save Original (Copy)
    base_name, ext = os.path.splitext(filename)
    
    # Save Image
    Image.fromarray(image).save(os.path.join(out_img_dir, filename))
    # Save Mask (Use PIL to preserve exact pixel values, prevent compression artifacts)
    Image.fromarray(mask).save(os.path.join(out_mask_dir, filename))

    # 3. Generate Augmentations
    for suffix, transform in transforms_dict.items():
        # Apply Transform
        # Albumentations handles mask flipping automatically if passed as 'mask'
        augmented = transform(image=image, mask=mask)
        
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        # Construct new filename
        new_filename = f"{base_name}{suffix}{ext}"

        # Save Augmented Image
        Image.fromarray(aug_img).save(os.path.join(out_img_dir, new_filename))
        
        # Save Augmented Mask
        # Note: Lighting/Blur transforms do not change mask values, which is correct.
        # Geometric transforms (Flips) DO change mask positions, which is correct.
        Image.fromarray(aug_mask).save(os.path.join(out_mask_dir, new_filename))

def main():
    print("Starting Offline Augmentation...")
    
    # Get transforms
    transforms_dict = get_transforms()
    
    # Process both Train and Val splits
    for split in ['train', 'val']:
        print(f"\nProcessing split: {split}")
        
        # Source Paths
        src_img_dir = os.path.join(INPUT_DIR, split, 'Color_Images')
        src_mask_dir = os.path.join(INPUT_DIR, split, 'Segmentation')
        
        # Destination Paths
        dst_img_dir = os.path.join(OUTPUT_DIR, split, 'Color_Images')
        dst_mask_dir = os.path.join(OUTPUT_DIR, split, 'Segmentation')
        
        # Create Directories
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_mask_dir, exist_ok=True)
        
        # List files
        files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(files)} original images.")
        
        # Prepare arguments for multiprocessing
        # We assume mask has same filename as image (standard structure)
        tasks = []
        for f in files:
            tasks.append((f, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, transforms_dict))
        
        # Run Multiprocessing
        # Adjust cpu_count() fraction if your PC lags, e.g., cpu_count() // 2
        num_workers = max(1, multiprocessing.cpu_count() - 2) 
        
        print(f"Augmenting with {num_workers} workers...")
        with multiprocessing.Pool(num_workers) as pool:
            # Use tqdm to show progress bar
            list(tqdm(pool.imap_unordered(process_single_pair, tasks), total=len(tasks)))
            
    print("\n" + "="*50)
    print(f"Augmentation Complete!")
    print(f"Dataset saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("Structure matched. You can point your training script to this new folder.")
    print("="*50)

if __name__ == "__main__":
    main()
"""
Unified Testing Script for V1, V2, V3 Models
PyTorch Implementation

Loads trained models and evaluates on test dataset
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== CONFIG ====================
TEST_CONFIG = {
    "input_size": 512,
    "num_classes": 10,
    "batch_size": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "test_dir": "/Users/abhayatrivedi/Downloads/test3",
    "images_dir": "/Users/abhayatrivedi/Downloads/test3/images",
    "labels_dir": "/Users/abhayatrivedi/Downloads/test3/labels",
    "checkpoints": {
        "v1": "./checkpoints/v1_best.pth",
        "v2": "./checkpoints/v2_best.pth",
        "v3": "./checkpoints/v3_best.pth",
    },
    "output_dir": "./test_results",
}

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers",
    "Logs", "Rocks", "Sky"
]

COLOR_MAP = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [50, 205, 50],
    [210, 180, 140],
    [139, 69, 19],
    [0, 0, 255],
    [255, 192, 203],
    [101, 67, 33],
    [128, 128, 128],
    [135, 206, 235],
], dtype=np.uint8)

# ==================== DATASET ====================
class TestDataset(Dataset):
    """Test dataset loader"""
    
    def __init__(self, images_dir, labels_dir, input_size=512):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        if os.path.exists(label_path):
            mask = np.loadtxt(label_path, dtype=np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        original_shape = mask.shape
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask, img_name, original_shape

# ==================== MODEL LOADING ====================
def load_model_v1(checkpoint_path, device):
    """Load V1 Model: UNet ResNet50"""
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=10,
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"✓ V1 model loaded from {checkpoint_path}")
    else:
        print(f"✗ V1 checkpoint not found: {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    return model

def load_model_v2(checkpoint_path, device):
    """Load V2 Model: DeepLabV3+ ResNet50"""
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=10,
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"✓ V2 model loaded from {checkpoint_path}")
    else:
        print(f"✗ V2 checkpoint not found: {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    return model

def load_model_v3(checkpoint_path, device):
    """Load V3 Model: Hybrid Transformer"""
    try:
        from train_v3 import HybridTransformerSegmenter
        
        model = HybridTransformerSegmenter(10, device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            print(f"✓ V3 model loaded from {checkpoint_path}")
        else:
            print(f"✗ V3 checkpoint not found: {checkpoint_path}")
            return None
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"✗ Error loading V3 model: {e}")
        return None

# ==================== METRICS ====================
def calculate_metrics(pred, target, num_classes):
    """Calculate IoU, Dice, Accuracy"""
    
    pred = pred.numpy()
    target = target.numpy()
    
    # Pixel Accuracy
    accuracy = (pred == target).sum() / target.size
    
    # Per-class IoU
    ious = []
    dices = []
    
    for class_id in range(num_classes):
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        
        # Dice
        if (pred_mask.sum() + target_mask.sum()) > 0:
            dice = 2 * intersection / (pred_mask.sum() + target_mask.sum())
            dices.append(dice)
    
    miou = np.mean(ious) if ious else 0
    mdice = np.mean(dices) if dices else 0
    
    return {
        "accuracy": accuracy,
        "miou": miou,
        "mdice": mdice,
        "per_class_iou": ious,
    }

# ==================== INFERENCE ====================
def run_inference(model, loader, device, model_name, output_dir):
    """Run inference on test dataset"""
    
    if model is None:
        return None
    
    print(f"\n[{model_name}] Running inference...")
    
    model.eval()
    results = []
    all_metrics = {
        "accuracy": [],
        "miou": [],
        "mdice": [],
    }
    
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Testing {model_name}", leave=False)
        for images, masks, names, orig_shapes in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu()
            
            # Resize back to original
            for i, (pred, mask, name, orig_shape) in enumerate(
                    zip(predictions, masks, names, orig_shapes)):
                
                pred_resized = cv2.resize(
                    pred.numpy().astype(np.uint8),
                    (orig_shape[1], orig_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                mask_resized = cv2.resize(
                    mask.cpu().numpy().astype(np.uint8),
                    (orig_shape[1], orig_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Calculate metrics
                metrics = calculate_metrics(pred_resized, mask_resized, 10)
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                # Save visualization
                save_visualization(pred_resized, mask_resized, name, 
                                 os.path.join(output_dir, model_name))
                
                results.append({
                    "image": name,
                    "accuracy": metrics["accuracy"],
                    "miou": metrics["miou"],
                    "mdice": metrics["mdice"],
                })
    
    # Aggregate results
    avg_metrics = {
        "accuracy": np.mean(all_metrics["accuracy"]),
        "miou": np.mean(all_metrics["miou"]),
        "mdice": np.mean(all_metrics["mdice"]),
    }
    
    print(f"\n[{model_name}] Results:")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  mIoU:     {avg_metrics['miou']:.4f}")
    print(f"  mDice:    {avg_metrics['mdice']:.4f}")
    
    return results, avg_metrics

def save_visualization(pred, mask, name, output_dir):
    """Save prediction visualization"""
    
    pred_rgb = COLOR_MAP[pred]
    mask_rgb = COLOR_MAP[mask]
    
    # Create side-by-side comparison
    comparison = np.hstack([pred_rgb, mask_rgb])
    
    output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_comparison.png")
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

# ==================== MAIN ====================
def main():
    os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)
    device = torch.device(TEST_CONFIG["device"])
    
    print("=" * 60)
    print("SEMANTIC SEGMENTATION TEST - V1, V2, V3")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Test directory: {TEST_CONFIG['images_dir']}\n")
    
    # Verify test directories exist
    for path in [TEST_CONFIG["images_dir"], TEST_CONFIG["labels_dir"]]:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. Creating...")
            os.makedirs(path, exist_ok=True)
    
    # Load test dataset
    test_dataset = TestDataset(
        TEST_CONFIG["images_dir"],
        TEST_CONFIG["labels_dir"],
        TEST_CONFIG["input_size"]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    if len(test_dataset) == 0:
        print("ERROR: No test images found!")
        return
    
    # Load models
    print("Loading models...")
    model_v1 = load_model_v1(TEST_CONFIG["checkpoints"]["v1"], device)
    model_v2 = load_model_v2(TEST_CONFIG["checkpoints"]["v2"], device)
    model_v3 = load_model_v3(TEST_CONFIG["checkpoints"]["v3"], device)
    
    # Run inference
    all_results = {}
    
    if model_v1:
        results_v1, metrics_v1 = run_inference(model_v1, test_loader, device, "V1", 
                                               TEST_CONFIG["output_dir"])
        all_results["V1"] = {"results": results_v1, "metrics": metrics_v1}
    
    if model_v2:
        results_v2, metrics_v2 = run_inference(model_v2, test_loader, device, "V2",
                                               TEST_CONFIG["output_dir"])
        all_results["V2"] = {"results": results_v2, "metrics": metrics_v2}
    
    if model_v3:
        results_v3, metrics_v3 = run_inference(model_v3, test_loader, device, "V3",
                                               TEST_CONFIG["output_dir"])
        all_results["V3"] = {"results": results_v3, "metrics": metrics_v3}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for version, data in all_results.items():
        metrics = data["metrics"]
        print(f"\n{version}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  mIoU:     {metrics['miou']:.4f}")
        print(f"  mDice:    {metrics['mdice']:.4f}")
    
    # Save results
    results_path = os.path.join(TEST_CONFIG["output_dir"], "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Visualizations saved to {TEST_CONFIG['output_dir']}")

if __name__ == "__main__":
    main()

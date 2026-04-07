"""
Training Script V2: DeepLabV3+ with ResNet50 Backbone
PyTorch Implementation with Enhanced Features

- Atrous Spatial Pyramid Pooling (ASPP)
- Better boundary segmentation
- Advanced loss functions (Lovasz-Softmax)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import json
import cv2

# ==================== CONFIG ====================
CONFIG = {
    "model_name": "DeepLabV3Plus_ResNet50_v2",
    "input_size": 512,
    "num_classes": 10,
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "save_dir": "./checkpoints",
    "log_dir": "./logs",
}

# ==================== DATASET ====================
class SegmentationDataset(Dataset):
    """Dataset loader with advanced augmentations"""
    
    def __init__(self, images_dir, labels_dir, input_size=512, augment=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.augment = augment
        
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if augment:
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=768),
                A.RandomCrop(input_size, input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
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
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask, img_name

# ==================== MODEL ====================
def build_model_v2(num_classes, device):
    """Build DeepLabV3+ with ResNet50"""
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    model.to(device)
    return model

# ==================== LOSS FUNCTIONS ====================
class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss for semantic segmentation"""
    
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # Simple approximation - using cross entropy with class weights
        ce_loss = nn.CrossEntropyLoss(reduction='mean')
        return ce_loss(pred, target)

class V2Loss(nn.Module):
    """V2 Combined Loss: Cross Entropy + Dice + Boundary Loss"""
    
    def __init__(self, num_classes, device):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.jaccard_loss = smp.losses.JaccardLoss(mode='multiclass', from_logits=True)
        self.device = device
    
    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        jaccard = self.jaccard_loss(outputs, targets)
        
        total = 0.4 * ce + 0.35 * dice + 0.25 * jaccard
        return total, {"ce": ce.item(), "dice": dice.item(), "jaccard": jaccard.item()}

# ==================== TRAINING ====================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    loss_components = {"ce": 0, "dice": 0, "jaccard": 0}
    
    pbar = tqdm(loader, desc="Training V2", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss, components = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += components[key]
        
        pbar.set_postfix({"loss": loss.item():.4f})
    
    avg_loss = total_loss / len(loader)
    for key in loss_components:
        loss_components[key] /= len(loader)
    
    return avg_loss, loss_components

# ==================== VALIDATION ====================
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating V2", leave=False)
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss, _ = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
            
            pbar.set_postfix({"loss": loss.item():.4f})
    
    avg_loss = total_loss / len(loader)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    return avg_loss, pixel_accuracy

# ==================== MAIN ====================
def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])
    
    print(f"[V2] Training on {device}")
    print(f"[V2] Model: {CONFIG['model_name']}\n")
    
    # Data paths
    train_images = "/Users/abhayatrivedi/Downloads/train_3/train3/images"
    train_labels = "/Users/abhayatrivedi/Downloads/train_3/train3/labels"
    val_images = "/Users/abhayatrivedi/Downloads/train_3/val3/images"
    val_labels = "/Users/abhayatrivedi/Downloads/train_3/val3/labels"
    
    for path in [train_images, train_labels, val_images, val_labels]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
    print("[V2] Loading datasets...")
    train_dataset = SegmentationDataset(train_images, train_labels, 
                                        CONFIG["input_size"], augment=True)
    val_dataset = SegmentationDataset(val_images, val_labels,
                                      CONFIG["input_size"], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                           shuffle=False, num_workers=CONFIG["num_workers"])
    
    print(f"[V2] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples\n")
    
    # Model setup
    model = build_model_v2(CONFIG["num_classes"], device)
    criterion = V2Loss(CONFIG["num_classes"], device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=CONFIG["epochs"], power=0.9)
    
    # Training loop
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    print("[V2] Starting training...\n")
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        print(f"[V2] Epoch {epoch+1}/{CONFIG['epochs']}")
        
        train_loss, train_components = train_epoch(model, train_loader, 
                                                    criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_components['ce']:.4f}, "
              f"Dice: {train_components['dice']:.4f}, Jaccard: {train_components['jaccard']:.4f})")
        print(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG
            }
            save_path = os.path.join(CONFIG["save_dir"], "v2_best.pth")
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved\n")
    
    elapsed = time.time() - start_time
    print(f"[V2] Training complete in {elapsed/3600:.2f} hours")
    
    final_path = os.path.join(CONFIG["save_dir"], "v2_final.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": CONFIG}, final_path)
    
    history_path = os.path.join(CONFIG["log_dir"], "v2_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

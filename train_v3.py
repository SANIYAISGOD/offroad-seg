"""
Training Script V3: Hybrid Transformer-CNN (DINOv2 + CNN Pyramid)
PyTorch Implementation as described in README

- DINOv2 Vision Transformer backbone
- Manual CNN Feature Pyramid Neck
- Multi-scale deep supervision
- Self-supervised pretrained features
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import json
import cv2
from collections import OrderedDict

# ==================== CONFIG ====================
CONFIG = {
    "model_name": "HybridTransformer_DINOv2_v3",
    "input_size": 512,
    "num_classes": 10,
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "save_dir": "./checkpoints",
    "log_dir": "./logs",
}

# ==================== DATASET ====================
class SegmentationDataset(Dataset):
    """Dataset with aggressive augmentations for transformer robustness"""
    
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
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Transpose(p=0.2),
                A.RandomBrightnessContrast(p=0.4, brightness_limit=0.3, contrast_limit=0.3),
                A.GaussNoise(p=0.3, var_limit=(10.0, 50.0)),
                A.Blur(blur_limit=5, p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
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

# ==================== HYBRID TRANSFORMER MODEL ====================
class HybridTransformerSegmenter(nn.Module):
    """
    Hybrid Transformer-CNN Architecture for semantic segmentation
    - DINOv2 backbone (if available, fallback to standard ViT)
    - Manual CNN Feature Pyramid Neck
    - Multi-scale supervision heads
    """
    
    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        # Try to use DINOv2, fallback to standard ViT
        try:
            print("[V3] Loading DINOv2 backbone...")
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.backbone_channels = 768
            self.patch_size = 14
        except:
            print("[V3] DINOv2 not available, using standard ViT backbone")
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(weights='IMAGENET1K_V1')
            self.backbone_channels = 768
            self.patch_size = 16
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Feature pyramid construction
        self.pyramid_layers = nn.ModuleDict({
            'p2': nn.Sequential(
                nn.ConvTranspose2d(self.backbone_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'p3': nn.Sequential(
                nn.Conv2d(self.backbone_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'p4': nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(self.backbone_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'p5': nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Conv2d(self.backbone_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
        })
        
        # Segmentation heads
        self.head_p2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.head_p3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.head_p4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.head_p5 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Final fusion head
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self.to(device)
    
    def extract_features(self, x):
        """Extract features from backbone"""
        B, C, H, W = x.shape
        
        # For DINOv2 / ViT
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            # Standard ViT forward
            x_prep = self.backbone._process_input(x)
            n, _, h, w = x_prep.shape
            p = self.backbone.patch_embed.patch_size
            
            x_prep = self.backbone.patch_embed(x_prep)
            x_prep = self.backbone._pos_embed(x_prep)
            x_prep = self.backbone.dropout(x_prep)
            x_prep = self.backbone.layers(x_prep)
            x_prep = self.backbone.ln(x_prep)
            
            features = x_prep[:, 1:, :]  # Remove class token
            features = features.reshape(B, h // p, w // p, -1).permute(0, 3, 1, 2)
        
        return features
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Extract backbone features
        features = self.extract_features(x)
        # Ensure features are 4D: [B, C, H', W']
        if features.dim() == 3:
            # Reshape if needed
            seq_len = features.shape[1]
            side = int(np.sqrt(seq_len))
            features = features[:, 1:, :].reshape(B, side, side, -1).permute(0, 3, 1, 2)
        
        # Build pyramid
        p2 = self.pyramid_layers['p2'](features)
        p3 = self.pyramid_layers['p3'](features)
        p4 = self.pyramid_layers['p4'](features)
        p5 = self.pyramid_layers['p5'](features)
        
        # Multi-scale supervision
        out_p2 = self.head_p2(p2)
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)
        
        # Upsample all to same resolution as p2
        h, w = p2.shape[2:]
        out_p3_up = nn.functional.interpolate(out_p3, size=(h, w), mode='bilinear', align_corners=False)
        out_p4_up = nn.functional.interpolate(out_p4, size=(h, w), mode='bilinear', align_corners=False)
        out_p5_up = nn.functional.interpolate(out_p5, size=(h, w), mode='bilinear', align_corners=False)
        
        # Fusion
        fused = torch.cat([p2, p3, p4, p5], dim=1)
        output = self.fusion(fused)
        
        # Upsample to original resolution
        output = nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output

# ==================== LOSS FUNCTIONS ====================
class V3Loss(nn.Module):
    """V3 Loss: Cross Entropy + Dice + Deep supervision"""
    
    def __init__(self, num_classes, device):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', from_logits=True)
        self.device = device
    
    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        
        total = 0.4 * ce + 0.4 * dice + 0.2 * focal
        return total, {"ce": ce.item(), "dice": dice.item(), "focal": focal.item()}

# ==================== TRAINING ====================
def train_epoch(model, loader, criterion, optimizer, device, unfreeze_epoch=20, current_epoch=0):
    """Train one epoch with progressive unfreezing"""
    model.train()
    
    # Progressive unfreezing
    if current_epoch == unfreeze_epoch:
        print("[V3] Unfreezing backbone weights...")
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    total_loss = 0
    loss_components = {"ce": 0, "dice": 0, "focal": 0}
    
    pbar = tqdm(loader, desc="Training V3", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss, components = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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
        pbar = tqdm(loader, desc="Validating V3", leave=False)
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
    
    print(f"[V3] Training on {device}")
    print(f"[V3] Model: {CONFIG['model_name']}\n")
    
    # Data paths
    train_images = "/Users/abhayatrivedi/Downloads/train_3/train3/images"
    train_labels = "/Users/abhayatrivedi/Downloads/train_3/train3/labels"
    val_images = "/Users/abhayatrivedi/Downloads/train_3/val3/images"
    val_labels = "/Users/abhayatrivedi/Downloads/train_3/val3/labels"
    
    for path in [train_images, train_labels, val_images, val_labels]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
    print("[V3] Loading datasets...")
    train_dataset = SegmentationDataset(train_images, train_labels, 
                                        CONFIG["input_size"], augment=True)
    val_dataset = SegmentationDataset(val_images, val_labels,
                                      CONFIG["input_size"], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                           shuffle=False, num_workers=CONFIG["num_workers"])
    
    print(f"[V3] Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
    
    # Model setup
    model = HybridTransformerSegmenter(CONFIG["num_classes"], device)
    criterion = V3Loss(CONFIG["num_classes"], device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    print("[V3] Starting training (Backbone frozen until epoch 20)...\n")
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        print(f"[V3] Epoch {epoch+1}/{CONFIG['epochs']}")
        
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            unfreeze_epoch=20, current_epoch=epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_components['ce']:.4f}, "
              f"Dice: {train_components['dice']:.4f})")
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
            save_path = os.path.join(CONFIG["save_dir"], "v3_best.pth")
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved\n")
    
    elapsed = time.time() - start_time
    print(f"[V3] Training complete in {elapsed/3600:.2f} hours")
    
    final_path = os.path.join(CONFIG["save_dir"], "v3_final.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": CONFIG}, final_path)
    
    history_path = os.path.join(CONFIG["log_dir"], "v3_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

"""
DINOv2-Attention-UNet COMPREHENSIVE Inference Script

Features:
✅ Layer-by-Layer Visualization (Encoder embeddings + Decoder progression)
✅ Class-Wise Pixel-Level Metrics (IoU, Dice, Precision, Recall per class)
✅ Overall Pixel-Wise Metrics (NOT averaged by class)
✅ Attention Map Visualization
✅ 3-Panel Standard Visualization

Output Structure:
- visualizations/standard/          → Input | GT | Prediction
- visualizations/layer_by_layer/    → Full pipeline visualization
- visualizations/attention_maps/    → Attention gate activations
- metrics/per_image_metrics.csv     → All metrics per image
- metrics/class_wise_summary.csv    → Aggregated class statistics
- metrics/summary.txt               → Final report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel
from PIL import Image
import numpy as np
import os
import csv
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ============================================================================
# CONFIGURATION (UPDATE THESE PATHS)
# ============================================================================

# Model Weights
ATTN_UNET_WEIGHTS = "models/best_model_payload_attn_unet_epoc2.pth"

# Test Data
TEST_DATA_DIR = "demo_test"

# Output Directory
OUTPUT_DIR = "demo_inference_results"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Input Size
MODEL_INPUT_SIZE = (448, 448)

# Number of classes
NUM_CLASSES = 10

# Class Names (EXACT ORDER FROM TRAINING)
CLASS_NAMES = [
    "Background",      # 0
    "Trees",          # 1
    "Lush Bushes",    # 2
    "Dry Grass",      # 3
    "Dry Bushes",     # 4
    "Ground Clutter", # 5
    "Logs",           # 6
    "Rocks",          # 7
    "Landscape",      # 8
    "Sky"             # 9
]

# How many samples to create layer-by-layer visualizations for (saves time)
NUM_LAYER_VIZ_SAMPLES = 20  # Set to 0 to disable, or -1 for all

# ============================================================================
# 1. ATTENTION UNET ARCHITECTURE (WITH INTERMEDIATE OUTPUT HOOKS)
# ============================================================================

class AttentionGate(nn.Module):
    """Attention Gate Module with attention map output"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.last_attention_map = None  # Store for visualization
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Store attention map
        self.last_attention_map = psi.detach()
        
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DinoAttentionUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Frozen DINOv2-Large Encoder
        self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.embed_dim = 1024
        
        # Decoder with Attention Gates
        self.bot = DoubleConv(self.embed_dim, 512)
        
        # Level 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attn1 = AttentionGate(F_g=512, F_l=self.embed_dim, F_int=512)
        self.conv1 = DoubleConv(512 + self.embed_dim, 512)
        
        # Level 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attn2 = AttentionGate(F_g=512, F_l=self.embed_dim, F_int=256)
        self.conv2 = DoubleConv(512 + self.embed_dim, 256)
        
        # Level 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attn3 = AttentionGate(F_g=256, F_l=self.embed_dim, F_int=128)
        self.conv3 = DoubleConv(256 + self.embed_dim, 128)
        
        # Level 4
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attn4 = AttentionGate(F_g=128, F_l=self.embed_dim, F_int=64)
        self.conv4 = DoubleConv(128 + self.embed_dim, 64)
        
        # Final upsampling and output
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(64, n_classes, 1)
        
        # Storage for intermediate outputs
        self.intermediates = {}
    
    def forward(self, x, save_intermediates=False):
        B, C, H, W = x.shape
        
        if save_intermediates:
            self.intermediates = {'input': x.detach().cpu()}
        
        # Extract multi-scale features from DINOv2
        with torch.no_grad():
            outputs = self.encoder(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            h_grid = H // 14
            w_grid = W // 14
            
            def reshape_feat(f):
                f = f[:, 1:, :]
                return f.permute(0, 2, 1).reshape(B, self.embed_dim, h_grid, w_grid)
            
            # Extract 4 encoder levels
            s1 = reshape_feat(hidden_states[7])   # Shallow
            s2 = reshape_feat(hidden_states[11])  # Mid-shallow
            s3 = reshape_feat(hidden_states[15])  # Mid-deep
            s4 = reshape_feat(hidden_states[23])  # Deep
        
        if save_intermediates:
            self.intermediates['encoder_s1'] = s1.detach().cpu()
            self.intermediates['encoder_s2'] = s2.detach().cpu()
            self.intermediates['encoder_s3'] = s3.detach().cpu()
            self.intermediates['encoder_s4'] = s4.detach().cpu()
        
        # Decoder with attention-gated skip connections
        x = self.bot(s4)
        
        if save_intermediates:
            self.intermediates['bottleneck'] = x.detach().cpu()
        
        # Level 1
        x = self.up1(x)
        s3 = F.interpolate(s3, size=x.shape[2:], mode='bilinear', align_corners=False)
        s3_att = self.attn1(g=x, x=s3)
        x = torch.cat([x, s3_att], dim=1)
        x = self.conv1(x)
        
        if save_intermediates:
            self.intermediates['decoder_level1'] = x.detach().cpu()
            self.intermediates['attention_map1'] = self.attn1.last_attention_map.cpu()
        
        # Level 2
        x = self.up2(x)
        s2 = F.interpolate(s2, size=x.shape[2:], mode='bilinear', align_corners=False)
        s2_att = self.attn2(g=x, x=s2)
        x = torch.cat([x, s2_att], dim=1)
        x = self.conv2(x)
        
        if save_intermediates:
            self.intermediates['decoder_level2'] = x.detach().cpu()
            self.intermediates['attention_map2'] = self.attn2.last_attention_map.cpu()
        
        # Level 3
        x = self.up3(x)
        s1 = F.interpolate(s1, size=x.shape[2:], mode='bilinear', align_corners=False)
        s1_att = self.attn3(g=x, x=s1)
        x = torch.cat([x, s1_att], dim=1)
        x = self.conv3(x)
        
        if save_intermediates:
            self.intermediates['decoder_level3'] = x.detach().cpu()
            self.intermediates['attention_map3'] = self.attn3.last_attention_map.cpu()
        
        # Level 4
        x = self.up4(x)
        s1_up = F.interpolate(s1, size=x.shape[2:], mode='bilinear', align_corners=False)
        s1_up_att = self.attn4(g=x, x=s1_up)
        x = torch.cat([x, s1_up_att], dim=1)
        x = self.conv4(x)
        
        if save_intermediates:
            self.intermediates['decoder_level4'] = x.detach().cpu()
            self.intermediates['attention_map4'] = self.attn4.last_attention_map.cpu()
        
        # Final output
        x = self.final_up(x)
        logits = self.outc(x)
        
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        if save_intermediates:
            self.intermediates['logits'] = logits.detach().cpu()
        
        return logits

# ============================================================================
# 2. COLOR MAP & UTILS
# ============================================================================

COLOR_PALETTE = np.array([
    [0, 0, 0],       # 0: Background
    [34, 139, 34],   # 1: Trees
    [50, 205, 50],   # 2: Lush Bushes
    [210, 180, 140], # 3: Dry Grass
    [139, 69, 19],   # 4: Dry Bushes
    [128, 128, 0],   # 5: Ground Clutter
    [160, 82, 45],   # 6: Logs
    [128, 128, 128], # 7: Rocks
    [244, 164, 96],  # 8: Landscape
    [135, 206, 235]  # 9: Sky
], dtype=np.uint8)

value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

def convert_mask_id(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

def colorize_mask(mask_2d):
    h, w = mask_2d.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(NUM_CLASSES):
        color_img[mask_2d == cid] = COLOR_PALETTE[cid]
    return color_img

# ============================================================================
# 3. PIXEL-WISE METRICS (CLASS-WISE + OVERALL)
# ============================================================================

def calculate_classwise_pixelwise_metrics(pred, target, num_classes):
    """
    Calculate PIXEL-WISE metrics for EACH class
    Returns: per-class IoU, Dice, Precision, Recall
    """
    pred = pred.flatten()
    target = target.flatten()
    
    class_metrics = []
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        # True Positives, False Positives, False Negatives (pixel-wise)
        tp = (pred_inds & target_inds).sum()
        fp = (pred_inds & ~target_inds).sum()
        fn = (~pred_inds & target_inds).sum()
        
        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        # Dice
        dice = (2.0 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        class_metrics.append({
            'class_id': cls,
            'class_name': CLASS_NAMES[cls],
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'total_pixels': int(target_inds.sum())
        })
    
    return class_metrics

def calculate_overall_pixelwise_metrics(pred, target):
    """
    Calculate OVERALL pixel-wise accuracy
    NOT averaged by class, pure pixel-level metrics
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Overall pixel accuracy
    correct_pixels = (pred == target).sum()
    total_pixels = len(target)
    pixel_accuracy = correct_pixels / total_pixels
    
    return {
        'pixel_accuracy': float(pixel_accuracy),
        'correct_pixels': int(correct_pixels),
        'total_pixels': int(total_pixels)
    }

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================

def create_standard_visualization(orig_img, gt_color, pred_color, fname):
    """3-Panel: Input | GT | Prediction"""
    h, w, _ = orig_img.shape
    pad_width = 15
    header_height = 60
    
    total_w = (w * 3) + (pad_width * 2)
    total_h = h + header_height
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    x_positions = [0, w + pad_width, (w + pad_width) * 2]
    
    canvas[header_height:, x_positions[0]:x_positions[0]+w, :] = orig_img
    canvas[header_height:, x_positions[1]:x_positions[1]+w, :] = gt_color
    canvas[header_height:, x_positions[2]:x_positions[2]+w, :] = pred_color
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thick = 2
    text_color = (0, 0, 0)
    
    def put_centered_text(text, center_x):
        text_size = cv2.getTextSize(text, font, font_scale, font_thick)[0]
        text_x = center_x - (text_size[0] // 2)
        text_y = (header_height // 2) + (text_size[1] // 2)
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, text_color, font_thick)
    
    put_centered_text("Input", x_positions[0] + w//2)
    put_centered_text("Ground Truth", x_positions[1] + w//2)
    put_centered_text("Prediction", x_positions[2] + w//2)
    
    return canvas

def visualize_feature_map(feature_tensor, orig_img_shape):
    """
    Visualize multi-channel feature map as heatmap overlay
    feature_tensor: [C, H, W]
    """
    # Average across channels
    if feature_tensor.dim() == 3:
        heatmap = feature_tensor.mean(dim=0).numpy()
    else:
        heatmap = feature_tensor.numpy()
    
    # Normalize to 0-255
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (orig_img_shape[1], orig_img_shape[0]))
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    return heatmap_color

def create_layer_by_layer_visualization(model_intermediates, orig_img, gt_mask, pred_mask, fname):
    """
    Create comprehensive layer-by-layer visualization
    
    Layout:
    Row 1: Input | Encoder L1 | Encoder L2 | Encoder L3 | Encoder L4
    Row 2: Bottleneck | Decoder L1 | Decoder L2 | Decoder L3 | Decoder L4
    Row 3: Attention Map 1 | Attention Map 2 | Attention Map 3 | Attention Map 4 | Final Output
    Row 4: Ground Truth | Prediction | Overlay
    """
    
    # Denormalize input
    input_tensor = model_intermediates['input'][0]  # [3, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_denorm = input_tensor * std + mean
    input_img = (input_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    input_img = cv2.resize(input_img, (orig_img.shape[1], orig_img.shape[0]))
    
    orig_shape = orig_img.shape[:2]
    
    # Process encoder features
    enc_viz = []
    for key in ['encoder_s1', 'encoder_s2', 'encoder_s3', 'encoder_s4']:
        feat = model_intermediates[key][0]  # [C, H, W]
        viz = visualize_feature_map(feat, orig_shape)
        enc_viz.append(viz)
    
    # Process bottleneck
    bottleneck = model_intermediates['bottleneck'][0]
    bottleneck_viz = visualize_feature_map(bottleneck, orig_shape)
    
    # Process decoder features
    dec_viz = []
    for key in ['decoder_level1', 'decoder_level2', 'decoder_level3', 'decoder_level4']:
        feat = model_intermediates[key][0]
        viz = visualize_feature_map(feat, orig_shape)
        dec_viz.append(viz)
    
    # Process attention maps
    attn_viz = []
    for key in ['attention_map1', 'attention_map2', 'attention_map3', 'attention_map4']:
        attn = model_intermediates[key][0, 0]  # [H, W]
        viz = visualize_feature_map(attn, orig_shape)
        attn_viz.append(viz)
    
    # Final output
    gt_color = colorize_mask(gt_mask)
    pred_color = colorize_mask(pred_mask)
    
    # Overlay prediction on input (50% alpha blend)
    overlay = cv2.addWeighted(orig_img, 0.6, pred_color, 0.4, 0)
    
    # Create grid
    cell_h, cell_w = 224, 224  # Each cell size
    pad = 10
    
    # 4 rows, 5 columns
    grid_h = 4 * cell_h + 3 * pad
    grid_w = 5 * cell_w + 4 * pad
    
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    def resize_and_place(img, row, col):
        img_resized = cv2.resize(img, (cell_w, cell_h))
        y_start = row * (cell_h + pad)
        x_start = col * (cell_w + pad)
        canvas[y_start:y_start+cell_h, x_start:x_start+cell_w] = img_resized
    
    # Row 0: Input + Encoder
    resize_and_place(input_img, 0, 0)
    for i, viz in enumerate(enc_viz):
        resize_and_place(viz, 0, i+1)
    
    # Row 1: Bottleneck + Decoder
    resize_and_place(bottleneck_viz, 1, 0)
    for i, viz in enumerate(dec_viz):
        resize_and_place(viz, 1, i+1)
    
    # Row 2: Attention Maps + Final Output placeholder
    for i, viz in enumerate(attn_viz):
        resize_and_place(viz, 2, i)
    resize_and_place(pred_color, 2, 4)
    
    # Row 3: GT, Pred, Overlay
    resize_and_place(gt_color, 3, 0)
    resize_and_place(pred_color, 3, 1)
    resize_and_place(overlay, 3, 2)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thick = 1
    text_color = (0, 0, 0)
    
    labels = [
        # Row 0
        (0, 0, "Input"),
        (0, 1, "Enc L1 (Layer 7)"),
        (0, 2, "Enc L2 (Layer 11)"),
        (0, 3, "Enc L3 (Layer 15)"),
        (0, 4, "Enc L4 (Layer 23)"),
        # Row 1
        (1, 0, "Bottleneck"),
        (1, 1, "Dec L1 (64x64)"),
        (1, 2, "Dec L2 (128x128)"),
        (1, 3, "Dec L3 (224x224)"),
        (1, 4, "Dec L4 (448x448)"),
        # Row 2
        (2, 0, "Attention Map 1"),
        (2, 1, "Attention Map 2"),
        (2, 2, "Attention Map 3"),
        (2, 3, "Attention Map 4"),
        (2, 4, "Final Logits"),
        # Row 3
        (3, 0, "Ground Truth"),
        (3, 1, "Prediction"),
        (3, 2, "Overlay (Pred on Input)"),
    ]
    
    for row, col, text in labels:
        y_pos = row * (cell_h + pad) + 15
        x_pos = col * (cell_w + pad) + 5
        cv2.putText(canvas, text, (x_pos, y_pos), font, font_scale, text_color, font_thick)
    
    return canvas

def create_attention_map_visualization(model_intermediates, orig_img):
    """
    Create focused visualization of all 4 attention maps
    """
    orig_shape = orig_img.shape[:2]
    
    attn_maps = []
    for key in ['attention_map1', 'attention_map2', 'attention_map3', 'attention_map4']:
        attn = model_intermediates[key][0, 0]  # [H, W]
        attn_np = attn.numpy()
        
        # Normalize
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        attn_np = (attn_np * 255).astype(np.uint8)
        
        # Resize
        attn_resized = cv2.resize(attn_np, (orig_shape[1], orig_shape[0]))
        
        # Colormap
        attn_color = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
        attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)
        
        # Overlay on original
        overlay = cv2.addWeighted(orig_img, 0.5, attn_color, 0.5, 0)
        
        attn_maps.append(overlay)
    
    # Create 2x2 grid
    cell_size = 300
    pad = 15
    
    grid_h = 2 * cell_size + pad
    grid_w = 2 * cell_size + pad
    
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    titles = ["Level 1 (Deep)", "Level 2 (Mid-Deep)", "Level 3 (Mid-Shallow)", "Level 4 (Shallow)"]
    
    for idx, ((row, col), title) in enumerate(zip(positions, titles)):
        img_resized = cv2.resize(attn_maps[idx], (cell_size, cell_size))
        y_start = row * (cell_size + pad)
        x_start = col * (cell_size + pad)
        canvas[y_start:y_start+cell_size, x_start:x_start+cell_size] = img_resized
        
        # Add title
        cv2.putText(canvas, title, (x_start + 10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas

# ============================================================================
# 5. DATASET
# ============================================================================

class InferenceDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, 'Color_Images')
        self.mask_dir = os.path.join(root_dir, 'Segmentation')
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg'))])
        self.model_transform = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)
        
        orig_img_cv = cv2.imread(img_path)
        orig_img_cv = cv2.cvtColor(orig_img_cv, cv2.COLOR_BGR2RGB)
        
        orig_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        gt_ids = convert_mask_id(orig_mask)
        
        pil_img = Image.fromarray(orig_img_cv)
        input_tensor = self.model_transform(pil_img)
        
        return input_tensor, gt_ids, orig_img_cv, fname

# ============================================================================
# 6. MAIN INFERENCE
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE DINOV2-ATTENTION-UNET INFERENCE")
    print("✅ Layer-by-Layer Visualization")
    print("✅ Class-Wise Pixel-Level Metrics")
    print("✅ Overall Pixel-Wise Metrics")
    print("="*80)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    vis_standard_dir = os.path.join(OUTPUT_DIR, "visualizations", "standard")
    vis_layer_dir = os.path.join(OUTPUT_DIR, "visualizations", "layer_by_layer")
    vis_attention_dir = os.path.join(OUTPUT_DIR, "visualizations", "attention_maps")
    metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
    
    for d in [vis_standard_dir, vis_layer_dir, vis_attention_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Load Model
    print("\n[1/4] Loading DINOv2-Attention-UNet model...")
    model = DinoAttentionUNet(NUM_CLASSES).to(DEVICE)
    
    if os.path.exists(ATTN_UNET_WEIGHTS):
        ckpt = torch.load(ATTN_UNET_WEIGHTS, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("✓ Model loaded successfully!")
    else:
        print(f"✗ Error: {ATTN_UNET_WEIGHTS} not found!")
        return
    
    model.eval()
    
    # Prepare Data
    print("\n[2/4] Loading test dataset...")
    dataset = InferenceDataset(TEST_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"✓ {len(dataset)} test images loaded!")
    
    # Initialize storage
    all_class_metrics = []  # List of per-image class metrics
    overall_metrics_list = []  # List of per-image overall metrics
    
    # Aggregate class-wise pixel counts across entire dataset
    class_aggregates = {cls: {'tp': 0, 'fp': 0, 'fn': 0, 'total_pixels': 0} 
                       for cls in range(NUM_CLASSES)}
    
    total_correct_pixels = 0
    total_pixels = 0
    
    # CSV files
    per_image_csv = os.path.join(metrics_dir, "per_image_metrics.csv")
    
    # Determine how many layer visualizations to create
    if NUM_LAYER_VIZ_SAMPLES == -1:
        num_layer_viz = len(dataset)
    else:
        num_layer_viz = min(NUM_LAYER_VIZ_SAMPLES, len(dataset))
    
    print(f"\n[3/4] Running Inference...")
    print(f"  - Creating standard visualizations for ALL {len(dataset)} images")
    print(f"  - Creating layer-by-layer visualizations for {num_layer_viz} images")
    
    # Inference Loop
    with open(per_image_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Filename', 'PixelAccuracy', 'mIoU_classwise', 'mDice_classwise']
        for cls_name in CLASS_NAMES:
            header.extend([f'{cls_name}_IoU', f'{cls_name}_Dice', 
                          f'{cls_name}_Precision', f'{cls_name}_Recall'])
        writer.writerow(header)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for idx, (input_tensor, gt_ids, orig_img, fnames) in enumerate(tqdm(dataloader, desc="Processing")):
                    fname = fnames[0]
                    input_tensor = input_tensor.to(DEVICE)
                    orig_h, orig_w = gt_ids.shape[1], gt_ids.shape[2]
                    
                    # Decide if we save intermediates for this image
                    save_intermediates = (idx < num_layer_viz)
                    
                    # Forward pass
                    logits = model(input_tensor, save_intermediates=save_intermediates)
                    logits = F.interpolate(logits, size=(orig_h, orig_w), 
                                         mode='bilinear', align_corners=True)
                    
                    # Get prediction
                    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                    gt_ids_np = gt_ids.numpy()[0]
                    orig_img_np = orig_img.numpy()[0]
                    
                    # ========================================
                    # METRICS CALCULATION
                    # ========================================
                    
                    # Class-wise pixel metrics
                    class_metrics = calculate_classwise_pixelwise_metrics(pred, gt_ids_np, NUM_CLASSES)
                    all_class_metrics.append({
                        'filename': fname,
                        'metrics': class_metrics
                    })
                    
                    # Overall pixel metrics
                    overall_metrics = calculate_overall_pixelwise_metrics(pred, gt_ids_np)
                    overall_metrics_list.append(overall_metrics)
                    
                    # Accumulate for dataset-level class metrics
                    for cm in class_metrics:
                        cls = cm['class_id']
                        class_aggregates[cls]['tp'] += cm['tp']
                        class_aggregates[cls]['fp'] += cm['fp']
                        class_aggregates[cls]['fn'] += cm['fn']
                        class_aggregates[cls]['total_pixels'] += cm['total_pixels']
                    
                    total_correct_pixels += overall_metrics['correct_pixels']
                    total_pixels += overall_metrics['total_pixels']
                    
                    # Calculate mean IoU and Dice for this image (across classes)
                    valid_ious = [cm['iou'] for cm in class_metrics if cm['iou'] > 0]
                    valid_dices = [cm['dice'] for cm in class_metrics if cm['dice'] > 0]
                    
                    miou_classwise = np.mean(valid_ious) if valid_ious else 0.0
                    mdice_classwise = np.mean(valid_dices) if valid_dices else 0.0
                    
                    # Write to CSV
                    row = [fname, overall_metrics['pixel_accuracy'], miou_classwise, mdice_classwise]
                    for cm in class_metrics:
                        row.extend([cm['iou'], cm['dice'], cm['precision'], cm['recall']])
                    writer.writerow(row)
                    
                    # ========================================
                    # VISUALIZATIONS
                    # ========================================
                    
                    # 1. Standard visualization (always)
                    gt_color = colorize_mask(gt_ids_np)
                    pred_color = colorize_mask(pred)
                    
                    standard_viz = create_standard_visualization(orig_img_np, gt_color, pred_color, fname)
                    standard_path = os.path.join(vis_standard_dir, fname)
                    cv2.imwrite(standard_path, cv2.cvtColor(standard_viz, cv2.COLOR_RGB2BGR))
                    
                    # 2. Layer-by-layer visualization (if enabled for this image)
                    if save_intermediates:
                        layer_viz = create_layer_by_layer_visualization(
                            model.intermediates, orig_img_np, gt_ids_np, pred, fname
                        )
                        layer_path = os.path.join(vis_layer_dir, fname)
                        cv2.imwrite(layer_path, cv2.cvtColor(layer_viz, cv2.COLOR_RGB2BGR))
                        
                        # 3. Attention map visualization
                        attn_viz = create_attention_map_visualization(model.intermediates, orig_img_np)
                        attn_path = os.path.join(vis_attention_dir, fname)
                        cv2.imwrite(attn_path, cv2.cvtColor(attn_viz, cv2.COLOR_RGB2BGR))
    
    # ========================================
    # FINAL AGGREGATION & REPORTING
    # ========================================
    
    print("\n[4/4] Calculating Final Metrics...")
    
    # Overall dataset pixel accuracy
    dataset_pixel_accuracy = total_correct_pixels / total_pixels
    
    # Calculate dataset-level class-wise metrics
    dataset_class_metrics = []
    total_intersection = 0  # For pixel-wise mIoU
    total_union = 0         # For pixel-wise mIoU
    
    for cls in range(NUM_CLASSES):
        agg = class_aggregates[cls]
        tp, fp, fn = agg['tp'], agg['fp'], agg['fn']
        
        # Per-class metrics
        intersection = tp
        union = tp + fp + fn
        
        iou = intersection / union if union > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Accumulate for pixel-wise mIoU
        total_intersection += intersection
        total_union += union
        
        dataset_class_metrics.append({
            'class_id': cls,
            'class_name': CLASS_NAMES[cls],
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'total_pixels': agg['total_pixels']
        })
    
    # PIXEL-WISE mIoU: Total intersection / Total union across ALL pixels
    pixel_wise_miou = total_intersection / total_union if total_union > 0 else 0.0
    
    # PIXEL-WISE mDice: 2 * Total intersection / (Total pred + Total actual) across ALL pixels
    # Dice = 2*intersection / (|A| + |B|) where A=prediction, B=ground truth
    total_pred_pixels = 0  # Total pixels predicted across all classes
    total_actual_pixels = 0  # Total actual pixels across all classes
    
    for cls in range(NUM_CLASSES):
        agg = class_aggregates[cls]
        tp, fp, fn = agg['tp'], agg['fp'], agg['fn']
        
        # For this class:
        # Predicted pixels = TP + FP
        # Actual pixels = TP + FN
        total_pred_pixels += (tp + fp)
        total_actual_pixels += (tp + fn)
    
    # Pixel-wise Dice = 2 * intersection / (pred + actual)
    pixel_wise_mdice = (2 * total_intersection) / (total_pred_pixels + total_actual_pixels) if (total_pred_pixels + total_actual_pixels) > 0 else 0.0
    
    # Mean IoU and Dice across classes (for reference - NOT the main metric)
    mean_iou_across_classes = np.mean([m['iou'] for m in dataset_class_metrics])
    mean_dice_across_classes = np.mean([m['dice'] for m in dataset_class_metrics])
    
    # ========================================
    # SAVE CLASS-WISE SUMMARY
    # ========================================
    
    class_summary_csv = os.path.join(metrics_dir, "class_wise_summary.csv")
    with open(class_summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'IoU', 'Dice', 'Precision', 'Recall', 'Total_Pixels'])
        for cm in dataset_class_metrics:
            writer.writerow([
                cm['class_name'],
                f"{cm['iou']:.4f}",
                f"{cm['dice']:.4f}",
                f"{cm['precision']:.4f}",
                f"{cm['recall']:.4f}",
                cm['total_pixels']
            ])
    
    # ========================================
    # SAVE SUMMARY REPORT
    # ========================================
    
    summary_path = os.path.join(metrics_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE DINOV2-ATTENTION-UNET INFERENCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Images: {len(dataset)}\n")
        f.write(f"Total Pixels Processed: {total_pixels:,}\n")
        f.write(f"Layer-by-Layer Visualizations Created: {num_layer_viz}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PRIMARY METRICS (PIXEL-WISE ACROSS ENTIRE DATASET)\n")
        f.write("="*80 + "\n")
        f.write(f"Pixel Accuracy:   {dataset_pixel_accuracy:.4f} ({total_correct_pixels:,}/{total_pixels:,})\n")
        f.write(f"Pixel-Wise mIoU:  {pixel_wise_miou:.4f} (Sum(Intersections) / Sum(Unions))\n")
        f.write(f"Pixel-Wise mDice: {pixel_wise_mdice:.4f} (2*Sum(Intersections) / (Sum(Pred) + Sum(Actual)))\n\n")
        
        f.write("="*80 + "\n")
        f.write("CLASS-WISE METRICS (AGGREGATED ACROSS ALL PIXELS)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8} {'Pixels':>12}\n")
        f.write("-"*80 + "\n")
        
        for cm in dataset_class_metrics:
            f.write(f"{cm['class_name']:<20} {cm['iou']:>8.4f} {cm['dice']:>8.4f} "
                   f"{cm['precision']:>10.4f} {cm['recall']:>8.4f} {cm['total_pixels']:>12,}\n")
        
        f.write("-"*80 + "\n")
        f.write(f"{'Mean (class-avg)':<20} {mean_iou_across_classes:>8.4f} "
               f"{mean_dice_across_classes:>8.4f}  ← For reference\n")
        f.write(f"{'Pixel-Wise (TRUE)':<20} {pixel_wise_miou:>8.4f} "
               f"{pixel_wise_mdice:>8.4f}  ← PRIMARY METRIC\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTES:\n")
        f.write("="*80 + "\n")
        f.write("★ PRIMARY METRIC: Pixel-Wise mIoU & mDice\n")
        f.write("  - Calculated as: Sum(all intersections) / Sum(all unions)\n")
        f.write("  - This is the TRUE pixel-level metric (NOT averaged by class)\n")
        f.write("  - Treats the entire dataset as one big segmentation task\n\n")
        f.write("- Class-wise metrics: Individual performance per class (for analysis)\n")
        f.write("- Mean (class-avg): Traditional mIoU (shown for comparison only)\n")
        f.write("- Pixel Accuracy: Total correct pixels / Total pixels\n")
        f.write("="*80 + "\n")
    
    # ========================================
    # PRINT TO CONSOLE
    # ========================================
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    
    print(f"\n📊 PRIMARY PIXEL-WISE METRICS:")
    print(f"  Pixel Accuracy:   {dataset_pixel_accuracy:.4f}")
    print(f"  Pixel-Wise mIoU:  {pixel_wise_miou:.4f}  ★ PRIMARY")
    print(f"  Pixel-Wise mDice: {pixel_wise_mdice:.4f}  ★ PRIMARY")
    print(f"\n  (For reference)")
    print(f"  Mean IoU (class-avg):  {mean_iou_across_classes:.4f}")
    print(f"  Mean Dice (class-avg): {mean_dice_across_classes:.4f}")
    
    print(f"\n📁 OUTPUT STRUCTURE:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── visualizations/")
    print(f"    │   ├── standard/          ({len(dataset)} images)")
    print(f"    │   ├── layer_by_layer/    ({num_layer_viz} images)")
    print(f"    │   └── attention_maps/    ({num_layer_viz} images)")
    print(f"    └── metrics/")
    print(f"        ├── per_image_metrics.csv")
    print(f"        ├── class_wise_summary.csv")
    print(f"        └── summary.txt")
    
    print("\n" + "="*80)
    print("CLASS-WISE PERFORMANCE:")
    print("="*80)
    
    for cm in dataset_class_metrics:
        print(f"{cm['class_name']:<20} IoU: {cm['iou']:.4f}  Dice: {cm['dice']:.4f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
# Getting Started - PyTorch Semantic Segmentation (V1, V2, V3)

## ⚡ Quick Start (5 minutes)

### Step 1: Verify Your Dataset Structure
```bash
python setup_and_utils.py
# Choose option 1 to verify/create directories
```

Expected structure:
```
/Users/abhayatrivedi/Downloads/
├── train_3/
│   ├── train3/
│   │   ├── images/     ← PNG image files
│   │   └── labels/     ← .txt label files (pixel-level class IDs)
│   └── val3/
│       ├── images/
│       └── labels/
└── test3/
    ├── images/        ← PNG image files
    └── labels/        ← .txt label files
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision albumentations segmentation-models-pytorch tqdm opencv-python matplotlib pillow numpy
```

Or use the setup script:
```bash
python setup_and_utils.py
# Choose option 4 to install everything
```

### Step 3: Train Your Models

**Option A: Train all three versions sequentially**
```bash
python train_v1.py   # ~2-3 hours
python train_v2.py   # ~3-4 hours
python train_v3.py   # ~8-12 hours (slower due to ViT)
```

**Option B: Train individual models**
```bash
# Just V1
python train_v1.py

# Just V2
python train_v2.py

# Just V3
python train_v3.py
```

### Step 4: Evaluate Your Models
```bash
python test_all_versions.py
```

This will:
- Load all trained models
- Run inference on test dataset
- Calculate metrics (Accuracy, mIoU, mDice)
- Save prediction visualizations
- Generate comparison report

---

## 📊 Model Comparison

### V1: UNet + ResNet50
**Best for:** Real-time applications, speed
- ✓ Fastest inference (~50ms per image)
- ✓ Simple architecture
- ✓ Good for standard use cases
- ⚠ Lower accuracy than V2/V3

**Configuration:**
```python
Input size: 512×512
Batch size: 8
Epochs: 50
Learning rate: 1e-4
Loss: CrossEntropy + Dice + Focal
```

### V2: DeepLabV3+ + ResNet50
**Best for:** Boundary precision, small objects
- ✓ Better boundary detection (ASPP module)
- ✓ Good balance of speed/accuracy
- ✓ Excellent for detailed segmentation
- ✓ Moderate computational cost

**Configuration:**
```python
Input size: 512×512
Batch size: 8
Epochs: 50
Learning rate: 2e-4
Loss: CrossEntropy + Dice + Jaccard
```

### V3: Hybrid Transformer (DINOv2 + CNN)
**Best for:** Maximum accuracy, research
- ✓ Highest accuracy
- ✓ Self-supervised pretrained backbone
- ✓ Global semantic understanding
- ⚠ Slowest inference (~200ms)
- ⚠ Requires more VRAM (recommend 8GB+)

**Configuration:**
```python
Input size: 512×512
Batch size: 4 (lower due to ViT)
Epochs: 100
Learning rate: 1e-4
Progressive unfreezing at epoch 20
Loss: CrossEntropy + Dice + Focal
```

---

## 🔧 Customization

### Change Input Size
Edit the `CONFIG` dictionary in each training script:
```python
CONFIG = {
    "input_size": 256,  # Change from 512 to 256
    ...
}
```

### Change Batch Size
For lower VRAM:
```python
CONFIG["batch_size"] = 4  # V1, V2
CONFIG["batch_size"] = 2  # V3
```

### Change Learning Rate
```python
CONFIG["learning_rate"] = 5e-5  # Smaller for fine-tuning
```

### Use CPU Instead of GPU
```python
CONFIG["device"] = "cpu"
```

### Change Number of Epochs
```python
CONFIG["epochs"] = 100
```

---

## 📈 Expected Results

After training with sample data:

| Metric | V1 | V2 | V3 |
|--------|----|----|-----|
| Pixel Accuracy | 0.78-0.82 | 0.82-0.86 | 0.85-0.90 |
| mIoU | 0.62-0.65 | 0.65-0.70 | 0.70-0.75 |
| mDice | 0.70-0.74 | 0.74-0.78 | 0.77-0.82 |

*Results vary based on dataset quality and complexity*

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
1. Reduce batch size: `CONFIG["batch_size"] = 4`
2. Use smaller input: `CONFIG["input_size"] = 256`
3. Use CPU: `CONFIG["device"] = "cpu"`

### Issue: "No such file or directory: train_3/train3/images"
**Solution:**
```bash
python setup_and_utils.py
# Choose option 1 or 4 to create directories
```

### Issue: "No module named segmentation_models_pytorch"
**Solution:**
```bash
pip install segmentation-models-pytorch
```

### Issue: "Model checkpoint not found for testing"
**Cause:** Training scripts haven't finished yet
**Solution:** Wait for training to complete and check `./checkpoints/` folder

### Issue: V3 Model runs very slowly
**Expected:** V3 (Transformer) is slower than V1/V2
**Mitigation:**
- Reduce batch size to 2
- Use CPU for inference testing only
- Consider using V1/V2 for production

### Issue: Out of labels/images
**Solution:**
```bash
python setup_and_utils.py
# Choose option 2 to generate sample data
```

---

## 📝 Understanding Output Files

### Checkpoints
```
checkpoints/
├── v1_best.pth     → Best V1 model (lowest validation loss)
├── v1_final.pth    → Final V1 model (after all epochs)
├── v2_best.pth
├── v2_final.pth
├── v3_best.pth
└── v3_final.pth
```

### Logs
```
logs/
├── v1_history.json  → Training history (loss curves)
├── v2_history.json
└── v3_history.json
```

### Test Results
```
test_results/
├── V1/              → V1 prediction visualizations
├── V2/              → V2 prediction visualizations
├── V3/              → V3 prediction visualizations
└── test_results.json → Quantitative metrics
```

---

## 🎯 Workflow Examples

### Example 1: Quick Testing
```bash
# Use existing trained models for inference
python test_all_versions.py
```

### Example 2: Retrain Best Model (V2)
```bash
python train_v2.py
python test_all_versions.py
```

### Example 3: Compare All Models
```bash
python train_v1.py
python train_v2.py
python train_v3.py
python test_all_versions.py
# Check test_results/test_results.json for comparison
```

### Example 4: Fine-tune on Custom Data
1. Place your data in proper directories
2. Modify learning rate: `CONFIG["learning_rate"] = 5e-5`
3. Reduce epochs: `CONFIG["epochs"] = 10`
4. Load pretrained weights (modify train_vX.py):
   ```python
   checkpoint = torch.load("./checkpoints/v1_best.pth")
   model.load_state_dict(checkpoint["model_state_dict"])
   ```

---

## 📚 File Organization

```
project/
├── train_v1.py              → V1 training script
├── train_v2.py              → V2 training script
├── train_v3.py              → V3 training script
├── test_all_versions.py     → Testing script
├── setup_and_utils.py       → Setup utilities
├── GETTING_STARTED.md       → This file
├── checkpoints/             → Saved models
├── logs/                    → Training history
├── test_results/            → Inference outputs
└── training_config.json     → Configuration
```

---

## ⚙️ System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB

**Recommended for GPU training:**
- GPU: NVIDIA RTX 2080 or better (8GB+ VRAM)
- RAM: 16GB
- Storage: 5GB

**For V3 (Transformer):**
- GPU: NVIDIA RTX 3080 or better (10GB+ VRAM)
- RAM: 32GB
- Storage: 10GB

---

## 🚀 Advanced Usage

### Using Different Pretrained Backbones
Edit the encoder in training scripts:
```python
# V1
model = smp.Unet(
    encoder_name="efficientnet-b4",  # Change from resnet50
    encoder_weights="imagenet",
    ...
)

# V2
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",  # Change from resnet50
    ...
)
```

### Custom Loss Function
Modify `CombinedLoss` class:
```python
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lovasz_loss = LovaszSoftmaxLoss()  # Add custom loss
        ...
    
    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        lovasz = self.lovasz_loss(outputs, targets)
        total = 0.6 * ce + 0.4 * lovasz
        return total, {...}
```

### Resume Training
```python
# In train script, before training loop:
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
```

---

## 📞 Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review the training/test script comments
3. Check PyTorch and `segmentation-models-pytorch` documentation
4. Verify your dataset format matches expected structure

---

**Happy Training! 🎉**

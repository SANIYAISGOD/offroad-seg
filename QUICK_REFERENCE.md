# 🚀 Quick Reference Card

## One-Liner Start
```bash
python run_pipeline.py
```

---

## Training Commands

```bash
# All three models
python train_v1.py && python train_v2.py && python train_v3.py

# Just one model
python train_v1.py  # UNet - Fastest
python train_v2.py  # DeepLabV3+ - Balanced
python train_v3.py  # Transformer - Most accurate
```

---

## Testing Command
```bash
python test_all_versions.py
```

---

## Data Structure
```
/Users/abhayatrivedi/Downloads/
├── train_3/train3/{images,labels}
├── train_3/val3/{images,labels}
└── test3/{images,labels}
```

---

## Model Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **V1** | ⚡⚡⚡ | ⭐⭐⭐ | 6GB | Real-time |
| **V2** | ⚡⚡ | ⭐⭐⭐⭐ | 7GB | Production |
| **V3** | ⚡ | ⭐⭐⭐⭐⭐ | 10GB | Research |

---

## File Locations After Training

```
checkpoints/      → Trained model weights
logs/            → Training history (JSON)
test_results/    → Predictions & metrics
```

---

## Key Files Created

| File | Purpose |
|------|---------|
| train_v1.py | Train UNet model |
| train_v2.py | Train DeepLabV3+ model |
| train_v3.py | Train Transformer model |
| test_all_versions.py | Test all models |
| run_pipeline.py | Automate everything |
| setup_and_utils.py | Setup & utilities |

---

## Configuration Template

```python
CONFIG = {
    "input_size": 512,         # Image size
    "num_classes": 10,         # Number of classes
    "batch_size": 8,           # Batch size
    "learning_rate": 1e-4,     # Learning rate
    "epochs": 50,              # Training epochs
    "device": "cuda",          # "cuda" or "cpu"
}
```

---

## Metrics Explained

- **Accuracy**: Correct pixels / Total pixels
- **IoU**: Intersection / Union (per class)
- **mIoU**: Mean IoU across all classes
- **Dice**: F1-score for segmentation

---

## Classes (0-9)
```
0: Background   1: Trees       2: Lush Bushes  3: Dry Grass
4: Dry Bushes   5: Ground      6: Flowers      7: Logs
8: Rocks        9: Sky
```

---

## Troubleshooting Cheat Sheet

| Problem | Solution |
|---------|----------|
| GPU OOM | Reduce batch size or input size |
| Data not found | Run `setup_and_utils.py` |
| Module not found | `pip install segmentation-models-pytorch` |
| Slow training | Reduce input size or epochs |
| Poor accuracy | Use V2 or V3 model |

---

## Performance Expectations

```
V1: 0.82 accuracy, 0.65 mIoU, 50ms/image
V2: 0.84 accuracy, 0.68 mIoU, 75ms/image
V3: 0.86 accuracy, 0.72 mIoU, 200ms/image
```

---

## Label Format

Each `.txt` file = pixel-level class indices:
```
# Example (512×512 image)
0 0 1 1 2 2 3 3 ...
0 1 1 2 2 3 3 4 ...
...
```

Create with:
```python
import numpy as np
mask = np.random.randint(0, 10, (512, 512))
np.savetxt("label.txt", mask, fmt='%d')
```

---

## Training Timeline

| Step | Time |
|------|------|
| Setup | 5 min |
| V1 Training | 2-3 hours |
| V2 Training | 3-4 hours |
| V3 Training | 8-12 hours |
| Testing | 5-10 min |
| **Total** | **1-2 days** |

---

## GPU Memory Requirements

- V1: 6GB (batch_size=8)
- V2: 7GB (batch_size=8)
- V3: 10GB (batch_size=4)

Reduce batch_size if OOM.

---

## Typical Output

```
checkpoints/
├── v1_best.pth      (~200MB)
├── v1_final.pth     (~200MB)
├── v2_best.pth      (~220MB)
├── v2_final.pth     (~220MB)
├── v3_best.pth      (~350MB)
└── v3_final.pth     (~350MB)

logs/
├── v1_history.json
├── v2_history.json
└── v3_history.json

test_results/
├── V1/               (Visualizations)
├── V2/               (Visualizations)
├── V3/               (Visualizations)
└── test_results.json (Metrics)
```

---

## Common Commands

```bash
# Setup everything
python setup_and_utils.py

# Train all models at once
python run_pipeline.py

# Train individual model
python train_v1.py

# Test all models
python test_all_versions.py

# Check GPU
nvidia-smi

# View training history
cat logs/v1_history.json
```

---

## Python Version & Dependencies

```bash
# Python 3.8+
python --version

# Required packages
pip install torch torchvision albumentations \
    segmentation-models-pytorch tqdm opencv-python \
    matplotlib pillow numpy
```

---

## Architecture Comparison

```
V1 (UNet)
Input → Encoder → Decoder → Output
Simple, fast, good baseline

V2 (DeepLabV3+)
Input → Encoder → ASPP → Decoder → Output
Multi-scale features, better boundaries

V3 (Hybrid)
Input → DINOv2 → Pyramid → Multi-head → Output
State-of-the-art, slow but accurate
```

---

## Loss Functions

- **CrossEntropy**: Standard classification
- **Dice**: Penalizes FP/FN equally
- **Focal**: Focuses on hard examples
- **Jaccard**: Direct IoU optimization

V1: CE + Dice + Focal
V2: CE + Dice + Jaccard
V3: CE + Dice + Focal

---

## Customization Quick Fixes

```python
# Faster training
CONFIG["epochs"] = 10
CONFIG["learning_rate"] = 1e-3

# Fit in memory
CONFIG["batch_size"] = 2
CONFIG["input_size"] = 256

# Better accuracy
CONFIG["epochs"] = 100
CONFIG["batch_size"] = 4
CONFIG["learning_rate"] = 1e-5

# Use CPU
CONFIG["device"] = "cpu"
```

---

## Export Models

```python
import torch

# TorchScript
model = torch.load("checkpoints/v1_best.pth")
scripted = torch.jit.script(model)
scripted.save("model.pt")

# ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## Documentation Map

```
QUICK_REFERENCE.md  ← You are here (1-page cheat sheet)
    ↓
GETTING_STARTED.md  (5-minute setup)
    ↓
COMPLETE_GUIDE.md   (Comprehensive reference)
```

---

## Key Milestones

✅ Environment setup
✅ Data preparation
✅ Model training
✅ Model testing
✅ Result analysis
✅ Production deployment

---

## Success Indicators

After each step:
- ✅ Setup: Directories created
- ✅ Training: Loss decreasing, checkpoints saved
- ✅ Testing: Metrics printed, visualizations saved
- ✅ Complete: pipeline_report.json generated

---

## Tips & Tricks

1. **Monitor training**: Watch loss decrease in logs
2. **Save disk space**: Delete old checkpoints
3. **Speed up V3**: Use smaller input size
4. **Improve accuracy**: Increase augmentation
5. **Debug issues**: Check data first!

---

## Emergency Commands

```bash
# If stuck, kill training
Ctrl+C

# Clear cache
rm -rf __pycache__

# Remove old checkpoints
rm checkpoints/*

# Start fresh
rm -rf checkpoints logs test_results
python run_pipeline.py
```

---

## Next Steps

1. **Run now**: `python run_pipeline.py`
2. **Read more**: Check GETTING_STARTED.md
3. **Deep dive**: Read COMPLETE_GUIDE.md
4. **Customize**: Edit CONFIG in training scripts
5. **Deploy**: Use best model for production

---

## One-Page Summary

| Item | Details |
|------|---------|
| **Language** | Pure PyTorch |
| **Models** | 3 (UNet, DeepLabV3+, Transformer) |
| **Classes** | 10 semantic categories |
| **Format** | Images (PNG/JPG) + Labels (TXT) |
| **Output** | Models, metrics, visualizations |
| **Time** | 1-2 days full pipeline |
| **Start** | `python run_pipeline.py` |

---

**⚡ Ready? Run: `python run_pipeline.py`**

**📚 Need more? Read: `GETTING_STARTED.md`**

**🎓 Want details? See: `COMPLETE_GUIDE.md`**

---

**Made with ❤️ for semantic segmentation**

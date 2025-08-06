# 🎯 Polygon Color Generation UNet - Project Guide

## 📋 Project Overview
This project implements a conditional UNet model that generates colored polygon images from polygon outlines and color specifications. The model learns to fill polygon shapes with specified colors while preserving the original shape boundaries.

## 🏗️ Architecture
- **Model**: Conditional UNet with color conditioning
- **Inputs**: 
  - Polygon outline image (256x256 grayscale)
  - Color name (8 possible colors)
- **Output**: Colored polygon image (256x256 RGB)
- **Conditioning**: Color embedding layer (64-dimensional)

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone or download the project
cd c:\download\dataset\dataset

# Install dependencies
pip install -r requirements.txt

# One-click setup (recommended)
python setup_and_run.py --install-only
```

### 2. Quick Training
```bash
# Full setup + training
python setup_and_run.py

# Or manual training
python train_unet.py
```

### 3. Interactive Testing
```bash
# Open Jupyter notebook for inference
python setup_and_run.py --notebook

# Or run evaluation
python evaluate_model.py --visualize --samples 10
```

## 🔧 Detailed Setup Instructions

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional but recommended)
- 4GB+ RAM
- 2GB+ disk space

### Step 1: Install Dependencies
```bash
# Option A: Automatic setup
python setup_and_run.py --install-only

# Option B: Manual installation
pip install torch torchvision torchaudio
pip install wandb pillow numpy tqdm matplotlib
```

### Step 2: Configure Weights & Biases
```bash
# Login to wandb (for experiment tracking)
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

### Step 3: Verify Dataset Structure
```
dataset/
├── training/
│   ├── inputs/          # Polygon outlines
│   ├── outputs/         # Colored polygons  
│   └── data.json        # Training mappings
├── validation/
│   ├── inputs/          # Validation outlines
│   ├── outputs/         # Validation colored
│   └── data.json        # Validation mappings
```

## 🎯 Training Guide

### Basic Training
```bash
python train_unet.py
```

### Advanced Training Options
```bash
# Custom hyperparameters
python train_unet.py --lr 0.001 --epochs 150 --batch_size 32

# Resume training
python train_unet.py --resume best_polygon_unet.pth

# CPU training (slower)
python train_unet.py --device cpu
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Batch Size | 16 | Training batch size |
| Epochs | 100 | Total training epochs |
| Image Size | 256 | Input/output image resolution |
| Embed Dim | 64 | Color embedding dimension |

## 📊 Evaluation & Testing

### Model Evaluation
```bash
# Evaluate on validation set
python evaluate_model.py --data validation --visualize

# Evaluate on training set
python evaluate_model.py --data training --visualize --samples 15
```

### Metrics Available
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **PSNR**: Peak Signal-to-Noise Ratio
- **Color-wise Performance**: Per-color accuracy analysis

### Interactive Testing
```bash
# Launch Jupyter notebook
jupyter notebook inference_demo.ipynb

# Or use setup script
python setup_and_run.py --notebook
```

## 🔍 Inference Examples

### Python API Usage
```python
from PIL import Image
import torch
from unet_model import ConditionalUNet

# Load model
model = ConditionalUNet(n_channels=1, n_classes=3, num_colors=8, embed_dim=64)
model.load_state_dict(torch.load('best_polygon_unet.pth')['model_state_dict'])
model.eval()

# Prepare input
input_img = Image.open('triangle.png').convert('L')
input_tensor = transforms.ToTensor()(input_img).unsqueeze(0)

# Generate colored output
color_idx = torch.tensor([0])  # 0=red, 1=blue, etc.
with torch.no_grad():
    output = model(input_tensor, color_idx)
```

### Supported Colors
- Red (0), Blue (1), Green (2), Yellow (3)
- Purple (4), Orange (5), Cyan (6), Magenta (7)

## 📈 Expected Performance

### Training Metrics
- **Training Loss**: Should decrease from ~0.8 to ~0.1
- **Validation Loss**: Should stabilize around 0.05-0.15
- **PSNR**: Expected 25-35 dB for good results
- **Training Time**: ~30-60 minutes on GPU, ~2-4 hours on CPU

### Sample Results
| Shape | Color | MSE | PSNR | Visual Quality |
|-------|-------|-----|------|----------------|
| Triangle | Red | 0.02 | 32.5 dB | Excellent |
| Square | Blue | 0.03 | 30.2 dB | Good |
| Circle | Green | 0.01 | 35.1 dB | Excellent |

## 🛠️ Advanced Features

### Data Augmentation
```bash
# Generate augmented dataset
python augment_data.py --rotate --scale --output_dir augmented_dataset

# Train with augmented data
python train_unet.py --dataset augmented_dataset
```

### Custom Colors
```python
# Add new colors to color_map in train_unet.py
color_map = {
    'red': 0, 'blue': 1, 'green': 2, 'yellow': 3,
    'purple': 4, 'orange': 5, 'cyan': 6, 'magenta': 7,
    'pink': 8, 'brown': 9  # Add new colors
}
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Reduce batch size
python train_unet.py --batch_size 8

# Use CPU
python train_unet.py --device cpu
```

**Issue**: wandb login required
```bash
# Disable wandb logging
python train_unet.py --no_wandb

# Or login
wandb login your_api_key
```

**Issue**: Poor training results
- Check dataset paths are correct
- Verify image formats (PNG expected)
- Ensure color mappings match between training/validation
- Try reducing learning rate

### Performance Tips
- Use GPU for 10x faster training
- Increase batch size if memory allows
- Enable data augmentation for better generalization
- Monitor wandb dashboard for early stopping

## 📁 Project Structure Reference

```
dataset/
├── training/
│   ├── inputs/          # 8 polygon outlines
│   ├── outputs/         # 64 colored variants (8 shapes × 8 colors)
│   └── data.json        # Training mappings
├── validation/
│   ├── inputs/          # 4 polygon outlines
│   ├── outputs/         # 5 colored variants
│   └── data.json        # Validation mappings
├── unet_model.py        # UNet architecture
├── train_unet.py        # Training script
├── evaluate_model.py    # Evaluation tools
├── inference_demo.ipynb # Interactive testing
├── augment_data.py      # Data augmentation
├── setup_and_run.py     # Setup automation
└── requirements.txt     # Dependencies
```

## 🔗 Useful Commands Summary

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Quick setup | `python setup_and_run.py` |
| Train model | `python train_unet.py` |
| Evaluate model | `python evaluate_model.py --visualize` |
| Open notebook | `python setup_and_run.py --notebook` |
| Generate augmented data | `python augment_data.py --rotate --scale` |
| Check CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify dataset structure matches requirements
3. Check wandb dashboard for training progress
4. Ensure all dependencies are installed correctly

Happy training! 🎉
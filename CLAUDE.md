# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning image classification project for identifying plant diseases and pests. The model will be deployed on a Linux server with NVIDIA GPU (dnlab-server) and serve as the core inference engine for a web service.

# rules
Answer in Traditional Chinese

## System Environment

- **OS**: Linux (Ubuntu/Debian)
- **Hardware**: NVIDIA GPU with CUDA support
- **Package Manager**: `uv` (Modern Python Package Manager)
- **Framework**: PyTorch
- **Model Library**: `timm` (PyTorch Image Models)
- **Python Version**: 3.10+

## Project Setup Commands

### Initialize environment with uv
```bash
# Initialize project with uv
uv init

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install timm scikit-learn tqdm
```

### Data validation
```bash
# Run data structure validation script
python check_data.py
```

### Training
```bash
# Train the model
python train.py

# Training with custom hyperparameters (adjust as needed)
python train.py --batch-size 4 --epochs 50 --lr 0.001
```

## Architecture and Model Specifications

### Model Architecture
- **Backbone**: `convnext_large.fb_in1k` from timm
- **Rationale**: ConvNeXt Large provides strong feature extraction for fine-grained disease patterns
- **Training Method**: Transfer learning/Fine-tuning with ImageNet-1k pretrained weights
- **Classifier Head**: Modified to match specific disease class count

### Optimization
- **Mixed Precision Training (AMP)**: Uses `torch.amp` to reduce VRAM and accelerate training
- **Optimizer**: AdamW

### Data Structure
Dataset must follow `torchvision.datasets.ImageFolder` format:
```
dataset/
  ├── train/  (classA/, classB/, classC/, ...)
  └── val/    (classA/, classB/, classC/, ...)
```

### Preprocessing Requirements
- Use `timm.data.resolve_data_config` to automatically get model-required input size, mean, std
- **Training set**: Include data augmentation (Resize, Flip, Rotation, Normalize)
- **Validation set**: Only Resize and Normalize

## Training Output Artifacts

The training script must produce:

1. **Model weights (`.pth`)**: Saves the model with best validation accuracy
2. **Class mapping (`classes.json`)**: Maps index (0, 1, 2...) to label (disease name) for inference
3. **Training logs**: Console output showing Loss and Accuracy per epoch

## Key Implementation Notes

### Memory Management
- If encountering OOM (Out of Memory) errors, reduce batch size (recommended: 8 or 4)
- AMP (Automatic Mixed Precision) is essential for memory efficiency with ConvNeXt Large

### Critical Code Requirements
- The training script MUST save `dataset.classes` to `classes.json` for inference mapping
- Best model checkpoint should be saved based on validation accuracy, not training loss
- Use `timm.create_model('convnext_large.fb_in1k', pretrained=True, num_classes=N)` where N is the number of disease classes

### Hyperparameters
Training scripts should expose easily adjustable hyperparameters:
- Batch Size (default: 8 or 4)
- Learning Rate
- Number of Epochs
- Data augmentation parameters

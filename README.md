# ICL: Independent and Collaborative Learning for Long-Tail Visual Recognition

Enhancing Mixture of Experts with Independent and Collaborative Learning for Long‑Tail Visual Recognition (IJCAI 2025)

This project implements a Mixture of Experts (MoE) framework that integrates **Independent Learning** and **Collaborative Learning** to address long-tail visual recognition problems. The framework is specifically designed for space group classification tasks on XRD (X-ray Diffraction) data, achieving excellent performance on datasets with extreme long-tail distributions.

---

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
- [Optimal Training Configurations](#-optimal-training-configurations)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Loss Functions](#-loss-functions)
- [FAQ](#-faq)

---

## ✨ Features

- **Mixture of Experts Architecture (MoE)**: 8 expert networks, each focusing on different data distributions
- **Rule-Guided Gating Network**: Dynamically routes samples to appropriate experts using crystallographic prior knowledge
- **Hierarchical Classification**: Crystal system classification (7 classes) → Space group classification (230 classes)
- **Dynamic Masking Mechanism**: Constrains space group prediction space based on predicted crystal system, improving accuracy
- **Physics-Constrained Contrastive Learning**: Enhances feature representation using rule similarity matrix
- **Multiple Frontend Networks**: Supports both ResTcn and ViT frontend architectures
- **Adaptive Learning Rate Strategy**: Early decay and learning rate restart mechanisms

---

## 🔧 Requirements

- **Python**: ≥ 3.8
- **PyTorch**: ≥ 1.13.0
- **CUDA**: ≥ 11.0 (Recommended for GPU acceleration)
- **Operating System**: Linux / macOS / Windows

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ICL
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n icl python=3.10
conda activate icl

# Or using venv
python -m venv icl_env
source icl_env/bin/activate  # Linux/Mac
# or
icl_env\Scripts\activate  # Windows
```

### 3. Install Dependencies

**Method 1: Using requirements.txt (Recommended)**

```bash
# First install PyTorch (choose according to your CUDA version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU version
pip install torch torchvision torchaudio

# Then install other dependencies
pip install -r requirements.txt
```

**Method 2: Manual Installation**

```bash
# Install PyTorch (choose according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy scikit-learn loguru tqdm wandb nni
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📊 Data Preparation

### Data Format Requirements

The project supports three data source formats:

1. **Preprocessed NPY Format** (Recommended, fastest)
   - Directory structure:
     ```
     processed_data_dir/
     ├── ccdc_sg_train/
     │   ├── train_ccdc_sg.npy
     │   └── test_val_ccdc_sg.npy
     └── ccdc_sg_test/
         └── test_ccdc_sg.npy
     ```

2. **LMDB Format** (Requires real-time processing)
   - LMDB database containing raw XRD data

3. **Raw File Format** (MP20 format)
   - Files with train and test prefixes

### Required Files

- `rule_matrix.csv`: Rule similarity matrix (230×230), defining physical similarity between space groups
- `sg_count.csv` or `ccdc_sg_count.csv`: Space group classification file (for Head/Medium/Tail division)

---

## 🚀 Quick Start

### Basic Training Command

```bash
python train.py \
    --use_processed_npy \
    --processed_data_dir /path/to/your/data \
    --rule_matrix_path /path/to/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --epochs 100 \
    --use_adamw \
    --lr_restart_enabled \
    --adjust_lr_strategy
```

### Parameter Description

- `--use_processed_npy`: Use preprocessed NPY data format
- `--processed_data_dir`: Path to processed data directory
- `--rule_matrix_path`: Path to rule matrix file (required)
- `--device`: GPU device ID (e.g., 0, 1, etc.)
- `--loss_scheme`: Loss scheme (scheme2 is the recommended configuration)
- `--use_contrast_scheme2`: Enable contrastive learning
- `--epochs`: Number of training epochs
- `--use_adamw`: Use AdamW optimizer
- `--lr_restart_enabled`: Enable learning rate restart
- `--adjust_lr_strategy`: Adjust learning rate strategy (delay early decay)

---

## 🎯 Optimal Training Configurations

Based on extensive experiments and hyperparameter tuning, here are the optimal configurations for different datasets:

### CCDC Dataset (Recommended Configuration)

```bash
python train.py \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --use_adamw \
    --adjust_lr_strategy \
    --lr_restart_enabled \
    --lambda_dol 0.5 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --max_grad_norm 1.0 \
    --epochs 100 \
    --seed 42 \
    --save_log True
```

**Optimal Hyperparameters**:
- `lambda_dol`: 0.5 (MoE independent optimization loss weight)
- `lambda_hier`: 1.5 (Hierarchical loss weight, includes L_sys and L_sg)
- `lambda_scl`: 0.2 (Contrastive learning loss weight)
- `max_grad_norm`: 1.0 (Gradient clipping threshold)
- `optimizer`: AdamW (learning rate: 5e-4)
- `early_decay_ratio`: 0.2 (Early decay ratio)
- `early_decay_target_lr`: 1e-3 (Target learning rate)

### Coremof19 Dataset

```bash
python train.py \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --use_adamw \
    --lambda_dol 0.4 \
    --lambda_hier 1.8 \
    --lambda_scl 0.25 \
    --max_grad_norm 1.0 \
    --epochs 100 \
    --seed 42
```

**Optimal Hyperparameters**:
- `lambda_dol`: 0.4
- `lambda_hier`: 1.8
- `lambda_scl`: 0.25
- `early_decay_ratio`: 0.1
- `early_decay_target_lr`: 5e-4

### Inor Dataset

```bash
python train.py \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --use_adamw \
    --lambda_dol 0.5 \
    --lambda_hier 1.2 \
    --lambda_scl 0.15 \
    --max_grad_norm 1.0 \
    --epochs 100 \
    --seed 42
```

**Optimal Hyperparameters**:
- `lambda_dol`: 0.5
- `lambda_hier`: 1.2
- `lambda_scl`: 0.15
- `early_decay_ratio`: 0.2
- `early_decay_target_lr`: 1e-3

### Using ViT Frontend

```bash
python train.py \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --use_adamw \
    --frontend_type vit \
    --vit_patch_size 50 \
    --vit_embed_dim 256 \
    --vit_depth 6 \
    --vit_num_heads 8 \
    --vit_mlp_ratio 4.0 \
    --vit_use_cls_token \
    --lambda_dol 0.5 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 100
```

---

## 📁 Project Structure

```
ICL/
├── Trainer/                  # Trainer modules
│   ├── moe_trainer.py       # MoE trainer (main training logic)
│   └── default_trainer.py  # Base trainer
├── config/                   # Configuration files
│   ├── config_cifar_base.py # Base configuration
│   └── config_cifar_moe.py  # MoE configuration
├── datasets/                 # Dataset modules
│   ├── OneD_Dataset.py      # 1D dataset (NPY format)
│   └── LMDB_Dataset.py      # LMDB dataset
├── loss/                     # Loss functions
│   ├── physics_contrast_loss.py  # Physics-constrained contrastive loss
│   └── moe_loss.py          # MoE loss
├── models/                   # Model definitions
│   ├── Resnet1D.py          # ResNet1D_MoE main model
│   └── CrystalFusionNet.py  # Frontend network components
├── metrics/                  # Evaluation metrics
├── utils/                    # Utility functions
│   ├── utils.py             # General utilities
│   ├── schedular.py         # Learning rate scheduler
│   └── sg_classifier.py     # Space group classification utilities
├── train.py                  # Main training script
└── README.md                 # This document
```

---

## 🏗️ Model Architecture

### Core Components

1. **Frontend Network**: ResTcn (default) or ViT
   - ResTcn: ResNet-based 1D convolutional network
   - ViT: Vision Transformer with configurable patch size and depth

2. **Crystal System Classifier**: 7-class crystal system classification
   - Input: Frontend features (B, 1024)
   - Projection: Linear(1024 → 256)
   - Output: 7-class crystal system logits

3. **Expert Networks**: 8 independent experts
   - Lightweight mode: Shared Layer3, experts only responsible for classification
   - Original mode: Each expert has independent Layer3

4. **Gating Network**: Rule-guided soft routing
   - Input: Crystal system projection features (B, 256)
   - Rule bias: (7, 8) matrix, injecting crystallographic priors
   - Output: Weight distribution over 8 experts

5. **Ensemble Prediction**: Weighted average + dynamic masking
   - Weighted average: Combines expert outputs according to gating weights
   - Dynamic masking: Constrains space group prediction space based on predicted crystal system



---

## 📉 Loss Functions

### Scheme2 Loss Formula

```
L_total = λ_dol * L_dol + λ_hier * (L_sys + L_sg) + λ_scl * L_scl
```

**Loss Components**:

1. **L_dol** (Independent Optimization Loss)
   - Purpose: Makes each expert independently optimize its own predictions
   - Default weight: 0.5
   - Range: 0.3 - 0.8

2. **L_hier** (Hierarchical Loss = L_sys + L_sg)
   - L_sys: Crystal system classification loss (7 classes)
   - L_sg: Space group classification loss (230 classes, with masking)
   - Default weight: 1.5
   - Range: 1.0 - 2.0

3. **L_scl** (Contrastive Learning Loss)
   - Purpose: Physics-constrained contrastive learning using rule similarity matrix
   - Default weight: 0.2
   - Range: 0.1 - 0.5
   - Only enabled when `--use_contrast_scheme2` is used

For detailed loss function description, please refer to [`loss/Loss模块介绍.md`](loss/Loss模块介绍.md)

---

## 🔍 FAQ

### Q1: How to choose loss weights?

**A**: Adjust according to dataset characteristics:
- **Unstable training**: Reduce `lambda_dol` to 0.3-0.4, increase `lambda_hier` to 2.0
- **Poor tail class performance**: Increase `lambda_scl` to 0.3-0.4
- **Poor head class performance**: Increase `lambda_dol` to 0.6-0.8

### Q2: Too many gradient norm warnings?

**A**: 
- Check if loss weights are too large
- Adjust `--max_grad_norm` (default: 1.0)
- Increase `--grad_warn_threshold` (default: 100.0) or use `--disable_grad_warn`

### Q3: How to choose frontend network?

**A**:
- **ResTcn** (default): Fast training, fewer parameters, suitable for most scenarios
- **ViT**: More expressive, suitable for complex data, but slower training

### Q4: Is it normal for loss to be negative during training?

**A**: Yes, it's normal. The contrastive learning loss (L_scl) can be negative, which is normal and indicates high feature similarity.

### Q5: How to monitor training progress?

**A**:
- Check training logs in the `Log/` directory
- Use `--save_log True` to save detailed logs
- Optional: Use wandb (`--use_wandb`)

### Q6: Where are models saved?

**A**: Models are saved in `Log/{task}/{dataset}-{epochs}_{model}_{timestamp}/` directory

---



---

## 📝 Citation

If you use this project, please cite:

```bibtex
@article{icl2025,
  title={Enhancing Mixture of Experts with Independent and Collaborative Learning for Long-Tail Visual Recognition},
  author={...},
  journal={IJCAI},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

---

## 📄 License

[Add your license information]

---

## 👥 Authors

[Add author information]

---

**Happy Training! 🚀**

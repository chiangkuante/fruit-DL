# AGENTS.md

此檔案為 AI 的指導方針。

## 專案概述

這是一個用於辨識植物病蟲害的深度學習影像分類專案。模型部署於配備 NVIDIA GPU 的 Linux 伺服器上，並作為網頁服務的核心推論引擎。

# 規則
- 請使用 **繁體中文** 回答。

## 系統環境

- **作業系統**: Linux (Ubuntu/Debian)
- **硬體**: 支援 CUDA 的 NVIDIA GPU
- **套件管理**: `uv` (現代 Python 套件管理器)
- **框架**: PyTorch
- **模型庫**: `timm` (PyTorch Image Models)
- **Python 版本**: 3.10+

## 專案設定指令

### 使用 uv 初始化環境
```bash
# 同步環境依賴 (會自動安裝 pyproject.toml 中的套件)
uv sync

# 若需手動安裝 DINOv3 支援的版本
uv add "timm>=1.0.20" PyYAML
```

### 資料集處理
```bash
# 分割資料集 (預設 70/15/15，參數由 config.yaml 讀取)
python split_dataset.py --source-dir disease --copy

# 驗證資料結構
python check_data.py
```

### 訓練與評估
```bash
# 執行訓練 (優先從 config.yaml 讀取超參數與 Resample 設定)
python train.py

# 對獨立測試集進行完整評估
python evaluate.py --model-path output/best_model.pth --data-dir dataset/test
```

## 架構與模型規格

### 設定管理
- **config.yaml**: 統一管理所有關鍵參數。
  - `model`: 名稱與輸入尺寸。
  - `dataset`: `split` (比例與種子) 及 `resample` (固定數量/比例重採樣)。
  - `train`: `batch_size`, `epochs`, `lr`, `weight_decay`。

### 模型架構
- **Backbone**: `vit_large_patch16_dinov3.lvd1689m` (DINOv3 ViT Large)
- **理由**: DINOv3 提供強大的自監督預訓練特徵，適合細微的病徵辨識。
- **訓練方式**: 基於 LVD-1689M 預訓練權重進行遷移學習/微調。

### 資料結構
資料集遵循 `torchvision.datasets.ImageFolder` 格式：
```
dataset/
  ├── train/ (學習用，支援 Resample 平衡)
  ├── val/   (訓練中監控與選取最佳模型用)
  └── test/  (訓練後最終公正評估用)
```

## 訓練輸出產物
1. **模型權重 (`output/best_model.pth`)**: 驗證集準確率最高的版本。
2. **類別映射 (`output/classes.json`)**: 索引與病害名稱的映射表。

## 關鍵實作筆記

### 記憶體管理
- DINOv3 ViT Large 模型龐大（約 1.2GB），若發生 OOM，請調降 `config.yaml` 中的 `batch_size` (建議 8 或 4)。
- 必須啟用 `torch.amp` 混合精度訓練。

### 資料重新採樣 (Resample)
- 透過 `config.yaml` 控制是否開啟 `resample`。
- 支援 `fixed_count` 方法，能將各類別訓練樣本平衡至指定數量（如 200），解決資料不平衡問題。

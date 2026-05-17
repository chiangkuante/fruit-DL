# 植物病蟲害智慧辨識系統

基於 **DINOv3 (ViT Large)** 的深度學習影像分類模型，專為植物病害與蟲害精準辨識而設計。本系統整合了現代化的參數管理與資料平衡技術，提供高效且易於擴充的解決方案。

## 支援的病蟲害類別

本系統目前支援以下 8 種類別（與新資料集 `disease` 同步）：

| 類別 | 中文名稱 | 類型 |
|------|---------|------|
| `healthy` | 健康 | - |
| `algal_leaf_spot` | 藻斑病 | 病害 |
| `anthracnose` | 炭疽病 | 病害 |
| `pest_kanzawa_spider_mite` | 神澤式葉螨 | 蟲害 |
| `pest_mealybug` | 粉介殼蟲 | 蟲害 |
| `pest_tea_mite` | 茶葉螨 | 蟲害 |
| `pest_whitefly` | 番荔枝粉蝨 | 蟲害 |
| `rust` | 銹病 | 病害 |

## 核心特色

- **先進模型**: 採用 Meta 最新推出的 **DINOv3 (ViT Large)** 架構，具備卓越的特徵提取能力。
- **參數設定化**: 透過 `config.yaml` 統一管理模型、訓練、資料分割與重採樣設定，無需更動程式碼。
- **資料平衡 (Resample)**: 內建資料集重採樣功能，支援固定數量或比例平衡各類別樣本，解決資料不均問題。
- **現代化管理**: 使用 `uv` 進行極速依賴管理與環境隔離。

## 安裝與快速開始

### 環境需求
- Python 3.10+
- NVIDIA GPU (建議) 或 CPU
- [uv](https://github.com/astral-sh/uv) 套件管理器

### 安裝步驟

```bash
# 複製專案
git clone <your-repo-url>
cd fruit-DL

# 初始化與同步環境
uv sync
```

## 資料集準備

### 1. 原始資料結構
將照片按類別放入 `disease/` 目錄：
```
disease/
  ├── healthy/
  ├── rust/
  └── ...
```

### 2. 分割資料集
系統會根據 `config.yaml` 中的比例（預設 70/15/15）將資料切分為 Train/Val/Test：
```bash
# 執行分割
python split_dataset.py --copy
```

## 參數設定 (`config.yaml`)

所有關鍵參數皆可在 `config.yaml` 中調整：
```yaml
model:
  name: "vit_large_patch16_dinov3.lvd1689m"
dataset:
  split: { val_ratio: 0.15, test_ratio: 0.15 }
  resample: { enabled: true, target: 200 }
train:
  batch_size: 8
  epochs: 30
  lr: 0.0001
```

## 使用方式

### 1. 訓練模型
```bash
# 開始訓練 (自動讀取 config.yaml)
python train.py

# 或使用 uv 指定參數覆蓋
uv run python train.py --batch-size 4 --epochs 50
```

### 2. 模型評估 (測試集)
```bash
# 使用訓練後最佳模型評估獨立測試集
python evaluate.py --model-path output/best_model.pth --data-dir dataset/test
```

### 3. Web 介面 (Streamlit)
```bash
# 啟動智慧辨識介面
uv run streamlit run app.py
```

## 技術規格

- **模型架構**: DINOv3 ViT Large (`vit_large_patch16_dinov3.lvd1689m`)
- **參數規模**: ~1.2GB 權重檔案
- **優化技術**: AdamW Optimizer, AMP (混合精度訓練)
- **硬體需求**: 
  - 訓練 VRAM: >= 8GB (建議)
  - 推論 VRAM: ~2GB

### **NPUST DN-LAB © 2026**

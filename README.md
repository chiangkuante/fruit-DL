# 植物病蟲害辨識系統

基於 ConvNeXt Large 的深度學習影像分類模型，用於辨識植物病害與蟲害。

## 支援的病蟲害類別

本系統支援以下 9 種類別：

| 類別 | 中文名稱 | 類型 |
|------|---------|------|
| `healthy` | 健康 | - |
| `canker` | 潰瘍病 | 病害 |
| `greasy_spot` | 油斑病 | 病害 |
| `melanose` | 黑點病 | 病害 |
| `sooty_mold` | 煤煙病 | 病害 |
| `pest_aphid` | 蚜蟲 | 蟲害 |
| `pest_leaf_miner` | 潛葉蛾 | 蟲害 |
| `pest_scale_insect` | 介殼蟲 | 蟲害 |
| `pest_thrips` | 薊馬 | 蟲害 |

## 下載資料集

### 資料集分為原始資料集 `disease` 與已經切割train/val完成可直接測試的分割資料集 `dataset` 

原始資料集:
https://github.com/chiangkuante/fruit-DL/releases/download/v1.0/disease.zip

分割資料集:
https://github.com/chiangkuante/fruit-DL/releases/download/v1.0/dataset.zip


## 安裝

### 環境需求

- Python 3.10+
- NVIDIA GPU (CUDA 11.8) 或 CPU

### 方法 1：使用 uv（推薦）

```bash
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步所有依賴
uv sync

# 啟動虛擬環境（可選）
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 方法 2：使用 pip

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 手動安裝依賴

```bash
# 安裝 PyTorch (CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install timm scikit-learn tqdm streamlit pandas altair pillow
```

## 資料集

### 資料集結構

資料必須按以下 ImageFolder 格式分類：

```
disease/
  ├── healthy/
  │   ├── img001.jpg
  │   ├── img002.jpg
  │   └── ...
  ├── canker/
  │   ├── img001.jpg
  │   └── ...
  └── pest_aphid/
      ├── img001.jpg
      └── ...
```

### 分割資料集

使用內建工具將資料分割為訓練集和驗證集（預設 80/20）：

```bash
# 基本用法（移動檔案）
python split_dataset.py --source-dir disease --target-dir dataset

# 複製檔案而非移動
python split_dataset.py --source-dir disease --target-dir dataset --copy

# 自訂分割比例（例如 70/30）
python split_dataset.py --source-dir disease --val-ratio 0.3 --copy
```

分割後結構：
```
dataset/
  ├── train/
  │   ├── healthy/
  │   ├── canker/
  │   └── ...
  └── val/
      ├── healthy/
      ├── canker/
      └── ...
```

### 驗證資料集

```bash
# 檢查資料集完整性
python check_data.py --data-dir dataset
```

## 使用方式

### 1. 訓練模型

```bash
# 基本訓練（使用預設參數）
python train.py

# 自訂訓練參數
python train.py --batch-size 8 --epochs 30 --lr 1e-4

# 使用 uv
uv run python train.py --batch-size 8 --epochs 30
```

**訓練參數：**

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--batch-size` | 8 | 批次大小（若 OOM 可降為 4） |
| `--epochs` | 30 | 訓練週期數 |
| `--lr` | 1e-4 | 學習率 |
| `--data-dir` | dataset | 資料集目錄 |
| `--output-dir` | output | 輸出目錄 |

**訓練輸出：**
```
output/
  ├── best_model.pth      # 最佳模型權重
  ├── classes.json        # 類別映射檔案
  └── checkpoint_*.pth    # 各 epoch 檢查點
```

### 2. 命令列預測

```bash
# 預測單張圖片
python predict.py --image path/to/image.jpg

# 顯示前 5 個預測結果
python predict.py --image path/to/image.jpg --top-k 5

# 預測整個目錄
python predict.py --image path/to/folder/

# 使用 uv
uv run python predict.py --image test.jpg
```

### 3. Web 介面（推薦）

```bash
# 啟動 Streamlit 應用
streamlit run app.py

# 指定端口和地址
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# 使用 uv
uv run streamlit run app.py
```

瀏覽器開啟: `http://localhost:8501`

## 專案結構

```
fruit-DL/
├── app.py                # Streamlit Web 應用
├── train.py              # 模型訓練主程式
├── predict.py            # 預測 API 與命令列工具
├── split_dataset.py      # 資料集分割工具
├── check_data.py         # 資料集驗證工具
├── README.md             # 本文件
├── requirements.txt      # Python 依賴清單
├── pyproject.toml        # uv 專案設定
├── .gitignore            # Git 忽略清單
├── spy.PNG               # Web 介面用圖片
├── output/               # 訓練輸出目錄
│   ├── best_model.pth    # 最佳模型（749MB）
│   ├── classes.json      # 類別映射
│   └── checkpoint_*.pth  # 訓練檢查點
├── dataset/              # 訓練資料集
│   ├── train/            # 訓練集（80%）
│   │   ├── healthy/
│   │   ├── canker/
│   │   └── ...
│   └── val/              # 驗證集（20%）
│       ├── healthy/
│       └── ...
└── disease/              # 原始未分割資料（可選）
    ├── healthy/
    ├── canker/
    └── ...
```

## 技術規格

### 模型

- **架構**: ConvNeXt Large (`convnext_large.fb_in1k`)
- **預訓練**: ImageNet-1k
- **訓練方式**: 遷移學習 / Fine-tuning
- **優化器**: AdamW
- **混合精度**: AMP (Automatic Mixed Precision)
- **準確率**: 97.97% (驗證集)
- **模型大小**: 749MB

### 效能指標

- **推論速度**: ~100ms/張 (GPU)
- **VRAM 需求**:
  - 訓練: ~8GB (batch_size=8)
  - 推論: ~2GB
- **準確率**: 97.97%


### **NPUST DN-LAB © 2025**

# 植物病蟲害辨識系統

基於 ConvNeXt Large 的深度學習影像分類模型，用於辨識植物病害與蟲害。

## 特色功能

- 🎯 **高準確率**：驗證準確率達 97.97%
- 🚀 **多種使用方式**：命令列、Python API、Web 介面
- 🖼️ **即時診斷**：上傳圖片立即獲得結果
- 💡 **智能建議**：自動提供病害處理建議
- 📊 **視覺化**：圖表和進度條展示預測結果

## 快速開始

### 環境需求

- Python 3.10+
- NVIDIA GPU (CUDA 11.8) 或 CPU
- Linux / macOS / Windows

### 安裝

```bash
# 安裝 PyTorch (CUDA 版本)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
uv pip install timm scikit-learn tqdm

# Web 介面 (選用)
uv pip install streamlit pandas
```

## 使用方式

### 1. 命令列預測

```bash
# 預測單張圖片
python predict.py --image path/to/image.jpg

# 預測整個目錄
python predict.py --image path/to/folder/ --top-k 5
```

### 2. Web 介面 (推薦)

```bash
# 啟動 Streamlit 應用
streamlit run app.py
```

開啟瀏覽器訪問 `http://localhost:8501`

**Web 介面功能：**
- 拖放上傳圖片
- 即時病害診斷
- 視覺化預測結果
- 自動病害處理建議
- 可調整預測參數

### 3. Python API

```python
from predict import PlantDiseasePredictor

# 初始化預測器
predictor = PlantDiseasePredictor()

# 預測圖片
predictions = predictor.predict("image.jpg", top_k=3)

# 結果: [('healthy', 99.92), ('canker', 0.03), ...]
```

## 模型訓練

### 準備資料集

資料必須按類別分類：

```
disease/
  ├── 病害A/
  ├── 病害B/
  └── 健康/
```

### 分割資料集

```bash
python split_dataset.py --source-dir disease --target-dir dataset --copy
```

### 開始訓練

```bash
python train.py --batch-size 8 --epochs 30
```

**訓練參數：**

| 參數 | 預設 | 說明 |
|------|------|------|
| `--batch-size` | 8 | 批次大小 |
| `--epochs` | 30 | 訓練週期 |
| `--lr` | 1e-4 | 學習率 |
| `--data-dir` | dataset | 資料集目錄 |
| `--output-dir` | output | 輸出目錄 |

### 訓練輸出

```
output/
  ├── best_model.pth      # 最佳模型
  ├── classes.json        # 類別映射
  └── checkpoint_*.pth    # 各 epoch 檢查點
```

## Streamlit 部署

### 本機部署

```bash
streamlit run app.py --server.port 8501
```

### Docker 部署

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

### 雲端部署

**Streamlit Cloud (免費):**
1. 推送到 GitHub
2. 訪問 [share.streamlit.io](https://share.streamlit.io)
3. 連接儲存庫並部署

**其他平台：**
- AWS EC2 / Google Cloud / Azure
- Heroku / Railway / Render
- 使用 Nginx 反向代理

## API 文檔

### PlantDiseasePredictor 類別

```python
class PlantDiseasePredictor:
    def __init__(
        self,
        model_path: str = 'output/best_model.pth',
        classes_path: str = 'output/classes.json',
        device: Optional[str] = None,
        verbose: bool = True
    )
```

**方法：**

- `predict(image, top_k=3)` - 預測單張圖片
- `predict_batch(images, top_k=3)` - 批次預測
- `get_class_names()` - 取得類別名稱
- `get_model_info()` - 取得模型資訊

### 使用範例

```python
# 初始化
predictor = PlantDiseasePredictor(verbose=False)

# 預測檔案路徑
predictions = predictor.predict("image.jpg")

# 預測 PIL Image
from PIL import Image
img = Image.open("image.jpg")
predictions = predictor.predict(img)

# 批次預測
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = predictor.predict_batch(images)

# 取得資訊
info = predictor.get_model_info()
print(info['accuracy'])  # 97.97
```

## 技術規格

### 模型

- **架構**: ConvNeXt Large (ImageNet-1k 預訓練)
- **訓練方式**: 遷移學習 (Fine-tuning)
- **優化**: AdamW + 混合精度訓練 (AMP)
- **準確率**: 97.97% (驗證集)

### 資料處理

- **訓練增強**: Resize, Flip, Rotation, ColorJitter, Normalize
- **驗證**: Resize, Normalize
- **自動配置**: 使用 timm 自動獲取模型參數

### 效能

- **推論速度**: ~100ms/張 (GPU)
- **模型大小**: 749MB
- **記憶體**: ~2GB VRAM (推論)

## 疑難排解

### CUDA Out of Memory

```bash
# 降低批次大小
python train.py --batch-size 4

# 使用較小模型
python train.py --model-name convnext_base.fb_in1k

# 使用 CPU
predictor = PlantDiseasePredictor(device='cpu')
```

### Streamlit 載入慢

```python
# 使用 @st.cache_resource 快取模型
@st.cache_resource
def load_predictor():
    return PlantDiseasePredictor(verbose=False)
```

### 圖片上傳大小限制

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
```

## 專案結構

```
fruit-DL/
├── train.py              # 訓練主程式
├── predict.py            # 預測 API
├── app.py                # Streamlit 應用
├── split_dataset.py      # 資料分割
├── check_data.py         # 資料驗證
├── example_usage.py      # API 範例
├── README.md             # 專案文檔
├── CLAUDE.md             # 開發指引
├── output/               # 訓練輸出
│   ├── best_model.pth
│   └── classes.json
└── dataset/              # 資料集
    ├── train/
    └── val/
```

## 工具腳本

### 資料集分割

```bash
python split_dataset.py --source-dir disease --val-ratio 0.2 --copy
```

### 資料驗證

```bash
python check_data.py --data-dir dataset
```

### API 範例

```bash
python example_usage.py
```

## 支援的病害類別

本專案預設支援以下 5 種類別（可依需求修改）：

- **canker** (潰瘍病)
- **greasy_spot** (油斑病)
- **healthy** (健康)
- **melanose** (黑點病)
- **sooty_mold** (煤煙病)

## 授權

本專案僅供教學與研究使用。

## 致謝

- 模型：[timm](https://github.com/huggingface/pytorch-image-models)
- 框架：PyTorch
- Web 介面：Streamlit

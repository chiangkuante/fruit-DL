
## 專案需求規格書：植物病蟲害辨識模型訓練模組

### 1\. 專案概述

本專案旨在開發一個基於深度學習的影像分類模型，用於辨識植物的病害與蟲害。模型將部署於具有 NVIDIA GPU 的 Linux 伺服器 (`dnlab-server`) 上，並做為後續 Web 服務的核心推論引擎。

### 2\. 系統環境與技術棧

  * **作業系統**：Linux (Ubuntu/Debian)
  * **硬體資源**：NVIDIA GPU (需支援 CUDA)
  * **套件管理工具**：`uv` (Modern Python Package Manager)
  * **核心框架**：PyTorch
  * **模型庫**：`timm` (PyTorch Image Models)
  * **程式語言**：Python 3.10+

### 3\. 模型規格要求

  * **基礎架構 (Backbone)**：`convnext_large.fb_in1k`
      * *理由*：利用 ConvNeXt Large 強大的特徵提取能力處理細微的病害紋理。
  * **訓練方式**：遷移學習 (Transfer Learning / Fine-tuning)
      * 載入 ImageNet-1k 預訓練權重。
      * 修改全連接層 (Classifier Head) 以適應特定病害類別數。
  * **優化機制**：
      * **混合精度訓練 (AMP)**：使用 `torch.amp` 以降低顯卡記憶體 (VRAM) 需求並加速訓練。
      * **優化器**：使用 `AdamW`。

### 4\. 資料輸入規格

  * **資料來源**：本地資料夾，已按類別分類完成。
  * **資料結構**：符合 `torchvision.datasets.ImageFolder` 標準格式。
    ```text
    dataset/
      ├── train/  (類別A, 類別B, 類別C...)
      └── val/    (類別A, 類別B, 類別C...)
    ```
  * **預處理 (Preprocessing)**：
      * 需使用 `timm.data.resolve_data_config` 自動獲取模型所需的 Input Size、Mean、Std。
      * **訓練集**：需包含資料增強 (Resize, Flip, Rotation, Normalize)。
      * **驗證集**：僅做 Resize 與 Normalize。

### 5\. 輸出產物 (Artifacts)

訓練腳本執行完畢後，必須產出以下檔案：

1.  **模型權重檔 (`.pth`)**：保存驗證集準確率最高 (Best Accuracy) 的模型參數。
2.  **類別映射表 (`classes.json`)**：記錄 `Index (0, 1, 2...)` 對應到的 `Label (病害名稱)`，供後續推論還原使用。
3.  **訓練紀錄**：於 Console 輸出每個 Epoch 的 Loss 與 Accuracy。

-----

## 專案執行 To-Do List

請 AI 依據上述需求書，按順序執行以下任務：

### Phase 1: 環境建置

  - [ ] **Task 1.1**: 撰寫 Shell 指令，使用 `uv` 初始化專案。
  - [ ] **Task 1.2**: 撰寫 `uv pip install` 指令，安裝 `torch` (CUDA版)、`timm`、`scikit-learn`、`tqdm` 等必要套件。

### Phase 2: 資料準備與驗證

  - [ ] **Task 2.1**: 撰寫一個 Python 輔助腳本，檢查 `dataset/` 目錄是否存在，並統計訓練集與驗證集的類別數量與圖片數量，確保資料結構正確。

### Phase 3: 核心訓練程式開發

  - [ ] **Task 3.1**: 撰寫 `train.py`。
      - [ ] 實作 `timm.create_model` 載入 `convnext_large.fb_in1k`。
      - [ ] 實作 `DataLoader` 與 Transform (含 Data Augmentation)。
      - [ ] 實作 Training Loop (包含 AMP 混合精度、Backpropagation)。
      - [ ] 實作 Validation Loop (計算準確率)。
      - [ ] 加入「儲存最佳模型」機制。
      - [ ] **關鍵點**：加入將類別名稱 (`dataset.classes`) 存為 `classes.json` 的功能。

### Phase 4: 參數調整與執行

  - [ ] **Task 4.1**: 設定易於調整的 Hyperparameters 區塊（如 Batch Size, Learning Rate, Epochs）。
  - [ ] **Task 4.2**: 針對顯存不足 (OOM) 情況，提供 Batch Size 的建議調整值（預設建議 8 或 4）。

-----
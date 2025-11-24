# 使用 NVIDIA CUDA 基礎映像
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# 安裝系統依賴和 Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 創建符號連結，讓 python 指向 python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# 安裝 uv 並設置 PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 設定工作目錄
WORKDIR /app

# 複製專案定義檔案（利用 Docker 快取）
COPY pyproject.toml README.md ./

# 使用 uv 安裝依賴（不安裝專案本身）
RUN uv sync --no-dev --no-install-project

# 複製其他專案檔案
COPY . .

# 暴露 Streamlit 端口
EXPOSE 8501

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 啟動 Streamlit 應用
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]

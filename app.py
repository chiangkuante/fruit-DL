#!/usr/bin/env python3
"""
植物病蟲害辨識 Streamlit Web 應用
基於 ConvNeXt Large 深度學習模型
"""

import streamlit as st
from PIL import Image
import pandas as pd
from predict import PlantDiseasePredictor

# ========== 頁面設定 ==========
st.set_page_config(
    page_title="植物病蟲害辨識系統",
    page_icon="🌿",
    layout="wide"
)

# ========== 載入模型 (快取) ==========
@st.cache_resource
def load_predictor():
    """載入預測器 (只執行一次)"""
    return PlantDiseasePredictor(
        model_path='output/best_model.pth',
        classes_path='output/classes.json',
        verbose=False
    )

try:
    predictor = load_predictor()
    model_info = predictor.get_model_info()
except Exception as e:
    st.error(f"❌ 無法載入模型: {e}")
    st.info("請確保 output/best_model.pth 和 output/classes.json 存在")
    st.stop()

# ========== 側邊欄 ==========
with st.sidebar:
    st.header("⚙️ 系統資訊")

    st.subheader("📊 模型狀態")
    st.write(f"**類別數量**: {model_info['num_classes']}")
    st.write(f"**計算裝置**: {model_info['device']}")
    if model_info['accuracy']:
        st.write(f"**模型準確率**: {model_info['accuracy']:.2f}%")

    with st.expander("檢視所有類別"):
        for i, cls in enumerate(model_info['class_names'], 1):
            st.write(f"{i}. {cls}")

    st.markdown("---")

    # 預測參數
    st.subheader("預測設定")
    top_k = st.slider(
        "顯示前 K 個結果",
        min_value=1,
        max_value=model_info['num_classes'],
        value=3
    )

    confidence_threshold = st.slider(
        "信心度閾值 (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="低於此閾值會顯示警告"
    )

    st.markdown("---")
    st.info("💡 支援格式: JPG, JPEG, PNG")

# ========== 主要內容 ==========
st.title("🌿 植物病蟲害智能辨識系統")

st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #555;'>
        使用深度學習技術，快速準確地診斷植物病蟲害
    </p>
</div>
""", unsafe_allow_html=True)

# ========== 檔案上傳 ==========
uploaded_file = st.file_uploader(
    "📤 上傳植物葉片圖片",
    type=['jpg', 'jpeg', 'png'],
    help="請上傳清晰的植物葉片照片以獲得最佳診斷結果"
)

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)

    # 建立兩欄布局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 上傳的圖片")
        st.image(image, use_column_width=True, caption=uploaded_file.name)

        # 圖片資訊
        with st.expander("檢視圖片資訊"):
            st.write(f"**檔案名稱**: {uploaded_file.name}")
            st.write(f"**圖片尺寸**: {image.size[0]} x {image.size[1]} px")
            st.write(f"**圖片格式**: {image.format}")
            st.write(f"**色彩模式**: {image.mode}")

    with col2:
        st.subheader("🔍 診斷結果")

        # 進行預測
        with st.spinner('🧠 AI 正在分析圖片...'):
            predictions = predictor.predict(image, top_k=top_k)

        # 最佳預測結果
        best_class, best_prob = predictions[0]

        # 根據信心度顯示不同訊息
        if best_prob >= confidence_threshold:
            st.success(f"✅ **診斷結果：{best_class}**")
        else:
            st.warning(f"⚠️ **可能診斷：{best_class}** (信心度較低)")

        # 顯示信心度
        st.metric(
            label="診斷信心度",
            value=f"{best_prob:.2f}%",
            delta=f"{best_prob - confidence_threshold:.2f}% vs 閾值"
        )

        # 建議措施
        st.markdown("---")
        st.markdown("### 💡 建議措施")

        disease_recommendations = {
            "healthy": "✅ 葉片健康，繼續保持良好的栽培管理。",
            "canker": "🔴 檢測到潰瘍病，建議：\n- 移除受感染組織\n- 使用銅基殺菌劑\n- 改善通風條件",
            "greasy_spot": "🟡 檢測到油斑病，建議：\n- 噴灑殺菌劑\n- 避免過度灌溉\n- 清除落葉",
            "melanose": "🟠 檢測到黑點病，建議：\n- 使用保護性殺菌劑\n- 修剪過密枝條\n- 注意排水",
            "sooty_mold": "⚫ 檢測到煤煙病，建議：\n- 控制蚜蟲等害蟲\n- 清洗葉面\n- 改善通風"
        }

        recommendation = disease_recommendations.get(
            best_class,
            "請諮詢專業植物病理學家以獲得詳細建議。"
        )
        st.info(recommendation)

    # ========== 詳細分析 ==========
    st.markdown("---")
    st.subheader("📊 詳細分析")

    # 建立 DataFrame
    df = pd.DataFrame(predictions, columns=['類別', '信心度 (%)'])
    df['排名'] = range(1, len(df) + 1)
    df = df[['排名', '類別', '信心度 (%)']]

    # 顯示表格
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # 顯示長條圖
    st.bar_chart(df.set_index('類別')['信心度 (%)'])

else:
    # 未上傳圖片時顯示說明
    st.info("👆 請上傳圖片開始診斷")

    # 使用說明
    with st.expander("📖 使用說明"):
        st.markdown("""
        ### 如何使用本系統

        1. **上傳圖片**：點擊上方的上傳按鈕，選擇植物葉片照片
        2. **等待分析**：系統會自動分析圖片並給出診斷結果
        3. **查看結果**：查看診斷結果、信心度和建議措施
        4. **調整參數**：可在側邊欄調整顯示結果數量和信心度閾值

        ### 拍攝建議

        - 📸 使用清晰的照片
        - 🌞 確保光線充足
        - 🎯 聚焦在病徵區域
        - 📏 保持適當距離（葉片佔畫面 50-80%）

        ### 支援的病害類別

        本系統可辨識以下 5 種類別：
        - 🟢 **healthy** (健康)
        - 🔴 **canker** (潰瘍病)
        - 🟡 **greasy_spot** (油斑病)
        - 🟠 **melanose** (黑點病)
        - ⚫ **sooty_mold** (煤煙病)
        """)

# ========== 頁尾 ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>🌿 植物病蟲害智能辨識系統 v1.0</p>
    <p>使用 ConvNeXt Large 深度學習模型 | 準確率: 97.97%</p>
    <p><small>© 2025 - 僅供教學與研究使用</small></p>
</div>
""", unsafe_allow_html=True)

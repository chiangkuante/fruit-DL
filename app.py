#!/usr/bin/env python3
"""
植物病蟲害辨識 Streamlit Web 應用
基於 DINOv3 (ViT Large) 深度學習模型
"""

import streamlit as st
import yaml
import os
from PIL import Image
import pandas as pd
from predict import PlantDiseasePredictor
import altair as alt
import html


def load_config(config_path="config.yaml"):
    """載入 YAML 設定檔"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

# 載入設定
config = load_config()
model_display_name = config.get('model', {}).get('name', 'DINOv3 (ViT Large)')

# ====== 網頁標題與設定 ======
st.set_page_config(
    page_title="植物病蟲害智慧辨識系統",
    page_icon="🌿",
    layout="wide"
)


# ====== 全站外觀設定 ======
# Streamlit 預設元件樣式較固定，這裡用 CSS 統一調整背景、側邊欄、滑桿與 expander 顏色。
st.markdown("""
    <style>
        /* 整個背景 */
        .stApp {
            background-color: #768f5f;
        }

        /* 把原本的 Streamlit header 壓扁、變透明 */
        [data-testid="stHeader"] {
            background: transparent;
            height: 0px;
        }


        /* 左側 sidebar 背景顏色 */
        [data-testid="stSidebar"] {
            background-color: #52663f;
        }

        /* Sidebar 全部文字改成白色 */
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Sidebar 標題：系統資訊、預測設定 */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #ffffff !important;
        }

        /* Expander 內文字：類別數量、計算裝置、模型準確率、所有類別 */
        [data-testid="stSidebar"] [data-testid="stExpander"] * {
            color: #ffffff !important;
        }

        /* 設定 expander 整體外圍大矩形 */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid #768f5f !important;
            border-radius: 10px !important;
            background-color: #52663f !important;
            overflow: hidden !important;
        }

        /* 外層頂部 bar：佔滿整個寬度 */
        .custom-top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 3rem;
            background-color: #768f5f;
            display: flex;
            align-items: center;
            z-index: 999;
        }

        /* Slider 已填滿的 active track 顏色 */
        [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stTickBar"] > div,
        [data-testid="stSidebar"] [data-baseweb="slider"] [class*="Track"] > div,
        [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
            background-color: #3b4f32 !important;
        }

        /* 滑桿圓形手把的顏色 */
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            background-color: #768f5f !important;
            border-color: #768f5f !important;
        }

        /* 圓點手把：常態的外圈陰影（無 hover 也存在） */
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            box-shadow: 0 0 0 3px #3b4f32 !important;
        }

        /* Slider 上方/下方顯示的數字與文字顏色 */
        [data-testid="stSidebar"] [data-baseweb="slider"] * {
            color: #FFFFFF !important;
        }
            
        /* 涵蓋各種版本的 tick 標記 /
        [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="TickBar"],
        [data-testid="stSidebar"] [data-baseweb="slider"] [class="tick"],
        [data-testid="stSidebar"] [data-baseweb="slider"] [class="Tick"] {
            color: #3b4f32 !important;
            background-color: #3b4f32 !important;
        }

        

        /* 修改 expander 標題（上方 summary）字體與背景 */
        details > summary {
            background-color: transparent !important; /* 背景透明，與外框矩形融為一體 */
            color: white !important;
            font-weight: bold !important;            /* 標題字體加粗 */
        }
        
        /* 調整 expander 展開時，標題與內容之間的分隔線顏色 */
        [data-testid="stExpander"] details[open] > summary {
            border-bottom: 1px solid #768f5f !important;
        }
        
        /* 移除內容區域的多餘邊框，保持整體矩形的乾淨 */
        [data-testid="stExpander"] details > div {
            border: none !important;
        }

        /* 圖片資訊 expander：與建議措施色塊一致 */
        .image-info-expander {
            background-color: #52663f;
            color: #ffffff;
            border-radius: 10px;
            margin-top: 0.5rem;
            overflow: hidden;
        }

        .image-info-expander summary {
            background-color: #52663f !important;
            color: #ffffff !important;
            cursor: pointer;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }

        .image-info-expander[open] summary {
            border-bottom: 1px solid #768f5f;
        }

        .image-info-expander div {
            background-color: #52663f;
            color: #ffffff;
            padding: 0.8rem 1rem;
            line-height: 1.8;
        }
        /* 調整 st.metric 裡 delta 背景與文字顏色 */
        [data-testid="stMetricDelta"] {
            background-color: #d8ffd3 !important;
            border-radius: 999px;
            padding: 0.15rem 0.4rem;
            width: fit-content;
        }

        [data-testid="stMetricDelta"] > div {
            color: #3b4f32 !important;
            font-weight:550;
        }
         /* 改變上升箭頭顏色 */
        [data-testid="stMetricDelta"] svg {
            fill: #3b4f32 !important;
            color: #3b4f32 !important;
        }

        /* 修正錯誤訊息和資訊框的文字溢出問題 */
        .stAlert {
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }

        /* 修正檔案上傳器的文字溢出問題 */
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] div {
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }

    </style>
""", unsafe_allow_html=True)


# ========== 疾病名稱中文映射 ==========
# 模型輸出的是英文類別名稱，畫面顯示時透過這個字典轉成中文。
DISEASE_NAME_ZH = {
    "healthy": "健康",
    "anthracnose": "炭疽病",
    "algal_leaf_spot": "藻斑病",
    "rust": "銹病",
    "pest_whitefly": "番荔枝粉蝨",
    "pest_kanzawa_spider_mite": "神澤氏葉蟎",
    "pest_mealybug": "粉介殼蟲",
    "pest_tea_mite": "茶葉蟎",
}

# ========== 載入模型 (快取) ==========
@st.cache_resource
def load_predictor():
    """載入預測器。

    st.cache_resource 會快取模型物件，避免每次互動都重新載入模型檔。
    """
    return PlantDiseasePredictor(
        model_path='output/best_model.pth',
        classes_path='output/classes.json',
        verbose=False
    )

try:
    # 啟動 App 時先載入模型，並取得類別數量、裝置與準確率等資訊。
    predictor = load_predictor()
    model_info = predictor.get_model_info()
except Exception as e:
    # 若模型檔或類別檔不存在，停止後續流程，避免預測階段才報錯。
    st.error(f"無法載入模型: {e}")
    st.info("請確保模型存在 (output/best_model.pth 、 output/classes.json) ")
    st.stop()

# ========== 側邊欄 ==========
with st.sidebar:
    st.header("系統資訊")

    # 模型狀態集中放在 expander，讓側邊欄保持簡潔。
    with st.expander("模型狀態", expanded=False):  # expanded=True 代表預設展開
        st.write(f"類別數量: {model_info['num_classes']}")
        st.write(f"計算裝置: {model_info['device']}")
        if model_info['accuracy']:
            st.write(f"模型準確率: {model_info['accuracy']:.2f}%")

    # 顯示模型支援的所有類別，並同步轉成中文名稱。
    with st.expander("檢視所有類別"):
        for i, cls in enumerate(model_info['class_names'], 1):
            cls_zh = DISEASE_NAME_ZH.get(cls, cls)
            st.write(f"{i}. {cls_zh}")

    st.markdown("---")

    # 預測參數：top_k 控制顯示幾個候選結果，threshold 用來提示信心度是否偏低。
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

    # 在側邊欄最下方放插圖，增加畫面識別度。
    st.markdown("---")
    st.image("spy.PNG", width="stretch")



# ========== 檔案上傳 ==========
# 使用者上傳圖片後，Streamlit 會重新執行整支程式，uploaded_file 會保存目前上傳的檔案。
uploaded_file = st.file_uploader(
    "上傳植物葉片照片",
    type=['jpg', 'jpeg', 'png'],
    help="請上傳清晰的植物葉片照片以獲得最佳診斷結果",
    label_visibility="hidden"
)

if uploaded_file is not None:
    # 將上傳檔案轉成 PIL Image，供模型預測與畫面顯示使用。
    image = Image.open(uploaded_file)

    # 建立兩欄布局：左側放原始圖片，右側放診斷結果。
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上傳的圖片")
        st.image(image, caption=uploaded_file.name)

        # 圖片資訊放在 expander，避免主要畫面被細節佔滿。
        file_name = html.escape(uploaded_file.name)
        image_format = html.escape(str(image.format))
        image_mode = html.escape(str(image.mode))
        st.markdown(
            f"""
            <details class="image-info-expander">
                <summary>檢視圖片資訊</summary>
                <div>
                    檔案名稱: {file_name}<br>
                    圖片尺寸: {image.size[0]} x {image.size[1]} px<br>
                    圖片格式: {image_format}<br>
                    色彩模式: {image_mode}
                </div>
            </details>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("診斷結果")

        # 呼叫預測器取得前 top_k 個分類結果，格式為 [(類別名稱, 信心度), ...]。
        with st.spinner('AI 正在分析圖片...'):
            predictions = predictor.predict(image, top_k=top_k)

        # 第一筆是信心度最高的預測，作為主要診斷結果。
        best_class, best_prob = predictions[0]
        best_class_zh = DISEASE_NAME_ZH.get(best_class, best_class)
        display_name = f"{best_class_zh} ({best_class})"

        # 若最高信心度低於使用者設定的閾值，標題會提醒結果較不確定。
        if best_prob >= confidence_threshold:
            result_bg = "#52663f"
            result_title = "診斷結果"
        else:
            result_bg = "#52663f"
            result_title = "可能診斷（信心度較低）"

        st.markdown(
        f"""
        <div style="
            background-color:{result_bg};
            border-radius:10px;
            padding:0.8rem 1.0rem;
            color:#ffffff;
            font-weight:600;
            font-size:1.05rem;
            margin-bottom:0.8rem;
        ">
            {result_title}：{display_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


        # st.metric 顯示目前信心度，delta 用來比較和閾值的差距。
        st.metric(
            label="診斷信心度",
            value=f"{best_prob:.2f}%",
            delta=f"{best_prob - confidence_threshold:.2f}% vs 閾值"
        )

        # 建議措施會依照最佳預測類別切換內容。
        st.markdown("---")
        st.markdown("### 建議措施")

        disease_recommendations = {
            "healthy": "葉片健康，繼續保持良好的栽培管理。",
            "anthracnose": "檢測到炭疽病，建議：\n- 剪除受害病葉與枯枝\n- 噴灑適當殺菌劑（如波爾多液）\n- 注意通風與排水",
            "algal_leaf_spot": "檢測到藻斑病，建議：\n- 改善通風與光照條件\n- 減少樹冠過度潮濕\n- 視情況使用銅劑進行防治",
            "rust": "檢測到銹病，建議：\n- 清除病殘體以減少感染源\n- 噴灑推薦的抗銹病殺菌劑\n- 避免氮肥過量導致嫩葉過多",
            "pest_whitefly": "檢測到番荔枝粉蝨，建議：\n- 使用黃色黏蟲板監測與誘殺\n- 噴灑礦物油或核准的殺蟲劑\n- 移除雜草以減少寄生源",
            "pest_kanzawa_spider_mite": "檢測到神澤氏葉蟎，建議：\n- 保持環境適當濕度，避免過於乾燥\n- 使用殺蟎劑交替防治以避免抗藥性\n- 保護天敵（如捕植蟎）",
            "pest_mealybug": "檢測到粉介殼蟲，建議：\n- 修剪受害嚴重的枝條\n- 使用系統性殺蟲劑或夏油噴灑\n- 防治共生螞蟻以減少擴散",
            "pest_tea_mite": "檢測到茶葉蟎，建議：\n- 加強嫩葉期監測\n- 使用推薦的殺蟎劑\n- 移除附近可能的寄主植物",
            "canker": "檢測到潰瘍病，建議：\n- 移除受感染組織\n- 使用銅基殺菌劑\n- 改善通風條件",
            "greasy_spot": "檢測到油斑病，建議：\n- 噴灑適當殺菌劑\n- 避免過度灌溉與葉面長期潮濕\n- 清除嚴重受害落葉",
            "melanose": "檢測到黑點病，建議：\n- 使用保護性殺菌劑\n- 修剪過密枝條\n- 注意排水與通風",
            "sooty_mold": "檢測到煤煙病，建議：\n- 先控制蚜蟲、介殼蟲等分泌蜜露的害蟲\n- 視情況清洗葉面\n- 改善園區通風與採光",
            "pest_aphid": "檢測到蚜蟲危害，建議：\n- 針對嫩梢與葉背進行防治\n- 可使用皂素、礦物油 or 選擇性殺蟲劑\n- 避免氮肥過量以減少嫩梢暴露",
            "pest_leaf_miner": "檢測到潛葉蛾危害，建議：\n- 剪除嚴重受害葉片\n- 適時使用系統性殺蟲劑\n- 監測成蟲發生期以提早防治",
            "pest_scale_insect": "檢測到介殼蟲危害，建議：\n- 修剪嚴重受害枝條\n- 使用礦物油或合適殺蟲劑\n- 搭配天敵保育降低族群密度",
            "pest_thrips": "檢測到薊馬危害，建議：\n- 加強花期與嫩葉期監測\n- 適時使用選擇性殺蟲劑\n- 搭配黃色/藍色黏蟲板監控族群變化",
        }

        # 預留不同類別使用不同顏色的彈性；目前先統一成綠色系。
        disease_colors = {
            "healthy": ("#52663f", "#ffffff"),
            "anthracnose": ("#52663f", "#ffffff"),
            "algal_leaf_spot": ("#52663f", "#ffffff"),
            "rust": ("#52663f", "#ffffff"),
            "pest_whitefly": ("#52663f", "#ffffff"),
            "pest_kanzawa_spider_mite": ("#52663f", "#ffffff"),
            "pest_mealybug": ("#52663f", "#ffffff"),
            "pest_tea_mite": ("#52663f", "#ffffff"),
            "canker": ("#52663f", "#ffffff"),
            "greasy_spot": ("#52663f", "#ffffff"),
            "melanose": ("#52663f", "#ffffff"),
            "sooty_mold": ("#52663f", "#ffffff"),
            "pest_aphid": ("#52663f", "#ffffff"),
            "pest_leaf_miner": ("#52663f", "#ffffff"),
            "pest_scale_insect": ("#52663f", "#ffffff"),
            "pest_thrips": ("#52663f", "#ffffff"),
        }

        recommendation = disease_recommendations.get(
            best_class,
            "請諮詢專業植物病理學家以獲得詳細建議。"
        )
        bg_color, text_color = disease_colors.get(best_class, ("#52663f", "#ffffff"))
        
        # 用自訂 HTML 色塊顯示建議內容，white-space: pre-line 可保留建議文字中的換行。
        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                color:{text_color};
                border-radius:10px;
                padding:0.8rem 1.0rem;
                white-space:pre-line;
                font-size:0.93rem;    
            ">{recommendation}</div>""",
            unsafe_allow_html=True,
        )

# ========== 詳細分析 ==========
    st.markdown("---")
    st.subheader("詳細分析")

    # 將模型回傳的預測結果轉成表格資料，方便同時顯示排名、類別與信心度。
    predictions_zh = [(DISEASE_NAME_ZH.get(cls, cls), prob) for cls, prob in predictions]
    df = pd.DataFrame(predictions_zh, columns=['類別', '信心度 (%)'])
    df['排名'] = range(1, len(df) + 1)
    df = df[['排名', '類別', '信心度 (%)']]

    # --------- 表格：整體顏色風格 ---------
    # pandas Styler 可替 dataframe 套用標題列與資料列樣式。
    # styled_df = (
    #     df.style
    #     # 標題列樣式
    #     .set_table_styles([
    #         {
    #             "selector": "th",
    #             "props": [
    #                 ("background-color", "#ebf1e5"),  # 標題列底色
    #                 ("color", "#000000"),             # 標題文字顏色
    #                 ("font-weight", "600"),
    #                 ("text-align", "right"),
    #             ],
    #         }
    #     ])
    #     # 資料列樣式
    #     .set_properties(**{
    #         "background-color": "#52663f",  # 每一列底色
    #         "color": "#ffffff",             # 每一列文字顏色
    #         "border-color": "#768f5f",
    #         "text-align": "right",
    #     })
    # )

    # st.dataframe(
    #     styled_df,
    #     width='stretch',
    #     hide_index=True,
    # )

    # --------- 表格：改用 HTML 渲染，徹底控制對齊與樣式 ---------
    styled_df = (
        df.style
        .hide(axis="index")  # 隱藏 index
        .format({"信心度 (%)": "{:.2f}"})  # 信心度只顯示 2 位小數
        .set_table_styles([
            {
                "selector": "",  # 整個表格
                "props": [
                    ("width", "100%"),
                    ("border-collapse", "separate"),
                    ("border-spacing", "0"),
                    ("border-radius", "6px"),
                    ("overflow", "hidden"),
                    ("font-size", "0.90rem"),
                    ("margin-bottom", "1rem"),
                ],
            },
            {
                "selector": "thead th",  # 標題列
                "props": [
                    ("background-color", "#ebf1e5"),
                    ("color", "#000000"),
                    ("font-weight", "500"),
                    ("text-align", "right"),
                    ("padding", "0.6rem 1rem"),
                ],
            },
            {
                "selector": "tbody td",  # 資料儲存格
                "props": [
                    ("background-color", "#52663f"),
                    ("color", "#ffffff"),
                    ("text-align", "right"),   # ← 全部右對齊
                    ("padding", "0.55rem 1rem"),
                    ("border-top", "1px solid #768f5f"),
                ],
            },
        ])
    )

    st.markdown(styled_df.to_html(), unsafe_allow_html=True)

    # --------- 長條圖：整體顏色風格（Altair） ---------
    # 用長條圖讓各類別信心度更容易比較。
    chart = (
        alt.Chart(df)
        .mark_bar(color="#3b4f32")
        .encode(
            x=alt.X(
                "類別:N",
                sort="-y",
                axis=alt.Axis(
                    title=None,
                    labelAngle=0,
                    labelFontSize=14,   # ← x 軸文字大小
                ),
            ),
            y=alt.Y(
                "信心度 (%):Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title=None),
            ),
        )
        .properties(
            height=260,
            width=600,              
            background="#52663f",
            padding={"left": 20, "right": 25, "top": 10, "bottom": 8},
        )
        .configure_view(
            strokeWidth=0,
        )
        .configure_axis(
            grid=True,
            gridColor="#768f5f",
            gridOpacity=0.6,
            labelColor="#ffffff",
            tickColor="#ffffff",
        )
        .interactive()             # ← 啟用拖曳、縮放
    )

    st.altair_chart(chart, width='stretch')


    # 尚未上傳圖片時，顯示開始提示與使用說明。
    st.markdown(
        "<p style='text-align:center; color:#ffffff;background-color: #3b4f32; border-radius:10px; padding:0.6rem 1rem;     '> 請上傳圖片開始診斷</p>",
        unsafe_allow_html=True,
    )


    # 使用說明預設收合，讓使用者需要時再展開查看。
    
    with st.expander("使用說明"):
        st.markdown("""
        ### 如何使用本系統

        1. **上傳圖片**：點擊上方的上傳按鈕，選擇植物葉片照片
        2. **等待分析**：系統會自動分析圖片並給出診斷結果
        3. **查看結果**：查看診斷結果、信心度和建議措施
        4. **調整參數**：可在側邊欄調整顯示結果數量和信心度閾值

        ### 拍攝建議

        - 使用清晰的照片
        - 確保光線充足
        - 聚焦在病徵區域
        - 保持適當距離（葉片佔畫面 50-80%）

        ### 支援的病害類別

        本系統目前可辨識以下 8 種類別：
        - **healthy** (健康)
        - **anthracnose** (炭疽病)
        - **algal_leaf_spot** (藻斑病)
        - **rust** (銹病)
        - **pest_whitefly** (番荔枝粉蝨)
        - **pest_kanzawa_spider_mite** (神澤氏葉蟎)
        - **pest_mealybug** (粉介殼蟲)
        - **pest_tea_mite** (茶葉蟎)

        """)



# ========== 頁尾 ==========
# 固定顯示系統名稱與模型資訊。
st.markdown(f"""
<div style='text-align: center; color: #000000; padding: 1rem;'>
    <p>植物病蟲害智慧辨識系統 v1.1</p>
    <p>使用 {model_display_name} 深度學習模型</p>
    <p>NPUST DN-LAB 2026</p>
</div>
""", unsafe_allow_html=True)

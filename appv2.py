import streamlit as st
import pandas as pd
import joblib
import io
import os

@st.cache_resource
def load_assets():
    # 町名とスコアの辞書
    town_mapping = joblib.load('town_mapping.joblib')
    combined_data = b""
    # 4分割モデルの読み込み
    for i in range(4):
        file_name = f"tokyo_price_v1_part{i}.pkl"
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                combined_data += f.read()
    
    if not combined_data:
        st.error("モデルファイル(part0〜part3)が見つかりません。")
        return town_mapping, None

    model = joblib.load(io.BytesIO(combined_data))
    return town_mapping, model

# データの読み込み
town_mapping, model = load_assets()

# --- 画面設定 ---
st.set_page_config(page_title="23区マンション価格AI査定", layout="wide")

# CSSによる見た目の調整（フォントサイズなど）
st.markdown("""
    <style>
    .main-title { font-size: 32px !important; font-weight: bold; color: #1E3A8A; margin-bottom: 20px; }
    .result-label { font-size: 18px !important; color: #6B7280; }
    .result-value { font-size: 36px !important; font-weight: bold; color: #1D4ED8; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🏙️ 23区マンションAI価格査定</p>', unsafe_allow_html=True)

# --- 1. 物件スペック設定（サイドバー） ---
st.sidebar.header("📏 物件スペック")
size = st.sidebar.slider("専有面積 (㎡)", 10.0, 200.0, 60.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2026, 2010)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)
age = 2026 - built_year

# --- 2. メイン画面のタブ構成 ---
tab1, tab2 = st.tabs(["📍 地点指定査定", "🏆 価格ランキング"])

# --- Tab 1: 地点指定査定 ---
with tab1:
    st.write("### 📍 所在地を設定してください")
    ward_list = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"]
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        selected_ward = st.selectbox("区を選択", ward_list, index=3)
    
    all_towns = sorted(list(town_mapping.keys()))
    filtered_full_towns = [t for t in all_towns if t.startswith(selected_ward)]
    display_to_full = {t.replace(selected_ward, ""): t for t in filtered_full_towns}
    display_town_list = list(display_to_full.keys())

    with col_w2:
        default_target_name = "西新宿"
        initial_index = 0
        if default_target_name in display_town_list:
            initial_index = display_town_list.index(default_target_name)
        selected_town_display = st.selectbox("町名を選択", display_town_list, index=initial_index)

    if model is not None:
        selected_town_full = display_to_full[selected_town_display]
        town_score = town_mapping[selected_town_full]
        input_df = pd.DataFrame([[size, age, walk_min, town_score]], columns=['size', 'age', 'walk', 'town_score'])
        predicted_price = int(model.predict(input_df.values)[0])

        st.divider()
        st.markdown(f"### 📊 {selected_ward} {selected_town_display} の査定結果")
        
        # 賃貸アプリ風のメトリック表示
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="result-label">AI査定価格</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{predicted_price:,} 円</p>', unsafe_allow_html=True)
        with c2:
            st.markdown('<p class="result-label">予測平米単価</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{int(predicted_price / size):,} 円/㎡</p>', unsafe_allow_html=True)
            
        st.info(f"💡 条件: {size}㎡ / 築{age}年 / 徒歩{walk_min}分")

# --- Tab 2: 23区価格ランキング ---
with tab2:
    st.write(f"### 🏆 {size}㎡ / 築{age}年 / 徒歩{walk_min}分の価格順位")
    order = st.radio("表示順", ["価格が安い順", "価格が高い順"], horizontal=True)

    if st.button("ランキングを表示"):
        if model is not None:
            with st.spinner('AIが23区すべての地点を査定中...'):
                results = []
                # ユーザーの要望通り、ランキングデータからカッコを除去
                for addr, ts in town_mapping.items():
                    X = pd.DataFrame([[size, age, walk_min, ts]], columns=['size', 'age', 'walk', 'town_score'])
                    pred = model.predict(X.values)[0]
                    # 表示用に地名を加工
                    clean_addr = addr.replace("(", "").replace(")", "")
                    results.append({"地点名": clean_addr, "予測価格": int(pred), "平米単価": int(pred/size)})
                
                df_res = pd.DataFrame(results)
                
                if "安い順" in order:
                    df_res = df_res.sort_values("予測価格").head(20)
                else:
                    df_res = df_res.sort_values("予測価格", ascending=False).head(20)
                
                df_res.index = range(1, len(df_res) + 1)
                
                # テーブル表示
                st.table(df_res.style.format({"予測価格": "{:,} 円", "平米単価": "{:,} 円/㎡"}))
        else:
            st.error("モデルが読み込まれていません。")

st.markdown("---")
st.caption("※2026年時点の統計データに基づくAI推計値です。")

import streamlit as st
import pandas as pd
import joblib
import io
import os

# 2026年基準のままとします
CURRENT_YEAR = 2026

@st.cache_resource
def load_assets():
    # 以前作成した賃貸用のマッピングと10分割モデルを読み込む設定
    # ※ファイル名はご自身の環境に合わせて適宜修正してください
    town_mapping = joblib.load('town_mapping.joblib')
    combined_data = b""
    # ユーザー様の環境に合わせて10分割（range(10)）としておきます
    for i in range(10):
        file_name = f"model_rent_v4_part{i}.pkl"
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                combined_data += f.read()
    model = joblib.load(io.BytesIO(combined_data))
    return town_mapping, model

town_mapping, model = load_assets()

st.set_page_config(page_title="23区賃貸AI査定", layout="centered")
st.title("🏙️ 23区賃貸AI家賃査定")

# --- スペック設定（サイドバーに残す部分） ---
st.sidebar.header("📏 予測条件の設定")
size = st.sidebar.slider("面積 (㎡)", 10.0, 200.0, 25.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2026, 2015)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)

# --- 所在地設定（サイドバーからメイン画面へ移動） ---
st.write("### 📍 所在地を設定してください")
ward_list = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"]

col_w1, col_w2 = st.columns(2)

with col_w1:
    selected_ward = st.selectbox("区を選択", ward_list, index=3) # デフォルト新宿区

# 町名の絞り込み処理
all_towns = sorted(list(town_mapping.keys()))
filtered_full_towns = [t for t in all_towns if t.startswith(selected_ward)]
display_to_full = {t.replace(selected_ward, ""): t for t in filtered_full_towns}
display_town_list = list(display_to_full.keys())

with col_w2:
    selected_town_display = st.selectbox("町名を選択", display_town_list)

# AI計算用のフルネーム
selected_town_full = display_to_full[selected_town_display]

# --- 予測計算（ロジックは変更なし） ---
age = CURRENT_YEAR - built_year
town_score = town_mapping[selected_town_full]
# カラム名は学習時の形式に合わせる（例：'size', 'age', 'walk', 'town_score'）
input_df = pd.DataFrame([[size, age, walk_min, town_score]], columns=['size', 'age', 'walk', 'town_score'])
predicted_price = int(model.predict(input_df.values)[0])

# --- 査定結果表示 ---
st.divider()
st.subheader(f"📊 {selected_ward} {selected_town_display} の査定結果")

col1, col2 = st.columns(2)
with col1:
    st.metric("予測賃料", f"{predicted_price:,} 円")
with col2:
    st.metric("予測平米単価", f"{int(predicted_price / size)::,} 円/㎡")

st.info(f"条件: {size}㎡ / 築{age}年 / 徒歩{walk_min}分")
st.caption("※2026年時点の賃料予測値です。")

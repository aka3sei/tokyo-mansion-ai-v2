import streamlit as st
import pandas as pd
import joblib
import io

@st.cache_resource
def load_assets():
    town_mapping = joblib.load('town_mapping.joblib')
    combined_data = b""
    for i in range(4):
        with open(f"tokyo_price_v1_part{i}.pkl", "rb") as f:
            combined_data += f.read()
    model = joblib.load(io.BytesIO(combined_data))
    return town_mapping, model

town_mapping, model = load_assets()

st.title("🏙️ 23区マンションAI価格査定")

# --- 所在地設定 ---
st.sidebar.header("📍 所在地設定")

ward_list = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"]

# 1. 区の選択
selected_ward = st.sidebar.selectbox("区を選択", ward_list, index=3) # デフォルト新宿区

# 2. 町名の絞り込み（「新宿区」で始まる町名だけを表示）
all_towns = sorted(list(town_mapping.keys()))
filtered_towns = [t for t in all_towns if t.startswith(selected_ward)]

# デフォルト設定（西新宿）
default_target = f"{selected_ward}西新宿"
initial_index = 0
if default_target in filtered_towns:
    initial_index = filtered_towns.index(default_target)

selected_town = st.sidebar.selectbox("町名を選択", filtered_towns, index=initial_index)

# --- スペック設定 ---
st.sidebar.divider()
st.sidebar.header("📏 物件スペック")
size = st.sidebar.slider("専有面積 (㎡)", 10.0, 200.0, 60.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2025, 2010)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)

# --- 予測 ---
age = 2026 - built_year
town_score = town_mapping[selected_town]
input_df = pd.DataFrame([[size, age, walk_min, town_score]], columns=['size', 'age', 'walk', 'town_score'])
predicted_price = int(model.predict(input_df)[0])

# --- 表示 ---
st.subheader(f"📊 {selected_town} の査定結果")
col1, col2 = st.columns(2)
with col1:
    st.metric("AI査定価格", f"{predicted_price:,} 円")
with col2:
    st.metric("予測平米単価", f"{int(predicted_price / size):,} 円/㎡")

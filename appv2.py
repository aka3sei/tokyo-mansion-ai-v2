import streamlit as st
import pandas as pd
import joblib
import io

@st.cache_resource
def load_assets():
    # 町名が「新宿区西新宿」のような形式で入っていることを想定
    town_mapping = joblib.load('town_mapping.joblib')
    combined_data = b""
    for i in range(4):
        file_name = f"tokyo_price_v1_part{i}.pkl"
        with open(file_name, "rb") as f:
            combined_data += f.read()
    model = joblib.load(io.BytesIO(combined_data))
    return town_mapping, model

town_mapping, model = load_assets()

st.set_page_config(page_title="23区マンションAI査定", layout="centered")
st.title("🏙️ 23区マンションAI価格査定")

# --- 所在地設定 ---
st.sidebar.header("📍 所在地設定")

ward_list = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"]

# 1. 区の選択
selected_ward = st.sidebar.selectbox("区を選択", ward_list, index=3) # デフォルト新宿区

# 2. 町名の絞り込みと表示の加工
all_towns = sorted(list(town_mapping.keys()))
# 選択された区で始まるフルネームのリストを作成（例：['新宿区西新宿', '新宿区北新宿', ...]）
filtered_full_towns = [t for t in all_towns if t.startswith(selected_ward)]

# 表示用に「区名」をカットした辞書を作成（例：{'西新宿': '新宿区西新宿', ...}）
display_to_full = {t.replace(selected_ward, ""): t for t in filtered_full_towns}
display_town_list = list(display_to_full.keys())

# デフォルト設定（西新宿）
default_target_name = "西新宿"
initial_index = 0
if default_target_name in display_town_list:
    initial_index = display_town_list.index(default_target_name)

# 【修正ポイント】ユーザーには「西新宿」だけ見せる
selected_town_display = st.sidebar.selectbox("町名を選択", display_town_list, index=initial_index)

# AIの計算には「新宿区西新宿」というフルネームを使用する
selected_town_full = display_to_full[selected_town_display]

# --- スペック設定 ---
st.sidebar.divider()
st.sidebar.header("📏 物件スペック")
size = st.sidebar.slider("専有面積 (㎡)", 10.0, 200.0, 60.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2025, 2010)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)

# --- 予測 ---
age = 2026 - built_year
# フルネームでスコアを引く
town_score = town_mapping[selected_town_full]
input_df = pd.DataFrame([[size, age, walk_min, town_score]], columns=['size', 'age', 'walk', 'town_score'])
predicted_price = int(model.predict(input_df)[0])

# --- 表示 ---
# タイトル部分も「中央区 八丁堀」のように分けると見やすいです
st.subheader(f"📊 {selected_ward} {selected_town_display} の査定結果")

col1, col2 = st.columns(2)
with col1:
    st.metric("AI査定価格", f"{predicted_price:,} 円")
with col2:
    st.metric("予測平米単価", f"{int(predicted_price / size):,} 円/㎡")

st.info("※2026年時点の統計推計値です。")

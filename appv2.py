import streamlit as st
import pandas as pd
import joblib
import io
import os

@st.cache_resource
def load_assets():
    # 町名が「新宿区西新宿」のような形式で入っていることを想定
    town_mapping = joblib.load('town_mapping.joblib')
    combined_data = b""
    # 10分割モデル(価格予想用)を読み込む設定
    for i in range(10):
        file_name = f"tokyo_price_v1_part{i}.pkl"
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                combined_data += f.read()
    
    if not combined_data:
        st.error("モデルファイルが見つかりません。")
        return town_mapping, None

    model = joblib.load(io.BytesIO(combined_data))
    return town_mapping, model

# データの読み込み
town_mapping, model = load_assets()

st.set_page_config(page_title="23区マンションAI査定", layout="centered")
st.title("🏙️ 23区マンションAI価格査定")

# --- 1. スペック設定（サイドバー） ---
st.sidebar.header("📏 物件スペック")
size = st.sidebar.slider("専有面積 (㎡)", 10.0, 200.0, 60.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2025, 2010)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)

# --- 2. 所在地設定（サイドバーからメイン画面へ移動） ---
st.write("### 📍 所在地を設定してください")
ward_list = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"]

col_w1, col_w2 = st.columns(2)

with col_w1:
    # 1. 区の選択
    selected_ward = st.selectbox("区を選択", ward_list, index=3) # デフォルト新宿区

# 2. 町名の絞り込みと表示の加工
all_towns = sorted(list(town_mapping.keys()))
filtered_full_towns = [t for t in all_towns if t.startswith(selected_ward)]
display_to_full = {t.replace(selected_ward, ""): t for t in filtered_full_towns}
display_town_list = list(display_to_full.keys())

# デフォルト設定（西新宿）のインデックス計算
default_target_name = "西新宿"
initial_index = 0
if default_target_name in display_town_list:
    initial_index = display_town_list.index(default_target_name)

with col_w2:
    # 【修正ポイント】ユーザーには「西新宿」だけ見せる
    selected_town_display = st.selectbox("町名を選択", display_town_list, index=initial_index)

# AIの計算には「新宿区西新宿」というフルネームを使用する
selected_town_full = display_to_full[selected_town_display]

# --- 3. 予測計算 ---
if model is not None:
    age = 2026 - built_year
    town_score = town_mapping[selected_town_full]
    # 入力形式とカラム名は変更なし
    input_df = pd.DataFrame([[size, age, walk_min, town_score]], columns=['size', 'age', 'walk', 'town_score'])
    predicted_price = int(model.predict(input_df.values)[0])

    # --- 4. 査定結果表示 ---
    st.divider()
    st.subheader(f"📊 {selected_ward} {selected_town_display} の査定結果")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("AI査定価格", f"{predicted_price:,} 円")
    with col2:
        st.metric("予測平米単価", f"{int(predicted_price / size)::,} 円/㎡")

    st.info(f"条件: {size}㎡ / 築{age}年 / 徒歩{walk_min}分")
    st.caption("※2026年時点の統計推計値です。")
else:
    st.warning("モデルの読み込みに失敗しているため、査定できません。")

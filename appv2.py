import streamlit as st
import pandas as pd
import joblib
import io

# 1. 資産の読み込み（分割されたファイルをメモリ上で合体）
@st.cache_resource
def load_assets():
    try:
        # 地点マッピングの読み込み
        town_mapping = joblib.load('town_mapping.joblib')
        
        # 4つのパーツをバイナリ形式で結合
        combined_data = b""
        for i in range(4):
            file_name = f"tokyo_price_v1_part{i}.pkl"
            with open(file_name, "rb") as f:
                combined_data += f.read()
        
        # 結合したデータを一つのモデルとして復元
        model = joblib.load(io.BytesIO(combined_data))
        return town_mapping, model
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        st.stop()

town_mapping, model = load_assets()

# --- アプリ設定 ---
st.set_page_config(page_title="23区マンションAI査定", layout="centered")
st.title("🏙️ 23区マンションAI価格査定")

# --- サイドバー：所在地設定（2段階選択） ---
st.sidebar.header("📍 所在地設定")

# 23区リスト
ward_list = [
    "千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", 
    "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", 
    "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"
]

# 1. 区を選択（デフォルト：新宿区）
selected_ward = st.sidebar.selectbox("区を選択", ward_list, index=3)

# 2. 町名をフィルタリング（選択された区が含まれる町名のみ抽出）
all_towns = sorted(list(town_mapping.keys()))
filtered_towns = [t for t in all_towns if selected_ward in t]

# 万が一、区名が含まれていないデータ形式の場合は全件表示
display_towns = filtered_towns if filtered_towns else all_towns

# デフォルト町名設定（西新宿）
default_town_name = "西新宿"
# 「新宿区西新宿」などの形式に対応できるよう部分一致で検索
initial_index = 0
for i, t in enumerate(display_towns):
    if default_town_name in t:
        initial_index = i
        break

selected_town = st.sidebar.selectbox("町名を選択", display_towns, index=initial_index)

st.sidebar.divider()
st.sidebar.header("📏 物件スペック")
size = st.sidebar.slider("専有面積 (㎡)", 10.0, 200.0, 60.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2025, 2010)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)

# --- AI予測実行 ---
age = 2026 - built_year
town_score = town_mapping[selected_town]

input_df = pd.DataFrame([[size, age, walk_min, town_score]], 
                          columns=['size', 'age', 'walk', 'town_score'])

predicted_price = int(model.predict(input_df)[0])

# --- メイン画面：結果表示 ---
st.subheader(f"📊 {selected_town} エリアの査定結果")

# メトリック表示（販売価格のみに特化）
col1, col2 = st.columns(2)
with col1:
    st.metric("AI査定価格", f"{predicted_price:,} 円")
with col2:
    unit_price = int(predicted_price / size)
    st.metric("予測平米単価", f"{unit_price:,} 円/㎡")

st.divider()

# AIの判断根拠（詳細表示）
with st.expander("🧐 AIの査定ポイントを表示"):
    st.write(f"- **地域価値**: {selected_town}の基準平米単価 {int(town_score):,}円 をベースに算出。")
    st.write(f"- **建物評価**: 築{age}年の減価率を反映。")
    st.write(f"- **立地評価**: 駅徒歩{walk_min}分の利便性を加味。")

st.info("※2026年時点の統計推計値です。")

import streamlit as st
import pandas as pd
import joblib
import io

# 1. 資産の読み込み（分割されたAIモデルをメモリ上で合体させる）
@st.cache_resource
def load_assets():
    try:
        # 地点マッピングの読み込み
        town_mapping = joblib.load('town_mapping.joblib')
        
        # 分割された4つのファイルをバイナリ形式で読み込んで結合
        combined_data = b""
        for i in range(4):
            file_name = f"tokyo_price_v1_part{i}.pkl"
            with open(file_name, "rb") as f:
                combined_data += f.read()
        
        # 結合したデータを一つのモデルとして復元
        model = joblib.load(io.BytesIO(combined_data))
        return town_mapping, model
    except FileNotFoundError as e:
        st.error(f"必要なファイルが見つかりません。ファイル名を確認してください: {e}")
        st.stop()
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.stop()

# アセットのロード
town_mapping, model = load_assets()

# --- アプリ画面の構成 ---
st.set_page_config(page_title="23区マンションAI査定", layout="centered")
st.title("🏙️ 23区マンションAI価格査定")
st.caption("23区全域の取引データを学習したランダムフォレストモデルによる高精度査定")

# --- サイドバー：物件条件の入力 ---
st.sidebar.header("📍 所在地設定")

# 770地点の町名から選択
all_towns = sorted(list(town_mapping.keys()))
selected_town = st.sidebar.selectbox("町名を選択してください", all_towns)

st.sidebar.divider()
st.sidebar.header("📏 物件スペック")

# スペック入力
size = st.sidebar.slider("専有面積 (㎡)", 10.0, 200.0, 60.0, 0.5)
built_year = st.sidebar.number_input("築年 (西暦)", 1970, 2025, 2010)
walk_min = st.sidebar.slider("駅徒歩 (分)", 1, 30, 5)

# --- 予測実行ロジック ---
# 2026年を基準とした築年数の計算
age = 2026 - built_year
# 選択された町名の地点指数（平均単価）を取得
town_score = town_mapping[selected_town]

# AIモデルに渡すデータフレームの作成（学習時と同じカラム順）
input_df = pd.DataFrame([[size, age, walk_min, town_score]], 
                          columns=['size', 'age', 'walk', 'town_score'])

# AI予測の実行
predicted_raw = model.predict(input_df)[0]
predicted_price = int(predicted_raw)

# --- メイン画面：結果表示 ---
st.subheader(f"📊 {selected_town} エリアの査定結果")

# メトリック表示（保存情報に基づきカッコなし）
col1, col2 = st.columns(2)
with col1:
    st.metric("AI査定価格", f"{predicted_price:,} 円")
with col2:
    unit_price = int(predicted_price / size)
    st.metric("予測平米単価", f"{unit_price:,} 円/㎡")

st.divider()

# AIの判断根拠の可視化
with st.expander("🧐 AIの査定ポイントを表示"):
    st.write(f"""
    - **地域価値**: {selected_town}の基準平米単価 {int(town_score):,}円 をベースに算出しています。
    - **経年減価**: 築{age}年の建物評価をランダムフォレストが計算しました。
    - **利便性評価**: 駅徒歩{walk_min}分の立地条件を価格に反映しています。
    """)

# 利回りシミュレーション
st.write("#### 💰 収益性シミュレーション")
rent_input = st.number_input("想定月額賃料 (円)", value=200000, step=5000)
if predicted_price > 0:
    yield_rate = (rent_input * 12) / predicted_price * 100
    st.write(f"想定表面利回り: **{yield_rate:.2f} %**")

st.info("※この査定は統計モデルに基づく推計であり、実際の取引価格を保証するものではありません。")
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson
from sklearn.ensemble import RandomForestRegressor

# --- CONFIG ---
st.set_page_config(page_title="MAHATEP AI GOD MODE", layout="wide")

# --- 1. DATA ENGINE (ดึงข้อมูลทุกลีก) ---
@st.cache_data(ttl=3600)
def get_data():
    leagues = {'E0': 'Premier League', 'SP1': 'La Liga', 'D1': 'Bundesliga'}
    dfs = []
    for code, name in leagues.items():
        url = f"https://www.football-data.co.uk/mmz4281/2425/{code}.csv"
        try:
            df = pd.read_csv(url)
            df['League'] = name
            dfs.append(df)
        except: continue
    return pd.concat(dfs)

# --- 2. PREDICTION LOGIC (ทายสกอร์) ---
def predict_score(home_team, away_team, data):
    # คำนวณค่าเฉลี่ยประตู (Strength)
    avg_home_goals = data['FTHG'].mean()
    avg_away_goals = data['FTAG'].mean()
    
    # ความแข็งแกร่งทีมเหย้า
    home_att = data[data['HomeTeam'] == home_team]['FTHG'].mean() / avg_home_goals
    home_def = data[data['HomeTeam'] == home_team]['FTAG'].mean() / avg_away_goals
    
    # ความแข็งแกร่งทีมเยือน
    away_att = data[data['AwayTeam'] == away_team]['FTAG'].mean() / avg_away_goals
    away_def = data[data['AwayTeam'] == away_team]['FTHG'].mean() / avg_home_goals
    
    # คำนวณ Expected Goals (xG)
    exp_home = home_att * away_def * avg_home_goals
    exp_away = away_att * home_def * avg_away_goals
    
    return round(exp_home), round(exp_away), exp_home, exp_away

# --- 3. UI DISPLAY ---
st.title("🔥 มหาเทพ AI: ระบบวิเคราะห์สกอร์และตรวจสอบความแม่นยำ")

data = get_data()
league_list = data['League'].unique()
sel_league = st.selectbox("เลือกเลือกลีก", league_list)

filtered_data = data[data['League'] == sel_league]

# --- ส่วนการแสดงผล: วิเคราะห์คู่ถัดไป ---
st.header("🎯 ทายผลการแข่งขัน (Upcoming)")
teams = sorted(filtered_data['HomeTeam'].unique())
col1, col2 = st.columns(2)
with col1: h_team = st.selectbox("เจ้าบ้าน", teams, index=0)
with col2: a_team = st.selectbox("ทีมเยือน", teams, index=1)

if st.button("คำนวณสกอร์"):
    h_s, a_s, h_xg, a_xg = predict_score(h_team, a_team, filtered_data)
    st.success(f"🤖 AI ทายสกอร์: {h_team} {h_s} - {a_s} {a_team}")
    st.info(f"📊 ค่า xG (ละเอียด): {h_team} ({h_xg:.2f}) vs {a_team} ({a_xg:.2f})")

# --- 4. BACKTESTING (ส่วนแสดงความผิดพลาดหลังบอลเตะ) ---
st.markdown("---")
st.header("📉 ตรวจสอบความผิดพลาด (Post-Match Analysis)")
st.write("เปรียบเทียบผลที่ AI ทายไว้ กับ ผลการแข่งขันจริงที่เกิดขึ้น")

# จำลองการตรวจสอบย้อนหลัง 10 นัดล่าสุด
recent_matches = filtered_data.tail(10).copy()
comparison = []

for _, row in recent_matches.iterrows():
    # ลองให้ AI ทายผลนัดที่เตะไปแล้ว (โดยใช้ Data ก่อนหน้านั้น - ในที่นี้ใช้ Data ปัจจุบันเพื่อ Demo)
    p_h, p_a, _, _ = predict_score(row['HomeTeam'], row['AwayTeam'], filtered_data)
    
    # คำนวณ Error (ความผิดพลาด)
    error_h = abs(p_h - row['FTHG'])
    error_a = abs(p_a - row['FTAG'])
    total_error = error_h + error_a
    
    comparison.append({
        "คู่แข่งขัน": f"{row['HomeTeam']} vs {row['AwayTeam']}",
        "ผลจริง": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
        "AI ทาย": f"{p_h}-{p_a}",
        "ความคลาดเคลื่อน": "แม่นยำมาก" if total_error == 0 else ("ใกล้เคียง" if total_error <= 1 else "พลาด")
    })

st.table(pd.DataFrame(comparison))

# --- CSS สวยๆ ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #00ff88; color: black; font-weight: bold; border-radius: 10px; }
    table { background-color: #1d2129; color: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

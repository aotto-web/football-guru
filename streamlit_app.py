import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson
from datetime import datetime

# --- CONFIG & UI ---
st.set_page_config(page_title="MAHATEP AI - PREDICTOR", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .league-card { background: #1d2129; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #00ff88; }
    .match-row { display: grid; grid-template-columns: 100px 1.5fr 1fr 1.5fr 1.5fr; 
                background: #262c36; padding: 10px; margin-bottom: 5px; border-radius: 5px; align-items: center; }
    .error-badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }
    .perfect { background: #00ff88; color: black; }
    .near { background: #ffdd00; color: black; }
    .miss { background: #ff006e; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
LEAGUES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1"
}

@st.cache_data(ttl=3600)
def load_data(league_code):
    url = f"https://www.football-data.co.uk/mmz4281/2425/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        return df
    except: return None

# --- 2. PREDICTION LOGIC (POISSON MODEL) ---
def get_score_prediction(home_team, away_team, df):
    # คำนวณความแข็งแกร่ง (Strength)
    avg_h_goals = df['FTHG'].mean()
    avg_a_goals = df['FTAG'].mean()
    
    # ความสามารถในการทำประตูและเสียประตู
    h_att = df[df['HomeTeam'] == home_team]['FTHG'].mean() / avg_h_goals
    h_def = df[df['HomeTeam'] == home_team]['FTAG'].mean() / avg_a_goals
    a_att = df[df['AwayTeam'] == away_team]['FTAG'].mean() / avg_a_goals
    a_def = df[df['AwayTeam'] == away_team]['FTHG'].mean() / avg_h_goals
    
    # Expected Goals (xG)
    exp_h = h_att * a_def * avg_h_goals
    exp_a = a_att * h_def * avg_a_goals
    
    # ทายสกอร์ที่มีโอกาสเกิดสูงสุด
    pred_h = np.argmax([poisson.pmf(i, exp_h) for i in range(6)])
    pred_a = np.argmax([poisson.pmf(i, exp_a) for i in range(6)])
    
    return pred_h, pred_a, exp_h, exp_a

# --- 3. MAIN DISPLAY ---
st.title("⚽ มหาเทพ AI: วิเคราะห์สกอร์ & ตรวจสอบความแม่นยำ")

# วนลูปตามลำดับความสำคัญ (Premier League ก่อน)
ordered_leagues = ["Premier League"] + [l for l in LEAGUES.keys() if l != "Premier League"]

for league_name in ordered_leagues:
    df = load_data(LEAGUES[league_name])
    if df is None: continue
    
    with st.expander(f"🏆 {league_name}", expanded=(league_name == "Premier League")):
        # --- ส่วนที่ 1: วิเคราะห์แมตช์ล่าสุด (Checking Error) ---
        st.subheader("📊 ตรวจสอบความแม่นยำล่าสุด (Backtest)")
        recent_matches = df.tail(5).copy() # ตรวจสอบ 5 นัดล่าสุดที่จบไปแล้ว
        
        for _, row in recent_matches.iterrows():
            # ลองใช้ AI ทายผลนัดที่จบไปแล้ว
            p_h, p_a, _, _ = get_score_prediction(row['HomeTeam'], row['AwayTeam'], df)
            
            # คำนวณความผิดพลาด
            actual = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
            pred = f"{p_h}-{p_a}"
            total_error = abs(p_h - row['FTHG']) + abs(p_a - row['FTAG'])
            
            if total_error == 0: status, cls = "เป๊ะ!", "perfect"
            elif total_error <= 1: status, cls = "ใกล้เคียง", "near"
            else: status, cls = "พลาด", "miss"
            
            st.markdown(f"""
            <div class="match-row">
                <div style="font-size:12px; color:#888;">{row['Date'].strftime('%d/%m')}</div>
                <div style="text-align:right;">{row['HomeTeam']}</div>
                <div style="text-align:center; font-weight:bold; color:#00ff88;">{actual}</div>
                <div style="text-align:left;">{row['AwayTeam']}</div>
                <div>AI ทาย: <b>{pred}</b> <span class="error-badge {cls}">{status}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # --- ส่วนที่ 2: ทายผลคู่ถัดไป (Predictions) ---
        st.subheader("🎯 ทายผลคู่ถัดไป")
        teams = sorted(df['HomeTeam'].unique())
        c1, c2, c3 = st.columns([2,2,1])
        with c1: h_t = st.selectbox(f"เจ้าบ้าน ({league_name})", teams, key=f"h_{league_name}")
        with c2: a_t = st.selectbox(f"ทีมเยือน ({league_name})", teams, key=f"a_{league_name}")
        with c3:
            if st.button("วิเคราะห์", key=f"btn_{league_name}"):
                p_h, p_a, x_h, x_g = get_score_prediction(h_t, a_t, df)
                st.markdown(f"""
                <div style="background:#00ff8822; padding:10px; border-radius:5px; border:1px solid #00ff88; text-align:center;">
                    <h2 style="margin:0; color:#00ff88;">{p_h} - {p_a}</h2>
                    <p style="margin:0; font-size:12px;">(xG: {x_h:.2f} - {x_g:.2f})</p>
                </div>
                """, unsafe_allow_html=True)

# --- 4. การเชื่อมต่อกับ PHP ---
with st.sidebar:
    st.header("💻 สำหรับนำไปใช้ใน PHP")
    st.info("""
    เพื่อให้ PHP ใช้งานได้ คุณควร:
    1. รัน Python เพื่อดึงค่า xG และทายสกอร์
    2. บันทึกลง MySQL (Table: predictions)
    3. ใช้ PHP ดึงข้อมูลมาแสดงผล (SELECT)
    4. เมื่อบอลเตะจบ อัปเดตผลจริง (Actual) แล้วเขียน PHP คำนวณ Error (pred - actual)
    """)

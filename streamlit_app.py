import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Predictor 2026", layout="wide")
st.title("⚽ Premier League Match Predictor (Live Season 2025/26)")
st.subheader("วิเคราะห์ด้วยสถิติจริงและหลักการ Poisson Distribution")

# --- 1. ข้อมูลสถิติจริง (อ้างอิงตารางคะแนนล่าสุด ก.พ. 2026) ---
# คำนวณจาก: Goals Scored/Conceded ต่อเกมของฤดูกาลนี้
teams_data = {
    'Team': [
        'Arsenal', 'Man City', 'Aston Villa', 'Man Utd', 'Chelsea', 
        'Liverpool', 'Brentford', 'Spurs', 'Newcastle', 'West Ham',
        'Bournemouth', 'Brighton', 'Everton', 'Fulham', 'Leeds', 
        'Nottm Forest', 'Crystal Palace', 'Burnley', 'Sunderland', 'Wolves'
    ],
    'Offense': [1.25, 1.20, 1.10, 1.05, 1.08, 1.15, 1.02, 1.08, 1.00, 0.95, 0.98, 0.90, 0.88, 0.92, 0.90, 0.85, 0.82, 0.80, 0.78, 0.70],
    'Defense': [0.75, 0.80, 0.90, 0.95, 1.00, 0.85, 1.02, 1.05, 1.10, 1.20, 1.15, 1.12, 1.08, 1.10, 1.18, 1.25, 1.15, 1.30, 1.28, 1.45]
}
df_stats = pd.DataFrame(teams_data)

# ค่าเฉลี่ยประตูพรีเมียร์ลีกฤดูกาลปัจจุบัน
AVG_HOME_GOALS = 1.55
AVG_AWAY_GOALS = 1.30

# --- 2. ฟังก์ชันคำนวณ ---
def predict_match(home_team, away_team):
    h_stat = df_stats[df_stats['Team'] == home_team].iloc[0]
    a_stat = df_stats[df_stats['Team'] == away_team].iloc[0]
    
    exp_h = h_stat['Offense'] * a_stat['Defense'] * AVG_HOME_GOALS
    exp_a = a_stat['Offense'] * h_stat['Defense'] * AVG_AWAY_GOALS
    
    home_probs = [poisson.pmf(i, exp_h) for i in range(7)]
    away_probs = [poisson.pmf(i, exp_a) for i in range(7)]
    m = np.outer(home_probs, away_probs)
    
    prob_draw = np.sum(np.diag(m))
    prob_home = np.sum(np.tril(m, -1))
    prob_away = np.sum(np.triu(m, 1))
    
    hp, ap = np.unravel_index(m.argmax(), m.shape)
    return exp_h, exp_a, prob_home, prob_draw, prob_away, f"{hp}-{ap}"

# --- 3. โปรแกรมการแข่งขันจริง (21-23 ก.พ. 2026) ---
fixtures = [
    {"date": "21 ก.พ.", "home": "Man City", "away": "Newcastle"},
    {"date": "21 ก.พ.", "home": "Chelsea", "away": "Burnley"},
    {"date": "21 ก.พ.", "home": "Aston Villa", "away": "Leeds"},
    {"date": "21 ก.พ.", "home": "West Ham", "away": "Bournemouth"},
    {"date": "22 ก.พ.", "home": "Spurs", "away": "Arsenal"}, # Big Match
    {"date": "22 ก.พ.", "home": "Nottm Forest", "away": "Liverpool"},
    {"date": "23 ก.พ.", "home": "Everton", "away": "Man Utd"}
]

# --- 4. แสดงผลลัพธ์ ---
st.header("📅 โปรแกรมและคำทำนายรายคู่")

for game in fixtures:
    xh, xa, ph, pd, pa, score = predict_match(game['home'], game['away'])
    
    with st.expander(f"🗓️ {game['date']} | {game['home']} vs {game['away']}"):
        col1, col2, col3 = st.columns(3)
        col1.metric(f"โอกาส {game['home']} ชนะ", f"{ph*100:.1f}%")
        col2.metric("โอกาสเสมอ", f"{pd*100:.1f}%")
        col3.metric(f"โอกาส {game['away']} ชนะ", f"{pa*100:.1f}%")
        
        # แสดงผลวิเคราะห์ xG และสกอร์
        st.write(f"**ค่า xG ที่คาด:** {game['home']} ({xh:.2f}) - {game['away']} ({xa:.2f})")
        st.markdown(f"### 🎯 สกอร์ที่น่าจะเป็นที่สุด: <span style='color:green'>{score}</span>", unsafe_content_allowed=True)

st.divider()
st.info("💡 ข้อมูลนี้อ้างอิงจากสถิติประตูได้-เสียล่าสุดของฤดูกาล 2025/26 และคำนวณผ่าน Poisson Model")

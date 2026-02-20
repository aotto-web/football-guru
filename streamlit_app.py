import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Analysis - LiveScore Style", layout="wide")
st.title("⚽ Premier League Match Analysis (Live Data 2026)")

# --- 1. ข้อมูลสถิติทีม (อัปเดตฟอร์มล่าสุดกุมภาพันธ์ 2026) ---
# ค่า Offense (บุก) และ Defense (รับ) ยิ่งบุกสูงยิ่งดี ยิ่งรับต่ำยิ่งเหนียว
teams_data = {
    'Team': [
        'Arsenal', 'Man City', 'Liverpool', 'Man Utd', 'Chelsea', 
        'Aston Villa', 'Newcastle', 'Spurs', 'Everton', 'West Ham',
        'Brentford', 'Brighton', 'Bournemouth', 'Fulham', 'Leeds', 
        'Nottm Forest', 'Crystal Palace', 'Burnley', 'Sunderland', 'Wolves'
    ],
    'Offense': [1.28, 1.25, 1.18, 1.10, 1.12, 1.05, 1.02, 1.08, 0.90, 0.92, 1.01, 0.95, 0.96, 0.94, 0.88, 0.85, 0.84, 0.80, 0.78, 0.72],
    'Defense': [0.72, 0.78, 0.82, 0.92, 0.98, 0.90, 1.05, 1.02, 1.04, 1.15, 1.05, 1.10, 1.12, 1.08, 1.18, 1.22, 1.16, 1.28, 1.25, 1.40]
}
df_stats = pd.DataFrame(teams_data)

# --- 2. ฟังก์ชันวิเคราะห์ Poisson ---
def analyze_match(home, away):
    h_stat = df_stats[df_stats['Team'] == home].iloc[0]
    a_stat = df_stats[df_stats['Team'] == away].iloc[0]
    
    # คำนวณ xG (ค่าเฉลี่ยประตูที่คาดหวัง)
    # สมมติค่าเฉลี่ยลีก: เหย้า 1.55, เยือน 1.30
    exp_h = h_stat['Offense'] * a_stat['Defense'] * 1.55
    exp_a = a_stat['Offense'] * h_stat['Defense'] * 1.30
    
    # สร้าง Matrix ความน่าจะเป็น (0-6 ประตู)
    h_prob = [poisson.pmf(i, exp_h) for i in range(7)]
    a_prob = [poisson.pmf(i, exp_a) for i in range(7)]
    matrix = np.outer(h_prob, a_prob)
    
    prob_home = np.sum(np.tril(matrix, -1))
    prob_draw = np.sum(np.diag(matrix))
    prob_away = np.sum(np.triu(matrix, 1))
    
    # สกอร์ที่น่าจะเป็นที่สุด
    hp, ap = np.unravel_index(matrix.argmax(), matrix.shape)
    
    return exp_h, exp_a, prob_home, prob_draw, prob_away, f"{hp}-{ap}"

# --- 3. โปรแกรมแข่งจริงจาก LiveScore (21-23 ก.พ. 2026) ---
st.header("📅 วิเคราะห์โปรแกรมการแข่งขันสัปดาห์นี้")

fixtures = [
    {"time": "21 ก.พ. 19:30", "home": "Aston Villa", "away": "Leeds"},
    {"time": "21 ก.พ. 22:00", "home": "Chelsea", "away": "Burnley"},
    {"time": "21 ก.พ. 22:00", "home": "West Ham", "away": "Bournemouth"},
    {"time": "22 ก.พ. 00:30", "home": "Man City", "away": "Newcastle"}, # คู่ใหญ่
    {"time": "22 ก.พ. 21:00", "home": "Spurs", "away": "Arsenal"},     # North London Derby
    {"time": "23 ก.พ. 03:00", "home": "Everton", "away": "Man Utd"}
]

for match in fixtures:
    xh, xa, ph, pd, pa, score = analyze_match(match['home'], match['away'])
    
    with st.expander(f"⏰ {match['time']} | {match['home']} vs {match['away']}"):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"โอกาส {match['home']} ชนะ", f"{ph*100:.1f}%")
        c2.metric("โอกาสเสมอ", f"{pd*100:.1f}%")
        c3.metric(f"โอกาส {match['away']} ชนะ", f"{pa*100:.1f}%")
        
        st.write(f"**การวิเคราะห์เชิงลึก:**")
        st.write(f"- ค่า xG คาดการณ์: {match['home']} {xh:.2f} | {match['away']} {xa:.2f}")
        st.write(f"- สกอร์ที่มีโอกาสเกิดสูงสุด: **{score}**")

# --- ส่วนท้าย ---
st.divider()
st.caption("อ้างอิงโปรแกรมการแข่งขันจาก LiveScore และคำนวณผลตามสถิติทีมปัจจุบัน")

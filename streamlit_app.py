import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- 1. การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Premier League Guru", layout="wide")
st.title("🏆 Premier League Predictor (Poisson Model)")

# --- 2. ข้อมูลจำลอง (ในงานจริงควรดึงจาก API หรือ CSV) ---
# ตัวอย่างค่า Strength ของทีม (ยิ่งสูงยิ่งดีสำหรับ Attack, ยิ่งต่ำยิ่งดีสำหรับ Defense)
teams_data = {
    'Team': ['Man City', 'Arsenal', 'Liverpool', 'Aston Villa', 'Spurs', 'Man Utd', 'Newcastle', 'Chelsea'],
    'Offense': [1.25, 1.15, 1.20, 1.05, 1.10, 0.95, 1.00, 1.05], # พลังบุก
    'Defense': [0.80, 0.75, 0.85, 1.00, 1.10, 1.05, 1.15, 1.20]  # พลังรับ (น้อยยิ่งเหนียว)
}
df_stats = pd.DataFrame(teams_data)

# ค่าเฉลี่ยประตูของลีก (Premier League ปกติจะอยู่ที่ประมาณนี้)
AVG_HOME_GOALS = 1.53
AVG_AWAY_GOALS = 1.32

# --- 3. ฟังก์ชันคำนวณความน่าจะเป็น ---
def predict_match(home_team, away_team):
    h_stat = df_stats[df_stats['Team'] == home_team].iloc[0]
    a_stat = df_stats[df_stats['Team'] == away_team].iloc[0]
    
    # สูตร xG: (ทีมเหย้าบุก * ทีมเยือนรับ * ค่าเฉลี่ยลีก)
    exp_h = h_stat['Offense'] * a_stat['Defense'] * AVG_HOME_GOALS
    exp_a = a_stat['Offense'] * h_stat['Defense'] * AVG_AWAY_GOALS
    
    # คำนวณโอกาสชนะ/เสมอ/แพ้ (Matrix 0-6 ประตู)
    home_probs = [poisson.pmf(i, exp_h) for i in range(7)]
    away_probs = [poisson.pmf(i, exp_a) for i in range(7)]
    
    m = np.outer(home_probs, away_probs)
    
    prob_draw = np.sum(np.diag(m))
    prob_home = np.sum(np.tril(m, -1))
    prob_away = np.sum(np.triu(m, 1))
    
    # สกอร์ที่น่าจะเป็นที่สุด (Correct Score)
    hp, ap = np.unravel_index(m.argmax(), m.shape)
    
    return exp_h, exp_a, prob_home, prob_draw, prob_away, f"{hp}-{ap}"

# --- 4. ส่วนแสดงผลการคำนวณรายคู่ ---
st.header("📅 วิเคราะห์โปรแกรมการแข่งขัน")

# รายชื่อคู่แข่งขันที่กำลังจะมาถึง (ตัวอย่าง)
fixtures = [
    ("Man City", "Arsenal"),
    ("Liverpool", "Chelsea"),
    ("Spurs", "Man Utd"),
    ("Newcastle", "Aston Villa")
]

for home, away in fixtures:
    xh, xa, ph, pd, pa, score = predict_match(home, away)
    
    with st.expander(f"🏟️ {home} vs {away} (คลิกเพื่อดูรายละเอียด)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"โอกาส {home} ชนะ", f"{ph*100:.1f}%")
        with col2:
            st.metric("โอกาสเสมอ", f"{pd*100:.1f}%")
        with col3:
            st.metric(f"โอกาส {away} ชนะ", f"{pa*100:.1f}%")
            
        st.write(f"**คาดการณ์ประตู (xG):** {home} {xh:.2f} - {xa:.2f} {away}")
        st.write(f"**สกอร์ที่น่าจะเป็นที่สุด:** :green[{score}]")

# --- 5. ตารางค่าพลังทีม ---
st.divider()
st.subheader("📊 ตารางค่าพลังทีม (Team Strength Stats)")
st.dataframe(df_stats, use_container_width=True)

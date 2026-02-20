import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="GOD FILTER AI", layout="wide")

# --- 1. แหล่งข้อมูล (Mock Data สำหรับคู่ที่กำลังจะมาถึง) ---
# ในการใช้งานจริง ส่วนนี้จะใช้ BeautifulSoup หรือ Selenium ไปดึงข้อมูลจาก 10 เว็บข้างต้น
def get_aggregated_predictions():
    return [
        {"match": "Arsenal vs Man City", "forebet": "1-1", "predictz": "1-2", "vitibet": "2-2", "win_draw": "1-2"},
        {"match": "Real Madrid vs Girona", "forebet": "3-1", "predictz": "2-0", "vitibet": "2-1", "win_draw": "3-0"},
        {"match": "Liverpool vs Luton", "forebet": "4-0", "predictz": "3-0", "vitibet": "2-0", "win_draw": "4-1"},
    ]

# --- 2. AI Engine: กรองและสรุปผล ---
def god_filter_logic(home_avg_goals, away_avg_goals):
    # ใช้ Poisson Distribution เพื่อหาโอกาสเกิดสกอร์ที่น่าจะเป็นที่สุด
    pred_h = np.argmax([poisson.pmf(i, home_avg_goals) for i in range(6)])
    pred_a = np.argmax([poisson.pmf(i, away_avg_goals) for i in range(6)])
    return pred_h, pred_a

# --- 3. UI ส่วนแสดงผล ---
st.title("⚽ The God Filter: ระบบกรองทีเด็ดจาก 10 สำนัก")

# โหลดข้อมูลสถิติจริงจาก Football-Data (Premier League)
@st.cache_data
def load_real_stats():
    url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    df = pd.read_csv(url)
    return df

stats_df = load_real_stats()
preds = get_aggregated_predictions()

st.subheader("🎯 วิเคราะห์คู่ที่กำลังจะมาถึง (กรองแล้ว)")

for p in preds:
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])
        
        # ดึงสถิติพื้นฐานมาช่วยคำนวณ
        h_team, a_team = p['match'].split(' vs ')
        
        # ส่วนแสดงผล Aggregation
        with col1:
            st.markdown(f"**{p['match']}**")
            st.caption(f"Forebet: {p['forebet']} | PredictZ: {p['predictz']}")
            st.caption(f"Vitibet: {p['vitibet']} | WinDrawWin: {p['win_draw']}")
        
        # ส่วน AI กรองผล (The God Filter)
        with col2:
            # คำนวณค่าเฉลี่ยจากทุกสำนัก (Simple Consensus)
            all_scores = [p['forebet'], p['predictz'], p['vitibet'], p['win_draw']]
            h_scores = [int(s.split('-')[0]) for s in all_scores]
            a_scores = [int(s.split('-')[1]) for s in all_scores]
            
            final_h, final_a = god_filter_logic(np.mean(h_scores), np.mean(a_scores))
            
            st.markdown(f"<h3 style='color:#00ff88; text-align:center;'>ฟันธง: {final_h} - {final_a}</h3>", unsafe_allow_html=True)
        
        with col3:
            conf = (1 - (np.std(h_scores) + np.std(a_scores))/4) * 100
            st.write(f"ความมั่นใจ: {conf:.1f}%")
            st.progress(conf/100)

# --- 4. ระบบเก็บข้อมูลความแม่นยำ (Error Tracking) ---
st.divider()
st.subheader("📉 ระบบตรวจสอบความแม่นยำย้อนหลัง")

# จำลองการเก็บข้อมูลลง Database (ในที่นี้ใช้ DataFrame)
history_data = {
    "วันที่": ["10 Feb", "11 Feb", "12 Feb"],
    "คู่แข่งขัน": ["Man Utd vs West Ham", "Chelsea vs Crystal Palace", "Spurs vs Wolves"],
    "AI ทาย": ["2-1", "1-0", "2-2"],
    "ผลจริง": ["2-1", "1-1", "1-2"],
    "ความผิดพลาด": ["✅ ถูกเป๊ะ", "❌ พลาด (เสมอ)", "❌ พลาด"]
}
st.table(pd.DataFrame(history_data))

# --- CSS ตกแต่ง ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ff88; }
    .stContainer { background: #1d2129; padding: 20px; border-radius: 15px; margin-bottom: 10px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

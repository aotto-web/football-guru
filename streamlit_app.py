import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Auto-Predictor", layout="wide")
st.title("⚽ Premier League Real-Time Predictor")
st.write("ดึงข้อมูลตารางแข่งและคำนวณอัตโนมัติจากสถิติล่าสุด")

# --- 1. ฟังก์ชันดึงข้อมูลตารางคะแนนสด (เพื่อหาค่าพลังทีม) ---
@st.cache_data(ttl=3600) # เก็บข้อมูลไว้ 1 ชม. จะได้ไม่ดึงบ่อยจนโดนบล็อก
def get_live_stats():
    try:
        # ดึงตารางคะแนนจาก FBRef (แหล่งข้อมูลสถิติฟุตบอลที่เสถียรที่สุดสำหรับ Python)
        url = "https://www.worldfootball.net/premier_league_2025_2026/table/"
        tables = pd.read_html(url)
        df = tables[0]
        
        # จัดรูปแบบข้อมูล
        df = df[['#', 'Team', 'M.', 'Goals', 'Pts']]
        # แยกประตูได้-เสียออกจากกัน (เช่น 45:20)
        df[['Scored', 'Conceded']] = df['Goals'].str.split(':', expand=True).astype(int)
        
        # คำนวณค่าเฉลี่ยของลีก
        avg_scored = df['Scored'].mean()
        avg_conceded = df['Conceded'].mean()
        
        # คำนวณ Offense และ Defense Strength ของแต่ละทีม
        df['Offense'] = df['Scored'] / avg_scored
        df['Defense'] = df['Conceded'] / avg_conceded
        
        return df, avg_scored / 2, avg_conceded / 2 # ส่งคืนค่าเฉลี่ยต่อทีม
    except Exception as e:
        st.error(f"ไม่สามารถดึงข้อมูลสถิติได้: {e}")
        return None, 1.5, 1.3

# --- 2. ฟังก์ชันดึงโปรแกรมการแข่งขัน (Fixtures) ---
@st.cache_data(ttl=3600)
def get_fixtures():
    try:
        # ดึงโปรแกรมการแข่งขัน
        url = "https://www.worldfootball.net/schedule/eng-premier-league-2025-2026-spieltag/25/" # Spieltag คือนัดที่
        tables = pd.read_html(url)
        fixtures_df = tables[1] # ตารางโปรแกรมมักอยู่ใน index 1 หรือ 2
        return fixtures_df
    except:
        return None

# --- 3. ฟังก์ชันคำนวณ Poisson ---
def predict_match(home_team, away_team, stats_df, avg_h, avg_a):
    try:
        h_stat = stats_df[stats_df['Team'].str.contains(home_team)].iloc[0]
        a_stat = stats_df[stats_df['Team'].str.contains(away_team)].iloc[0]
        
        exp_h = h_stat['Offense'] * a_stat['Defense'] * avg_h
        exp_a = a_stat['Offense'] * h_stat['Defense'] * avg_a
        
        h_prob = [poisson.pmf(i, exp_h) for i in range(7)]
        a_prob = [poisson.pmf(i, exp_a) for i in range(7)]
        m = np.outer(h_prob, a_prob)
        
        ph = np.sum(np.tril(m, -1))
        pd = np.sum(np.diag(m))
        pa = np.sum(np.triu(m, 1))
        hp, ap = np.unravel_index(m.argmax(), m.shape)
        
        return exp_h, exp_a, ph, pd, pa, f"{hp}-{ap}"
    except:
        return 0, 0, 0, 0, 0, "N/A"

# --- ส่วนหลักของโปรแกรม ---
df_stats, avg_h, avg_a = get_live_stats()
fixtures = get_fixtures()

if df_stats is not None:
    st.sidebar.header("📊 อันดับตารางคะแนนปัจจุบัน")
    st.sidebar.dataframe(df_stats[['Team', 'Pts', 'Offense', 'Defense']], hide_index=True)

    st.header("📅 โปรแกรมการแข่งขันนัดถัดไป")
    
    # วนลูปแสดงผลทุกคู่ที่ดึงมาได้
    for index, row in fixtures.iterrows():
        # ตรวจสอบว่าบรรทัดนั้นเป็นข้อมูลคู่แข่งจริงหรือไม่ (ข้อมูลเว็บมักมีบรรทัดว่าง)
        if isinstance(row[2], str) and ' - ' not in row[2]: 
            home = row[2]
            away = row[4]
            
            xh, xa, ph, pd, pa, score = predict_match(home, away, df_stats, avg_h, avg_a)
            
            with st.expander(f"🏟️ {home} vs {away}"):
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{home} ชนะ", f"{ph*100:.1f}%")
                c2.metric("เสมอ", f"{pd*100:.1f}%")
                c3.metric(f"{away} ชนะ", f"{pa*100:.1f}%")
                st.write(f"สกอร์ที่คาด: **{score}** (xG: {xh:.2f} - {xa:.2f})")

else:
    st.warning("กำลังรอข้อมูลจากเซิร์ฟเวอร์... กรุณารีเฟรชหน้าจอ")

st.info("💡 ระบบนี้ดึงข้อมูลจาก worldfootball.net อัตโนมัติ ทุกครั้งที่มีการแข่งขันนัดใหม่ สถิติจะถูกคำนวณใหม่ทันที")

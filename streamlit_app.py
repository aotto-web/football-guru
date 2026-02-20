import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Premier League Guru 2026", layout="wide", page_icon="⚽")

# --- ฟังก์ชันดึงข้อมูลแบบหลบการโดนบล็อก (ปลอมเป็น Browser) ---
def fetch_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"การดึงข้อมูลผิดพลาด: {e}")
        return None

# --- 1. ดึงตารางคะแนนสดเพื่อคำนวณค่าพลังทีม ---
@st.cache_data(ttl=3600)
def get_live_stats():
    # ใช้ worldfootball.net สำหรับตารางคะแนน
    url = "https://www.worldfootball.net/premier_league_2025_2026/table/"
    html = fetch_data(url)
    if html:
        try:
            tables = pd.read_html(html)
            df = tables[0]
            # กรองคอลัมน์สำคัญ: ทีม, แข่ง, ประตู (ได้:เสีย), แต้ม
            df = df[['Team', 'M.', 'Goals', 'Pts']]
            # แยกประตูได้/เสียออกจากกัน เช่น "40:20" -> 40 และ 20
            df[['Scored', 'Conceded']] = df['Goals'].str.split(':', expand=True).astype(int)
            
            # คำนวณค่าเฉลี่ยลีก
            avg_scored = df['Scored'].mean()
            avg_conceded = df['Conceded'].mean()
            
            # คำนวณความแข็งแกร่ง (Strength)
            df['Offense'] = df['Scored'] / avg_scored
            df['Defense'] = df['Conceded'] / avg_conceded
            
            return df, avg_scored / 20, avg_conceded / 20 # ค่าเฉลี่ยต่อเกม
        except:
            return None, 1.5, 1.3
    return None, 1.5, 1.3

# --- 2. ดึงโปรแกรมการแข่งขันนัดถัดไป ---
@st.cache_data(ttl=3600)
def get_fixtures():
    # ดึงตารางนัดล่าสุด/ถัดไป
    url = "https://www.worldfootball.net/schedule/eng-premier-league-2025-2026-spieltag/25/"
    html = fetch_data(url)
    if html:
        try:
            tables = pd.read_html(html)
            # ปกติโปรแกรมแข่งจะอยู่ใน table index 1
            return tables[1]
        except:
            return None
    return None

# --- 3. ฟังก์ชันคำนวณทำนายผล ---
def predict_match(home, away, stats_df, avg_h, avg_a):
    try:
        # ค้นหาค่าพลังจากชื่อทีม
        h_stat = stats_df[stats_df['Team'].str.contains(home, case=False)].iloc[0]
        a_stat = stats_df[stats_df['Team'].str.contains(away, case=False)].iloc[0]
        
        # สูตร xG: บุกเหย้า * รับเยือน * ค่าเฉลี่ยเหย้า
        exp_h = h_stat['Offense'] * a_stat['Defense'] * avg_h
        exp_a = a_stat['Offense'] * h_stat['Defense'] * avg_a
        
        # คำนวณ Poisson
        h_prob = [poisson.pmf(i, exp_h) for i in range(7)]
        a_prob = [poisson.pmf(i, exp_a) for i in range(7)]
        matrix = np.outer(h_prob, a_prob)
        
        prob_h = np.sum(np.tril(matrix, -1))
        prob_d = np.sum(np.diag(matrix))
        prob_a = np.sum(np.triu(matrix, 1))
        hp, ap = np.unravel_index(matrix.argmax(), matrix.shape)
        
        return exp_h, exp_a, prob_h, prob_d, prob_a, f"{hp}-{ap}"
    except:
        return 0,0,0,0,0,"N/A"

# --- ส่วนแสดงผลบน Streamlit ---
st.title("⚽ Premier League Auto-Predictor 2026")
st.markdown("ระบบวิเคราะห์ผลการแข่งขันอัตโนมัติจากสถิติจริงและหลักการ **Poisson Distribution**")

df_stats, avg_h, avg_a = get_live_stats()
df_fixtures = get_fixtures()

if df_stats is not None:
    # Sidebar แสดงอันดับ
    st.sidebar.header("📊 Live Table Strength")
    st.sidebar.dataframe(df_stats[['Team', 'Offense', 'Defense']].sort_values('Offense', ascending=False), hide_index=True)

    # หน้าหลักแสดงโปรแกรมแข่ง
    st.header("📅 วิเคราะห์โปรแกรมสัปดาห์นี้")
    
    if df_fixtures is not None:
        # วนลูปเฉพาะแถวที่มีคู่แข่งขันจริง
        for _, row in df_fixtures.iterrows():
            if isinstance(row[2], str) and ' - ' not in row[2]:
                home_team = row[2]
                away_team = row[4]
                match_time = row[0]
                
                xh, xa, ph, pd, pa, score = predict_match(home_team, away_team, df_stats, avg_h, avg_a)
                
                with st.expander(f"🏟️ {match_time} | {home_team} vs {away_team}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{home_team} ชนะ", f"{ph*100:.1f}%")
                    col2.metric("เสมอ", f"{pd*100:.1f}%")
                    col3.metric(f"{away_team} ชนะ", f"{pa*100:.1f}%")
                    
                    st.write(f"**การวิเคราะห์:** สกอร์คาดการณ์ **{score}** | ค่า xG: {xh:.2f} - {xa:.2f}")
    else:
        st.warning("ไม่สามารถโหลดโปรแกรมการแข่งขันได้ในขณะนี้")
else:
    st.error("ไม่สามารถโหลดสถิติลีกได้ กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")

st.divider()
st.caption("Data Source: worldfootball.net | วิเคราะห์โดยหลักการทางสถิติ Poisson Distribution")

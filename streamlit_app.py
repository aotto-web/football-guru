import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ (เน้นดูง่ายบนมือถือ) ---
st.set_page_config(page_title="PL GURU", layout="centered", page_icon="⚽")

# --- ข้อมูล API (Key ของคุณ) ---
API_KEY = "2ab1eb65a8b94e8ea240487d86d1e6a5"
BASE_URL = "https://api.football-data.org/v4"

def call_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=3600)
def get_all_data():
    s_data = call_api("competitions/PL/standings")
    f_data = call_api("competitions/PL/matches?status=SCHEDULED")
    
    if s_data and 'standings' in s_data:
        table = s_data['standings'][0]['table']
        df = pd.DataFrame([{
            'N': t['team']['shortName'],
            'P': t['playedGames'],
            'GF': t['goalsFor'],
            'GA': t['goalsAgainst']
        } for t in table])
        
        df['P'] = df['P'].replace(0, 1)
        avg_g = df['GF'].sum() / df['P'].sum()
        
        # คำนวณค่าพลัง (Strength)
        df['Att'] = (df['GF'] / df['P']) / (avg_g if avg_g > 0 else 1)
        df['Def'] = (df['GA'] / df['P']) / (avg_g if avg_g > 0 else 1)
        
        fixtures = f_data.get('matches', []) if f_data else []
        return df, avg_g, fixtures
    return None, 1.5, []

def predict_score(h, a, df, avg_l):
    try:
        hs = df[df['N'] == h].iloc[0]
        as_ = df[df['N'] == a].iloc[0]
        
        ex_h = hs['Att'] * as_['Def'] * avg_l
        ex_a = as_['Att'] * hs['Def'] * avg_l
        
        # Poisson Calculation
        h_p = [poisson.pmf(i, ex_h) for i in range(7)]
        a_p = [poisson.pmf(i, ex_a) for i in range(7)]
        matrix = np.outer(h_p, a_p)
        
        p_h, p_d, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
        idx = matrix.argmax()
        return f"{idx // 7} - {idx % 7}", p_h, p_d, p_a, ex_h, ex_a
    except:
        return "N/A", 0, 0, 0, 0, 0

# --- การแสดงผล (Native Streamlit เท่านั้นเพื่อความปลอดภัย) ---
st.title("⚽ PREMIER GURU")
st.write("วิเคราะห์ผลบอลพรีเมียร์ลีกอัตโนมัติ (Poisson Model)")

stats, avg_g, fixtures = get_all_data()

if stats is not None:
    if not fixtures:
        st.info("📅 ไม่มีโปรแกรมการแข่งขันในเร็วๆ นี้")
    else:
        st.subheader(f"🏟️ วิเคราะห์ {len(fixtures)} คู่ถัดไป")
        
        for m in fixtures:
            home = m['homeTeam']['shortName']
            away = m['awayTeam']['shortName']
            score, ph, pd, pa, xh, xa = predict_score(home, away, stats, avg_g)
            
            # ใช้ st.container เพื่อสร้าง "Card" ที่ดู "ล่ำ"
            with st.container(border=True):
                # ส่วนหัว: ชื่อทีม
                col_h, col_vs, col_a = st.columns([4, 1, 4])
                col_h.markdown(f"### **{home}**")
                col_vs.markdown("### VS")
                col_a.markdown(f"### **{away}**")
                
                # ส่วนกลาง: สกอร์คาดการณ์ (ใช้สัญลักษณ์เด่นๆ)
                st.write("---")
                st.markdown(f"#### 🎯 สกอร์ที่คาด: **{score}**")
                
                # ส่วนท้าย: เปอร์เซ็นต์ความน่าจะเป็น
                c1, c2, c3 = st.columns(3)
                c1.metric("🏠 ชนะ", f"{ph*100:.0f}%")
                c2.metric("🤝 เสมอ", f"{pd*100:.0f}%")
                c3.metric("🚀 ชนะ", f"{pa*100:.0f}%")
                
                # ข้อมูลเสริมแบบ Expander เพื่อไม่ให้รกมือถือ
                with st.expander("ดูสถิติเชิงลึก (xG)"):
                    st.write(f"ค่าประตูที่คาดหวัง (xG): {home} ({xh:.2f}) - {away} ({xa:.2f})")
                    st.write(f"วันที่แข่งขัน: {m['utcDate'][:10]}")

    # Sidebar สำหรับข้อมูลลีก
    with st.sidebar:
        st.header("📊 ตารางค่าพลังทีม")
        st.dataframe(stats[['N', 'Att', 'Def']].sort_values('Att', ascending=False), hide_index=True)

else:
    st.error("⚠️ ไม่สามารถโหลดข้อมูลได้ กรุณาตรวจสอบ API Key หรือรีเฟรชหน้าจอ")

st.divider()
st.caption("Data Source: football-data.org | AI Analysis by Poisson Distribution")

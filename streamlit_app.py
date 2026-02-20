import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL GURU", layout="centered", page_icon="⚽")

# --- ข้อมูล API ---
API_KEY = "2ab1eb65a8b94e8ea240487d86d1e6a5"
BASE_URL = "https://api.football-data.org/v4"

def call_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def get_all_data():
    s_data = call_api("competitions/PL/standings")
    f_data = call_api("competitions/PL/matches?status=SCHEDULED")
    
    if s_data and 'standings' in s_data:
        table = s_data['standings'][0]['table']
        df = pd.DataFrame([{
            'Name': t['team']['shortName'],
            'P': t['playedGames'],
            'GF': t['goalsFor'],
            'GA': t['goalsAgainst']
        } for t in table])
        
        df['P'] = df['P'].replace(0, 1)
        avg_gf = df['GF'].sum() / df['P'].sum()
        df['Att'] = (df['GF'] / df['P']) / (avg_gf if avg_gf > 0 else 1)
        df['Def'] = (df['GA'] / df['P']) / (avg_gf if avg_gf > 0 else 1)
        
        matches = f_data.get('matches', []) if f_data else []
        return df, avg_gf, matches
    return None, 1.5, []

def predict_match(h_name, a_name, df, avg_league):
    try:
        h_stat = df[df['Name'] == h_name].iloc[0]
        a_stat = df[df['Name'] == a_name].iloc[0]
        
        ex_h = h_stat['Att'] * a_stat['Def'] * avg_league
        ex_a = a_stat['Att'] * h_stat['Def'] * avg_league
        
        h_probs = [poisson.pmf(i, ex_h) for i in range(7)]
        a_probs = [poisson.pmf(i, ex_a) for i in range(7)]
        matrix = np.outer(h_probs, a_probs)
        
        p_h, p_d, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
        score_idx = matrix.argmax()
        return ex_h, ex_a, p_h, p_d, p_a, f"{score_idx // 7}-{score_idx % 7}"
    except:
        return 0, 0, 0, 0, 0, "N/A"

# --- MAIN APP ---
st.title("⚽ PREMIER GURU")
st.write("วิเคราะห์ผลบอลพรีเมียร์ลีกอัตโนมัติ")

stats, avg_g, fixtures = get_all_data()

if stats is not None:
    if not fixtures:
        st.info("ไม่มีโปรแกรมการแข่งขันเร็วๆ นี้")
    else:
        st.subheader(f"📅 วิเคราะห์ {len(fixtures)} คู่ถัดไป")
        
        for m in fixtures:
            h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
            xh, xa, ph, pd, pa, score = predict_match(h, a, stats, avg_g)
            
            # ใช้ st.container แทน CSS เพื่อความปลอดภัย
            with st.container(border=True):
                st.markdown(f"### **{h} vs {a}**")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🏠 เหย้า", f"{ph*100:.0f}%")
                col2.metric("🤝 เสมอ", f"{pd*100:.0f}%")
                col3.metric("🚀 เยือน", f"{pa*100:.0f}%")
                
                # แสดงสกอร์แบบเด่นๆ ด้วย st.success
                st.success(f"🎯 **สกอร์ที่คาด: {score}**")
                
                st.caption(f"วันที่เตะ: {m['utcDate'][:10]} | xG: {xh:.1f} - {xa:.1f}")
else:
    st.error("ไม่สามารถเชื่อมต่อข้อมูลได้ กรุณาลองใหม่อีกครั้ง")

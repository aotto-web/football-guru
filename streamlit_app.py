import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ (เน้น Mobile กะทัดรัดที่สุด) ---
st.set_page_config(page_title="PL GURU PRO", layout="centered", page_icon="⚽")

# --- ข้อมูล API ---
API_KEY = "2ab1eb65a8b94e8ea240487d86d1e6a5"
BASE_URL = "https://api.football-data.org/v4"

def call_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, timeout=10)
        return response.json() if response.status_code == 200 else None
    except: return None

@st.cache_data(ttl=3600)
def get_all_data():
    s_data = call_api("competitions/PL/standings")
    f_data = call_api("competitions/PL/matches?status=SCHEDULED")
    if s_data and 'standings' in s_data:
        table = s_data['standings'][0]['table']
        df = pd.DataFrame([{'N': t['team']['shortName'], 'P': t['playedGames'], 'GF': t['goalsFor'], 'GA': t['goalsAgainst']} for t in table])
        df['P'] = df['P'].replace(0, 1)
        avg_g = df['GF'].sum() / df['P'].sum()
        df['Att'] = (df['GF'] / df['P']) / (avg_g if avg_g > 0 else 1)
        df['Def'] = (df['GA'] / df['P']) / (avg_g if avg_g > 0 else 1)
        return df, avg_g, f_data.get('matches', []) if f_data else []
    return None, 1.5, []

def predict(h, a, df, avg_l):
    try:
        hs, as_ = df[df['N'] == h].iloc[0], df[df['N'] == a].iloc[0]
        ex_h, ex_a = hs['Att'] * as_['Def'] * avg_l, as_['Att'] * hs['Def'] * avg_l
        h_p, a_p = [poisson.pmf(i, ex_h) for i in range(6)], [poisson.pmf(i, ex_a) for i in range(6)]
        matrix = np.outer(h_p, a_p)
        ph, pd, pa = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
        return f"{matrix.argmax()//6}-{matrix.argmax()%6}", ph, pd, pa
    except: return "N/A", 0, 0, 0

# --- การแสดงผลแบบ "1 คู่ 1 บรรทัด" ---
st.title("🏆 PREMIER GURU PRO")
st.write("` คู่แข่งขัน | สกอร์ | ชนะ-เสมอ-แพ้ % `")

stats, avg_g, fixtures = get_all_data()

if stats is not None and fixtures:
    for m in fixtures:
        h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
        score, ph, pd, pa = predict(h, a, stats, avg_g)
        
        # คำนวณสีของแถบความมั่นใจ
        # ถ้าโอกาสชนะฝั่งไหนเกิน 50% จะเป็นสีเขียว
        h_color = "🟢" if ph > 0.5 else "⚪"
        a_color = "🔴" if pa > 0.5 else "⚪"
        
        with st.container():
            # บรรทัดเดียวจบ: [ทีม] vs [ทีม] | [สกอร์] | [%]
            col_match, col_score, col_prob = st.columns([5, 2, 4])
            
            with col_match:
                st.markdown(f"**{h}** - **{a}**")
            
            with col_score:
                st.info(f"**{score}**")
                
            with col_prob:
                # แสดงผลแบบย่อ W-D-L
                st.write(f"{ph*100:.0f}%-{pd*100:.0f}%-{pa*100:.0f}%")
            
            st.divider() # เส้นคั่นบางๆ ให้ดูเป็นระเบียบ
            
elif stats is None:
    st.error("API Error")
else:
    st.info("ไม่มีโปรแกรมแข่งเร็วๆ นี้")

st.caption("Auto-Predict by Poisson Model v2.0")

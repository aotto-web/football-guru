import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL One-Line", layout="centered")

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
        df['Att'] = (df['GF'] / df['P']) / avg_g
        df['Def'] = (df['GA'] / df['P']) / avg_g
        return df, avg_g, f_data.get('matches', []) if f_data else []
    return None, 1.5, []

def predict(h, a, df, avg):
    try:
        hs, as_ = df[df['N']==h].iloc[0], df[df['N']==a].iloc[0]
        ex_h, ex_a = hs['Att']*as_['Def']*avg, as_['Att']*hs['Def']*avg
        probs = np.outer([poisson.pmf(i, ex_h) for i in range(6)], [poisson.pmf(i, ex_a) for i in range(6)])
        return f"{probs.argmax()//6}-{probs.argmax()%6}", np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))
    except: return "N/A", 0, 0, 0

# --- ส่วนแสดงผลแบบบรรทัดเดียว ---
st.title("⚽ PL GURU: Quick View")

stats, avg_g, fixtures = get_all_data()

if stats is not None and fixtures:
    st.write("` คู่แข่งขัน | สกอร์คาด | ชนะ-เสมอ-แพ้ % `")
    
    for m in fixtures:
        h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
        score, ph, pd, pa = predict(h, a, stats, avg_g)
        
        # แสดงผลบรรทัดเดียวเน้นๆ
        # Format: [Home] vs [Away] | [Score] | [W-D-L %]
        match_str = f"**{h}** vs **{a}**"
        result_str = f"` {score} ` | {ph*100:.0f}%-{pd*100:.0f}%-{pa*100:.0f}%"
        
        st.write(f"{match_str}  \n {result_str}")
        st.divider()
elif stats is None:
    st.error("API Error")
else:
    st.info("ไม่มีแข่งเร็วๆ นี้")

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ (Mobile First) ---
st.set_page_config(page_title="PL GURU", layout="centered")

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
def get_data():
    s_data = call_api("competitions/PL/standings")
    f_data = call_api("competitions/PL/matches?status=SCHEDULED")
    if s_data and 'standings' in s_data:
        table = s_data['standings'][0]['table']
        df = pd.DataFrame([{'N': t['team']['shortName'], 'P': t['playedGames'], 'GF': t['goalsFor'], 'GA': t['goalsAgainst']} for t in table])
        df['P'] = df['P'].replace(0, 1)
        avg = df['GF'].sum() / df['P'].sum()
        df['Att'] = (df['GF'] / df['P']) / (avg if avg > 0 else 1)
        df['Def'] = (df['GA'] / df['P']) / (avg if avg > 0 else 1)
        return df, avg, f_data.get('matches', []) if f_data else []
    return None, 1.5, []

def predict(h, a, df, avg):
    try:
        hs, as_ = df[df['N']==h].iloc[0], df[df['N']==a].iloc[0]
        ex_h, ex_a = hs['Att']*as_['Def']*avg, as_['Att']*hs['Def']*avg
        probs = np.outer([poisson.pmf(i, ex_h) for i in range(6)], [poisson.pmf(i, ex_a) for i in range(6)])
        p_h, p_d, p_a = np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))
        idx = probs.argmax()
        return f"{idx//6}-{idx%6}", p_h, p_d, p_a
    except: return "N/A", 0, 0, 0

# --- การแสดงผล (เลี่ยงการใช้ CSS Block ยาวๆ เพื่อแก้บั๊ก Python 3.13) ---
st.markdown("<h1 style='text-align: center; color: #00ff87; font-family: sans-serif;'>⚽ PREMIER GURU</h1>", unsafe_content_allowed=True)

stats, avg_g, fixtures = get_data()

if stats is not None and fixtures:
    for m in fixtures:
        h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
        score, ph, pd, pa = predict(h, a, stats, avg_g)
        
        # สร้าง Card แบบล่ำๆ ด้วย Inline CSS เพื่อความเสถียร
        card_html = f"""
        <div style="background: linear-gradient(135deg, #1e1e26 0%, #2d2d3a 100%); 
                    padding: 20px; border-radius: 20px; border-left: 10px solid #e90052; 
                    margin-bottom: 20px; color: white; font-family: sans-serif;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <b style="font-size: 18px;">{h}</b>
                <span style="color: #e90052; font-weight: 900;">VS</span>
                <b style="font-size: 18px;">{a}</b>
            </div>
            <div style="text-align: center; margin: 15px 0;">
                <div style="background: #3d0158; padding: 12px 25px; border-radius: 15px; 
                            font-size: 32px; font-weight: 900; color: #00ff87; 
                            display: inline-block; box-shadow: inset 0 0 10px rgba(0,0,0,0.5);">
                    {score}
                </div>
            </div>
            <div style="display: flex; justify-content: space-around; font-size: 14px; font-weight: bold; color: #a0a0a0; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 10px;">
                <span>🏠 {ph*100:.0f}%</span>
                <span>🤝 {pd*100:.0f}%</span>
                <span>🚀 {pa*100:.0f}%</span>
            </div>
            <div style="font-size: 11px; color: #666; margin-top: 15px; text-align: center;">
                🗓️ Match Date: {m['utcDate'][:10]} | Poisson Model Auto-Analysis
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_content_allowed=True)
elif stats is None:
    st.error("API Error: โปรดตรวจสอบการเชื่อมต่อ")
else:
    st.info("ไม่มีโปรแกรมการแข่งขันเร็วๆ นี้")

st.markdown("<p style='text-align: center; color: #555; font-size: 12px;'>Powered by Football-Data API v4</p>", unsafe_content_allowed=True)

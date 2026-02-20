import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ (Mobile First) ---
st.set_page_config(page_title="PL GURU", layout="centered")

# --- CSS แบบปลอดภัย (Inline Markdown) เพื่อความล่ำและสวยงาม ---
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .match-card {
        background: linear-gradient(135deg, #1e1e26 0%, #2d2d3a 100%);
        padding: 20px;
        border-radius: 20px;
        border-left: 8px solid #e90052;
        margin-bottom: 20px;
        color: white;
    }
    .team-name { font-size: 18px; font-weight: 800; color: #ffffff; }
    .vs { color: #e90052; font-style: italic; font-weight: 900; }
    .score-box {
        background: #3d0158;
        padding: 10px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: 900;
        color: #00ff87;
        display: inline-block;
        margin-top: 10px;
        min-width: 80px;
        text-align: center;
    }
    .prob-text { font-size: 14px; color: #a0a0a0; margin-top: 5px; }
</style>
""", unsafe_content_allowed=True)

# --- API Config ---
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
        df['Att'] = (df['GF'] / df['P']) / avg
        df['Def'] = (df['GA'] / df['P']) / avg
        return df, avg, f_data.get('matches', []) if f_data else []
    return None, 1.5, []

def predict(h, a, df, avg):
    try:
        hs, as_ = df[df['N']==h].iloc[0], df[df['N']==a].iloc[0]
        ex_h, ex_a = hs['Att']*as_['Def']*avg, as_['Att']*hs['Def']*avg
        probs = np.outer([poisson.pmf(i, ex_h) for i in range(6)], [poisson.pmf(i, ex_a) for i in range(6)])
        return f"{probs.argmax()//6}-{probs.argmax()%6}", np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))
    except: return "N/A", 0, 0, 0

# --- การแสดงผลแบบ "ล่ำๆ" ---
st.markdown("<h1 style='text-align: center; color: #00ff87;'>⚽ PREMIER GURU</h1>", unsafe_content_allowed=True)

stats, avg_g, fixtures = get_data()

if stats is not None and fixtures:
    for m in fixtures:
        h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
        score, ph, pd, pa = predict(h, a, stats, avg_g)
        
        # UI Card แบบ Solid
        st.markdown(f"""
        <div class="match-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="team-name">{h}</span>
                <span class="vs">VS</span>
                <span class="team-name">{a}</span>
            </div>
            <div style="text-align: center;">
                <div class="score-box">{score}</div>
            </div>
            <div class="prob-text">
                🏠 {ph*100:.0f}% | 🤝 {pd*100:.0f}% | 🚀 {pa*100:.0f}%
            </div>
            <div style="font-size: 10px; color: #666; margin-top: 10px;">
                🗓️ {m['utcDate'][:10]} | Poisson Statistical Model
            </div>
        </div>
        """, unsafe_content_allowed=True)
elif stats is None:
    st.error("API Error: โปรดรีเฟรชหรือเช็ค Key")
else:
    st.info("ไม่มีโปรแกรมการแข่งขันเร็วๆ นี้")

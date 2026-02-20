import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอแบบกะทัดรัด ---
st.set_page_config(page_title="PL Guru", layout="centered", page_icon="⚽")

# --- สไตล์ปรับแต่งให้ดูแพงบนมือถือ ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stExpander"] { border: none !important; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 12px; margin-bottom: 15px; background: white; }
    .stMarkdown h1 { font-size: 24px !important; text-align: center; color: #3d0158; }
    </style>
    """, unsafe_content_allowed=True)

API_KEY = "2ab1eb65a8b94e8ea240487d86d1e6a5"
BASE_URL = "https://api.football-data.org/v4"

def call_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers)
        if response.status_code == 200: return response.json()
        return None
    except: return None

@st.cache_data(ttl=3600)
def get_data():
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
        df['Att'] = (df['GF'] / df['P']) / avg_gf
        df['Def'] = (df['GA'] / df['P']) / avg_gf
        return df, avg_gf, f_data.get('matches', [])
    return None, 1.5, []

def predict(h, a, df, avg):
    try:
        hs, as_ = df[df['Name']==h].iloc[0], df[df['Name']==a].iloc[0]
        ex_h, ex_a = hs['Att']*as_['Def']*avg, as_['Att']*hs['Def']*avg
        probs = np.outer([poisson.pmf(i, ex_h) for i in range(7)], [poisson.pmf(i, ex_a) for i in range(7)])
        return ex_h, ex_a, np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1)), f"{probs.argmax()//7}-{probs.argmax()%7}"
    except: return 0,0,0,0,0,"N/A"

# --- แสดงผล ---
st.write("# ⚽ PL GURU PREDICT")

stats, avg_g, fixtures = get_data()

if stats is not None:
    if not fixtures:
        st.info("ไม่มีคู่แข่งสัปดาห์นี้")
    else:
        for m in fixtures:
            h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
            xh, xa, ph, pd, pa, score = predict(h, a, stats, avg_g)
            
            # การ์ดแสดงผลรายคู่
            with st.expander(f"**{h} vs {a}**", expanded=True):
                # แถวบน: เปอร์เซ็นต์ชนะ
                c1, c2, c3 = st.columns(3)
                c1.metric("🏠 Win", f"{ph*100:.0f}%")
                c2.metric("🤝 Draw", f"{pd*100:.0f}%")
                c3.metric("🚀 Win", f"{pa*100:.0f}%")
                
                # แถวล่าง: ผลทำนาย
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #3d0158; color: white; border-radius: 8px; margin-top: 10px;">
                    <span style="font-size: 14px;">🎯 สกอร์ที่คาด:</span><br>
                    <b style="font-size: 22px;">{score}</b>
                </div>
                """, unsafe_content_allowed=True)
                
                st.caption(f"xG: {xh:.1f} - {xa:.1f} | เตะวันที่: {m['utcDate'][:10]}")

    # แถบด้านข้างสำหรับดูตารางแบบย่อ
    with st.sidebar:
        st.header("🏆 Top Attackers")
        st.dataframe(stats.sort_values('Att', ascending=False)[['Name', 'GF']].head(5), hide_index=True)
else:
    st.error("API Error - กรุณารีเฟรช")

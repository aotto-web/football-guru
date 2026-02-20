import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Unstoppable Predictor", layout="wide")
st.title("🏆 Premier League Predictor (API Version)")

# ใส่ API Key ของคุณที่นี่ (สมัครฟรีที่ football-data.org)
API_KEY = "ใส่_API_KEY_ของคุณที่ตรงนี้" 

# --- ฟังก์ชันดึงข้อมูลผ่าน API ---
def fetch_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    url = f"https://api.football-data.org/v4/{endpoint}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code}. ตรวจสอบ API Key ของคุณ")
        return None

# --- 1. ดึงตารางคะแนนเพื่อหาค่า Strength ---
@st.cache_data(ttl=3600)
def get_stats():
    data = fetch_api("competitions/PL/standings")
    if data:
        table = data['standings'][0]['table']
        rows = []
        for team in table:
            rows.append({
                'Team': team['team']['shortName'],
                'Played': team['playedGames'],
                'GF': team['goalsFor'],
                'GA': team['goalsAgainst'],
                'Pts': team['points']
            })
        df = pd.DataFrame(rows)
        
        # คำนวณค่าเฉลี่ยลีก
        avg_gf = df['GF'].mean()
        avg_ga = df['GA'].mean()
        
        # คำนวณ Strength
        df['Offense'] = df['GF'] / avg_gf
        df['Defense'] = df['GA'] / avg_ga
        
        return df, avg_gf / df['Played'].mean(), avg_ga / df['Played'].mean()
    return None, 1.5, 1.3

# --- 2. ดึงโปรแกรมการแข่งขัน (Fixtures) อัตโนมัติ ---
@st.cache_data(ttl=3600)
def get_fixtures():
    data = fetch_api("competitions/PL/matches?status=SCHEDULED")
    if data:
        matches = data['matches']
        upcoming = []
        for m in matches[:10]: # เอา 10 คู่ถัดไป
            upcoming.append({
                'Date': m['utcDate'][:10],
                'Home': m['homeTeam']['shortName'],
                'Away': m['awayTeam']['shortName']
            })
        return pd.DataFrame(upcoming)
    return None

# --- 3. สูตรคำนวณทำนายผล ---
def predict(home, away, df, avg_h, avg_a):
    try:
        h_s = df[df['Team'] == home].iloc[0]
        a_s = df[df['Team'] == away].iloc[0]
        
        exp_h = h_s['Offense'] * a_s['Defense'] * avg_h
        exp_a = a_s['Offense'] * h_s['Defense'] * avg_a
        
        # Poisson Matrix
        h_p = [poisson.pmf(i, exp_h) for i in range(7)]
        a_p = [poisson.pmf(i, exp_a) for i in range(7)]
        matrix = np.outer(h_p, a_p)
        
        prob_h = np.sum(np.tril(matrix, -1))
        prob_d = np.sum(np.diag(matrix))
        prob_a = np.sum(np.triu(matrix, 1))
        hp, ap = np.unravel_index(matrix.argmax(), matrix.shape)
        
        return exp_h, exp_a, prob_h, prob_d, prob_a, f"{hp}-{ap}"
    except:
        return 0,0,0,0,0,"N/A"

# --- การแสดงผล ---
if API_KEY == "ใส่_API_KEY_ของคุณที่ตรงนี้":
    st.warning("⚠️ กรุณาใส่ API Key ใน Code เพื่อเริ่มการดึงข้อมูลอัตโนมัติ")
else:
    df_stats, ah, aa = get_stats()
    df_fix = get_fixtures()

    if df_stats is not None:
        st.sidebar.header("📊 ค่าพลังทีมปัจจุบัน")
        st.sidebar.dataframe(df_stats[['Team', 'Offense', 'Defense']])

        st.header("📅 โปรแกรมการแข่งขันนัดถัดไป (Auto-Loaded)")
        if df_fix is not None and not df_fix.empty:
            for _, match in df_fix.iterrows():
                xh, xa, ph, pd, pa, score = predict(match['Home'], match['Away'], df_stats, ah, aa)
                with st.expander(f"🏟️ {match['Date']} | {match['Home']} vs {match['Away']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("เจ้าบ้านชนะ", f"{ph*100:.1f}%")
                    c2.metric("เสมอ", f"{pd*100:.1f}%")
                    c3.metric("ทีมเยือนชนะ", f"{pa*100:.1f}%")
                    st.write(f"สกอร์ที่คาด: **{score}** (xG: {xh:.2f} - {xa:.2f})")
        else:
            st.info("ไม่มีโปรแกรมการแข่งขันที่กำลังจะมาถึง")

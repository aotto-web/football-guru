import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Auto-Guru", layout="wide", page_icon="⚽")
st.title("🏆 Premier League Auto-Predictor (API v4)")

# ใส่ API Key ที่ได้รับจากอีเมล (ตัวอย่าง: 'your_api_key_here')
API_KEY = "ใส่_API_KEY_ของคุณที่ตรงนี้" 
BASE_URL = "https://api.football-data.org/v4"

# --- ฟังก์ชันช่วยดึงข้อมูลจาก API ---
def call_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("Error 429: คุณดึงข้อมูลบ่อยเกินไป (Free Tier จำกัดจำนวนครั้งต่อนาที)")
            return None
        else:
            st.error(f"Error {response.status_code}: ไม่สามารถดึงข้อมูลได้")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# --- 1. ดึงตารางคะแนนสด (Standings) ---
@st.cache_data(ttl=3600)
def get_league_stats():
    data = call_api("competitions/PL/standings")
    if data:
        # ดึงตารางแบบ Total (เหย้า+เยือน)
        table = data['standings'][0]['table']
        rows = []
        for team in table:
            rows.append({
                'TeamID': team['team']['id'],
                'TeamName': team['team']['shortName'],
                'Played': team['playedGames'],
                'GF': team['goalsFor'],
                'GA': team['goalsAgainst'],
                'Pts': team['points']
            })
        df = pd.DataFrame(rows)
        
        # คำนวณค่าเฉลี่ยประตูต่อเกมของลีก (ใช้หา xG)
        avg_gf = df['GF'].sum() / df['Played'].sum()
        
        # คำนวณความแข็งแกร่ง (Strength)
        df['Att_Strength'] = (df['GF'] / df['Played']) / avg_gf
        df['Def_Strength'] = (df['GA'] / df['Played']) / avg_gf
        
        return df, avg_gf
    return None, 1.3

# --- 2. ดึงโปรแกรมการแข่งขันนัดถัดไป (Scheduled Matches) ---
@st.cache_data(ttl=3600)
def get_upcoming_matches():
    # ดึงเฉพาะคู่ใน Premier League ที่มีสถานะ SCHEDULED
    data = call_api("competitions/PL/matches?status=SCHEDULED")
    if data:
        matches = data['matches']
        match_list = []
        for m in matches:
            match_list.append({
                'Home': m['homeTeam']['shortName'],
                'Away': m['awayTeam']['shortName'],
                'Date': m['utcDate']
            })
        return match_list
    return []

# --- 3. ฟังก์ชันคำนวณผล (Poisson) ---
def predict_score(home_name, away_name, stats_df, avg_league_goals):
    try:
        h_stat = stats_df[stats_df['TeamName'] == home_name].iloc[0]
        a_stat = stats_df[stats_df['TeamName'] == away_name].iloc[0]
        
        # สูตร xG: บุกเจ้าบ้าน * รับทีมเยือน * ค่าเฉลี่ยลีก
        exp_h = h_stat['Att_Strength'] * a_stat['Def_Strength'] * avg_league_goals
        exp_a = a_stat['Att_Strength'] * h_stat['Def_Strength'] * avg_league_goals
        
        # คำนวณความเป็นไปได้
        h_probs = [poisson.pmf(i, exp_h) for i in range(7)]
        a_probs = [poisson.pmf(i, exp_a) for i in range(7)]
        prob_matrix = np.outer(h_probs, a_probs)
        
        p_home = np.sum(np.tril(prob_matrix, -1))
        p_draw = np.sum(np.diag(prob_matrix))
        p_away = np.sum(np.triu(prob_matrix, 1))
        
        # สกอร์ที่น่าจะเป็นที่สุด (Max Probability)
        res_h, res_a = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
        
        return exp_h, exp_a, p_home, p_draw, p_away, f"{res_h}-{res_a}"
    except:
        return 0, 0, 0, 0, 0, "N/A"

# --- MAIN APP ---
if API_KEY == "ใส่_API_KEY_ของคุณที่ตรงนี้":
    st.info("💡 โปรดใส่ API Key ที่ได้รับจาก football-data.org ในโค้ดบรรทัดที่ 12")
else:
    stats, avg_g = get_league_stats()
    fixtures = get_upcoming_matches()

    if stats is not None:
        st.sidebar.header("📊 ค่าพลังทีม (Strength)")
        st.sidebar.dataframe(stats[['TeamName', 'Att_Strength', 'Def_Strength']], hide_index=True)

        if fixtures:
            st.header(f"📅 วิเคราะห์โปรแกรมที่กำลังจะมาถึง ({len(fixtures)} คู่)")
            for match in fixtures:
                xh, xa, ph, pd, pa, score = predict_score(match['Home'], match['Away'], stats, avg_g)
                
                with st.expander(f"🏟️ {match['Home']} vs {match['Away']} (เตะเมื่อ: {match['Date'][:10]})"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("เจ้าบ้านชนะ", f"{ph*100:.1f}%")
                    c2.metric("เสมอ", f"{pd*100:.1f}%")
                    c3.metric("ทีมเยือนชนะ", f"{pa*100:.1f}%")
                    st.write(f"🎯 สกอร์ที่น่าจะเป็นที่สุด: **{score}** | xG: {xh:.2f} - {xa:.2f}")
        else:
            st.write("ไม่มีโปรแกรมการแข่งขันในเร็วๆ นี้")

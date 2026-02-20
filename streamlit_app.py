import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Auto-Guru 2026", layout="wide", page_icon="⚽")
st.title("🏆 Premier League Auto-Predictor (Live API)")

# API Key ที่คุณให้มา
API_KEY = "2ab1eb65a8b94e8ea240487d86d1e6a5" 
BASE_URL = "https://api.football-data.org/v4"

# --- ฟังก์ชันดึงข้อมูลจาก API ---
def call_api(endpoint):
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("⚠️ Rate Limit: คุณดึงข้อมูลบ่อยเกินไป (Free Tier จำกัดจำนวนครั้งต่อนาที)")
            return None
        else:
            st.error(f"❌ Error {response.status_code}: ไม่สามารถดึงข้อมูลได้")
            return None
    except Exception as e:
        st.error(f"📡 Connection Error: {e}")
        return None

# --- 1. ดึงตารางคะแนนสดเพื่อคำนวณ Strength ---
@st.cache_data(ttl=3600)
def get_league_stats():
    data = call_api("competitions/PL/standings")
    if data and 'standings' in data:
        table = data['standings'][0]['table']
        rows = []
        for team in table:
            rows.append({
                'TeamName': team['team']['shortName'],
                'Played': team['playedGames'],
                'GF': team['goalsFor'],
                'GA': team['goalsAgainst'],
                'Pts': team['points']
            })
        df = pd.DataFrame(rows)
        
        # ป้องกันการหารด้วยศูนย์กรณีเริ่มฤดูกาล
        df['Played'] = df['Played'].replace(0, 1)
        
        # คำนวณค่าเฉลี่ยประตูต่อเกมของทั้งลีก
        avg_gf = df['GF'].sum() / df['Played'].sum()
        
        # คำนveณความแข็งแกร่ง (Strength)
        df['Att_Strength'] = (df['GF'] / df['Played']) / avg_gf
        df['Def_Strength'] = (df['GA'] / df['Played']) / avg_gf
        
        return df, avg_gf
    return None, 1.5

# --- 2. ดึงโปรแกรมการแข่งขันที่กำลังจะมาถึง ---
@st.cache_data(ttl=3600)
def get_upcoming_matches():
    data = call_api("competitions/PL/matches?status=SCHEDULED")
    if data and 'matches' in data:
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

# --- 3. ฟังก์ชันคำนวณผลด้วย Poisson ---
def predict_score(home_name, away_name, stats_df, avg_league_goals):
    try:
        # ดึงค่าพลังของทั้งสองทีม
        h_stat = stats_df[stats_df['TeamName'] == home_name].iloc[0]
        a_stat = stats_df[stats_df['TeamName'] == away_name].iloc[0]
        
        # สูตร Expected Goals (xG)
        exp_h = h_stat['Att_Strength'] * a_stat['Def_Strength'] * avg_league_goals
        exp_a = a_stat['Att_Strength'] * h_stat['Def_Strength'] * avg_league_goals
        
        # คำนวณความน่าจะเป็น 0-7 ประตู
        h_probs = [poisson.pmf(i, exp_h) for i in range(8)]
        a_probs = [poisson.pmf(i, exp_a) for i in range(8)]
        prob_matrix = np.outer(h_probs, a_probs)
        
        p_home = np.sum(np.tril(prob_matrix, -1))
        p_draw = np.sum(np.diag(prob_matrix))
        p_away = np.sum(np.triu(prob_matrix, 1))
        
        # สกอร์ที่มีโอกาสเกิดสูงสุด
        res_h, res_a = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
        
        return exp_h, exp_a, p_home, p_draw, p_away, f"{res_h}-{res_a}"
    except:
        return 0, 0, 0, 0, 0, "N/A"

# --- ส่วนการแสดงผลบนหน้าเว็บ ---
stats, avg_g = get_league_stats()
fixtures = get_upcoming_matches()

if stats is not None:
    # Sidebar: ตารางค่าพลัง
    st.sidebar.header("📊 Team Strength Index")
    st.sidebar.write("อ้างอิงจากฟอร์มปัจจุบัน")
    st.sidebar.dataframe(
        stats[['TeamName', 'Att_Strength', 'Def_Strength']].sort_values('Att_Strength', ascending=False),
        hide_index=True
    )

    # หน้าหลัก: รายการแข่ง
    if fixtures:
        st.header(f"📅 วิเคราะห์โปรแกรมล่วงหน้า ({len(fixtures)} คู่)")
        
        # วนลูปสร้างการ์ดวิเคราะห์รายคู่
        for match in fixtures:
            xh, xa, ph, pd, pa, score = predict_score(match['Home'], match['Away'], stats, avg_g)
            
            with st.expander(f"🏟️ {match['Home']} vs {match['Away']} ({match['Date'][:10]})"):
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{match['Home']} ชนะ", f"{ph*100:.1f}%")
                c2.metric("เสมอ", f"{pd*100:.1f}%")
                c3.metric(f"{match['Away']} ชนะ", f"{pa*100:.1f}%")
                
                st.write(f"**🎯 สกอร์ที่น่าจะเป็นที่สุด:** :green[{score}]")
                st.write(f"**💡 บทวิเคราะห์:** ค่า xG คาดการณ์ {match['Home']} **{xh:.2f}** และ {match['Away']} **{xa:.2f}**")
    else:
        st.info("ขณะนี้ไม่มีโปรแกรมการแข่งขันพรีเมียร์ลีกที่บันทึกอยู่ในระบบ")
else:
    st.warning("กรุณารอครู่หนึ่ง ระบบกำลังเชื่อมต่อข้อมูลจาก API...")

st.divider()
st.caption("Data provided by Football-Data.org API. คำนวณผลด้วยหลักการทางสถิติ Poisson Distribution")

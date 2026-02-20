import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from textwrap import dedent

# --- ตั้งค่าหน้าจอแบบ Mobile-First ---
st.set_page_config(page_title="PL GURU", layout="centered", page_icon="⚽")

# --- แก้ไขสไตล์ CSS (ใช้ dedent เพื่อป้องกัน Error ใน Python 3.13) ---
st.markdown(dedent("""
<style>
    .main { background-color: #f0f2f6; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    [data-testid="stExpander"] { 
        border: none !important; 
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); 
        border-radius: 15px; 
        margin-bottom: 20px; 
        background: white; 
    }
    .stMarkdown h1 { 
        font-size: 28px !important; 
        text-align: center; 
        color: #3d0158; 
        padding-bottom: 20px;
    }
    .predict-box {
        text-align: center; 
        padding: 15px; 
        background: linear-gradient(90deg, #3d0158, #e90052); 
        color: white; 
        border-radius: 12px; 
        margin-top: 15px;
    }
</style>
"""), unsafe_content_allowed=True)

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
        
        # ป้องกันการหารด้วย 0
        df['P'] = df['P'].replace(0, 1)
        
        # คำนวณค่าเฉลี่ยลีกเพื่อหา xG
        avg_gf = df['GF'].sum() / df['P'].sum()
        
        # คำนวณความแข็งแกร่ง บุก/รับ
        df['Att'] = (df['GF'] / df['P']) / avg_gf
        df['Def'] = (df['GA'] / df['P']) / avg_gf
        
        matches = f_data.get('matches', []) if f_data else []
        return df, avg_gf, matches
    return None, 1.5, []

def predict_match(h_name, a_name, df, avg_league):
    try:
        h_stat = df[df['Name'] == h_name].iloc[0]
        a_stat = df[df['Name'] == a_name].iloc[0]
        
        # สูตร xG
        ex_h = h_stat['Att'] * a_stat['Def'] * avg_league
        ex_a = a_stat['Att'] * h_stat['Def'] * avg_league
        
        # Poisson Matrix (0-6 ประตู)
        h_probs = [poisson.pmf(i, ex_h) for i in range(7)]
        a_probs = [poisson.pmf(i, ex_a) for i in range(7)]
        matrix = np.outer(h_probs, a_probs)
        
        p_h = np.sum(np.tril(matrix, -1))
        p_d = np.sum(np.diag(matrix))
        p_a = np.sum(np.triu(matrix, 1))
        
        # หาตัวเลขสกอร์ที่น่าจะเป็นที่สุด
        score_idx = matrix.argmax()
        score_h = score_idx // 7
        score_a = score_idx % 7
        
        return ex_h, ex_a, p_h, p_d, p_a, f"{score_h}-{score_a}"
    except:
        return 0, 0, 0, 0, 0, "N/A"

# --- แสดงผลหน้าหลัก ---
st.markdown("# ⚽ PREMIER GURU")

stats, avg_g, fixtures = get_all_data()

if stats is not None:
    if not fixtures:
        st.info("ไม่มีโปรแกรมการแข่งขันที่กำลังจะมาถึง")
    else:
        st.write(f"### วิเคราะห์ล่วงหน้า {len(fixtures)} คู่")
        for m in fixtures:
            h_team = m['homeTeam']['shortName']
            a_team = m['awayTeam']['shortName']
            match_date = m['utcDate'][:10]
            
            xh, xa, ph, pd, pa, score = predict_match(h_team, a_team, stats, avg_g)
            
            with st.expander(f"**{h_team} vs {a_team}**", expanded=True):
                # ส่วนของความน่าจะเป็น (Metric)
                c1, c2, c3 = st.columns(3)
                c1.metric("🏠 Win", f"{ph*100:.0f}%")
                c2.metric("🤝 Draw", f"{pd*100:.0f}%")
                c3.metric("🚀 Win", f"{pa*100:.1f}%")
                
                # ส่วนของสกอร์คาดการณ์ (Custom HTML)
                st.markdown(f"""
                <div class="predict-box">
                    <span style="font-size: 14px; opacity: 0.9;">🎯 สกอร์ที่คาด:</span><br>
                    <b style="font-size: 26px;">{score}</b>
                </div>
                """, unsafe_content_allowed=True)
                
                # สถิติเสริมเล็กๆ
                st.caption(f"📅 {match_date} | xG: {xh:.1f} - {xa:.1f}")

    # Sidebar สำหรับดูข้อมูลเพิ่มเติม
    with st.sidebar:
        st.title("📊 League Stats")
        st.write("ทีมที่บุกโหดที่สุด (Top 5)")
        top_att = stats.sort_values('Att', ascending=False).head(5)
        st.table(top_att[['Name', 'GF']])
        st.caption("อัปเดตอัตโนมัติจาก API v4")
else:
    st.error("ไม่สามารถเชื่อมต่อ API ได้ โปรดตรวจสอบ Key หรืออินเทอร์เน็ต")

st.divider()
st.center = st.caption("© 2026 Football Guru Predictor - Poisson Model")

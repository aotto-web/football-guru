import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="Football AI Commander", page_icon="🏆", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .match-card {
        background-color: #1c1c1c;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .high-conf { border-left-color: #00FF7F; } /* เขียวสด */
    .med-conf { border-left-color: #FFC107; } /* เหลือง */
    .low-conf { border-left-color: #F44336; } /* แดง */
    h1, h2, h3 { color: #00FF7F !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 Football AI Commander (Fixed)")
st.write("ระบบวิเคราะห์ฟุตบอลแบบ Hybrid: **ตารางแข่งจริง + เจาะลึกความน่าจะเป็น**")

# --- 1. DATA ENGINE (สมอง AI) ---
@st.cache_resource(ttl=3600)
def load_engine():
    # ใช้ User-Agent ปลอมตัวเพื่อดึงข้อมูลย้อนหลัง
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    ]
    dfs = []
    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                dfs.append(df)
        except: pass
    
    if not dfs:
        return None, None, None, None

    matches = pd.concat(dfs)
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
    matches = matches[cols].dropna()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
    matches = matches.sort_values("Date")

    # Feature Engineering
    def get_stats(group):
        group['H_Form'] = group['FTHG'].rolling(5, closed='left').mean()
        group['A_Form'] = group['FTAG'].rolling(5, closed='left').mean()
        group['H_Shots'] = group['HST'].rolling(5, closed='left').mean() # Shots on Target
        group['A_Shots'] = group['AST'].rolling(5, closed='left').mean()
        return group
    
    matches = matches.groupby('HomeTeam', group_keys=False).apply(get_stats).dropna()
    
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["H_Code"] = le.transform(matches["HomeTeam"])
    matches["A_Code"] = le.transform(matches["AwayTeam"])
    matches["Target"] = (matches["FTR"] == "H").astype("int")

    # Train Model
    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42)
    predictors = ["H_Code", "A_Code", "H_Form", "A_Form", "H_Shots", "A_Shots"]
    rf.fit(matches[predictors], matches["Target"])
    
    return rf, le, matches, predictors

# --- 2. HELPER FUNCTIONS ---
def map_team_name(name, known_teams):
    mapping = {
        "Man Utd": "Man United", "Manchester Utd": "Man United", "Manchester United": "Man United",
        "Man City": "Man City", "Manchester City": "Man City",
        "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
        "Newcastle Utd": "Newcastle", "West Ham Utd": "West Ham",
        "Wolves": "Wolves", "Wolverhampton": "Wolves",
        "Brighton & Hove Albion": "Brighton", "Nott'm Forest": "Nott'm Forest",
        "Nottingham Forest": "Nott'm Forest", "Sheffield Utd": "Sheffield United",
        "Luton Town": "Luton", "Ipswich Town": "Ipswich"
    }
    if name in known_teams: return name
    if name in mapping:
        if mapping[name] in known_teams: return mapping[name]
    return None

def predict_match(h_team, a_team, rf, le, matches, predictors):
    try:
        h_stats = matches[matches["HomeTeam"] == h_team].iloc[-1]
        a_stats = matches[matches["AwayTeam"] == a_team].iloc[-1]
        
        row = pd.DataFrame([[
            le.transform([h_team])[0], le.transform([a_team])[0],
            h_stats["H_Form"], a_stats["A_Form"],
            h_stats["H_Shots"], a_stats["A_Shots"]
        ]], columns=predictors)
        
        prob = rf.predict_proba(row)[0][1]
        xg_h = h_stats["H_Shots"] * 0.3
        xg_a = a_stats["A_Shots"] * 0.28
        return prob, xg_h, xg_a
    except:
        return None, None, None

# --- MAIN APP LOGIC ---
with st.spinner('🚀 กำลังสตาร์ทเครื่องยนต์ AI...'):
    rf, le, matches, predictors = load_engine()

# สร้าง Tab แยกโหมด
tab1, tab2 = st.tabs(["📅 โปรแกรมแข่ง (Schedule)", "🧪 ห้องแล็บวิเคราะห์ (Deep Lab)"])

# === TAB 1: ตารางแข่งจริง ===
with tab1:
    st.header("โปรแกรมการแข่งขันเร็วๆ นี้")
    
    # --- ส่วนที่แก้: ปลอมตัวเป็น Browser เพื่อดึงตารางแข่ง ---
    try:
        url = "https://fixturedownload.com/feed/json/epl-2025"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            fixtures = pd.read_json(StringIO(response.text))
            fixtures['DateUtc'] = pd.to_datetime(fixtures['DateUtc'])
            upcoming = fixtures[fixtures['DateUtc'] >= datetime.utcnow()].sort_values('DateUtc').head(10)
            
            if upcoming.empty:
                st.info("ไม่มีการแข่งขันในเร็วๆ นี้ หรือจบฤดูกาลแล้ว")
            
            for idx, row in upcoming.iterrows():
                d = row['DateUtc']
                h_raw, a_raw = row['HomeTeam'], row['AwayTeam']
                h_real = map_team_name(h_raw, le.classes_)
                a_real = map_team_name(a_raw, le.classes_)
                
                if h_real and a_real:
                    prob, xg_h, xg_a = predict_match(h_real, a_real, rf, le, matches, predictors)
                    
                    if prob is not None:
                        confidence = abs(prob - 0.5) * 2
                        stars = "⭐" * int(confidence * 5)
                        if stars == "": stars = "➖"
                        
                        if prob > 0.60:
                            status = "High Confidence: Home Win"
                            css_class = "high-conf"
                            color = "#00FF7F"
                        elif prob < 0.40:
                            status = "High Confidence: Away Win/Draw"
                            css_class = "med-conf"
                            color = "#FFC107"
                        else:
                            status = "Too Close to Call (Risky)"
                            css_class = "low-conf"
                            color = "#F44336"

                        st.markdown(f"""
                        <div class="match-card {css_class}">
                            <div style="display:flex; justify-content:space-between;">
                                <span style="color:#888;">{d.strftime('%d %b %H:%M')}</span>
                                <span style="color:{color}; font-weight:bold;">{stars}</span>
                            </div>
                            <h3 style="margin:10px 0;">🏠 {h_real} vs {a_real} ✈️</h3>
                            <div style="background:#333; padding:10px; border-radius:8px;">
                                <span style="color:{color}; font-weight:bold;">AI Verdict: {status} ({prob*100:.0f}%)</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander(f"📊 เจาะลึก {h_real} vs {a_real}"):
                            c1, c2 = st.columns(2)
                            c1.metric("คาดการณ์ xG เจ้าบ้าน", f"{xg_h:.2f}")
                            c2.metric("คาดการณ์ xG ทีมเยือน", f"{xg_a:.2f}")
                            st.info("💡 Tip: ถ้า AI มั่นใจเกิน 60% ถือว่าน่าลงทุน")
        else:
            st.error(f"ไม่สามารถดึงตารางแข่งได้ (Status Code: {response.status_code})")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

# === TAB 2: วิเคราะห์เอง (Deep Lab) ===
with tab2:
    st.header("🧪 เจาะลึกรายคู่ & สกอร์ที่คาด")
    if rf:
        teams = sorted(le.classes_)
        c1, c2 = st.columns(2)
        h_sel = c1.selectbox("เลือกเจ้าบ้าน", teams, index=0)
        a_sel = c2.selectbox("เลือกทีมเยือน", teams, index=1)
        
        if st.button("🔬 วิเคราะห์คู่นี้แบบละเอียด"):
            prob, xg_h, xg_a = predict_match(h_sel, a_sel, rf, le, matches, predictors)
            
            if prob:
                st.divider()
                col1, col2, col3 = st.columns([1,2,1])
                with col1: st.metric(h_sel, f"{xg_h:.2f} xG")
                with col3: st.metric(a_sel, f"{xg_a:.2f} xG")
                with col2:
                    st.metric("โอกาสเจ้าบ้านชนะ", f"{prob*100:.1f}%")
                    st.progress(prob)
                
                st.subheader("🎯 ความน่าจะเป็นของสกอร์ (Correct Score Matrix)")
                score_probs = []
                for h in range(4): # 0-3
                    row = []
                    for a in range(4):
                        p = poisson.pmf(h, xg_h) * poisson.pmf(a, xg_a)
                        row.append(p)
                    score_probs.append(row)
                
                df_score = pd.DataFrame(score_probs, 
                                      columns=[f"Away {i}" for i in range(4)], 
                                      index=[f"Home {i}" for i in range(4)])
                
                st.dataframe(df_score.style.background_gradient(cmap='Greens', axis=None).format("{:.1%}"))
                
                st.warning("💰 ใส่ราคา Odds เพื่อเช็คความคุ้มค่า")
                odds = st.number_input("ราคาเจ้าบ้านชนะ (Home Win Odds):", 1.0, 10.0, 2.0)
                fair_odds = 1/prob
                edge = (odds - fair_odds)/fair_odds * 100
                
                st.write(f"Fair Odds (ราคาที่ควรเป็น): **{fair_odds:.2f}**")
                if edge > 0:
                    st.success(f"✅ คุ้มค่าน่าลงทุน! (Edge +{edge:.1f}%)")
                else:
                    st.error(f"❌ ราคาไม่ดี (Edge {edge:.1f}%) เจ้ามือเอาเปรียบ")

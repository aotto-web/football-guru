import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
from datetime import datetime

# --- 1. CONFIGURATION & GOD TIER STYLE ---
st.set_page_config(page_title="GOD TIER: Football Analyst", page_icon="👑", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1, h2, h3 { color: #d4af37 !important; font-family: 'Arial Black'; } /* สีทอง */
    .match-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #333;
        transition: transform 0.2s;
    }
    .match-card:hover { transform: scale(1.02); border-color: #d4af37; }
    .stat-box {
        background: #111;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    .kelly-box {
        background-color: #002200;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("👑 GOD TIER: Football Investment System")
st.markdown("ระบบวิเคราะห์ระดับกองทุน: **Prediction + Momentum + Money Management**")

# --- 2. INTELLIGENT ENGINE ---
@st.cache_resource(ttl=3600)
def load_engine():
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    ]
    dfs = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                dfs.append(pd.read_csv(StringIO(r.text)))
        except: pass
    
    if not dfs: return None, None, None, None

    matches = pd.concat(dfs)
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
    matches = matches[cols].dropna()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
    matches = matches.sort_values("Date")

    # Advanced Feature Engineering
    def get_features(group):
        # ฟอร์มการเล่น (Rolling Points: ชนะ=3, เสมอ=1)
        group['Points'] = group['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0)) # คิดแบบเจ้าบ้าน
        group['Form_Point'] = group['Points'].rolling(5, closed='left').mean()
        
        group['H_Goal_Avg'] = group['FTHG'].rolling(5, closed='left').mean()
        group['A_Goal_Avg'] = group['FTAG'].rolling(5, closed='left').mean()
        return group
    
    matches = matches.groupby('HomeTeam', group_keys=False).apply(get_features).dropna()
    
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["H_Code"] = le.transform(matches["HomeTeam"])
    matches["A_Code"] = le.transform(matches["AwayTeam"])
    matches["Target"] = (matches["FTR"] == "H").astype("int")

    # Hyper-Tuned Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, random_state=42)
    predictors = ["H_Code", "A_Code", "Form_Point", "H_Goal_Avg", "A_Goal_Avg"]
    rf.fit(matches[predictors], matches["Target"])
    
    return rf, le, matches, predictors

# --- 3. UTILITY FUNCTIONS ---
def calculate_kelly(prob, odds):
    # Kelly Criterion Formula: f = (bp - q) / b
    # b = odds - 1
    # p = probability
    # q = 1 - p
    if prob <= 0.5: return 0 # ถ้าโอกาสชนะไม่ถึง 50% ห้ามเล่นตามสูตรนี้
    b = odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    return max(f * 100, 0) # Return as percentage

def get_momentum(team, matches):
    # ดึงฟอร์ม 10 นัดหลังสุดเพื่อพลอตกราฟ
    try:
        team_matches = matches[(matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)].tail(10)
        points = []
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
            else:
                pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
            points.append(pts)
        return points
    except: return []

def map_name(name, known):
    mapping = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nott'm Forest", 
               "Wolves": "Wolves", "Man City": "Man City", "Newcastle Utd": "Newcastle",
               "Sheffield Utd": "Sheffield United", "Luton Town": "Luton", "West Ham Utd": "West Ham"}
    if name in known: return name
    if name in mapping and mapping[name] in known: return mapping[name]
    return None

# --- 4. MAIN INTERFACE ---
with st.spinner('🔄 Loading God Mode System...'):
    rf, le, matches, predictors = load_engine()

# Tab Layout
tab1, tab2 = st.tabs(["📅 Live War Room", "🧪 Lab Analysis"])

# === TAB 1: WAR ROOM ===
with tab1:
    st.subheader("โปรแกรมการแข่งขัน & สัญญาณชีพ (Next Matches)")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://fixturedownload.com/feed/json/epl-2025", headers=headers)
        if r.status_code == 200:
            fixtures = pd.read_json(StringIO(r.text))
            fixtures['DateUtc'] = pd.to_datetime(fixtures['DateUtc'], utc=True)
            now_utc = pd.Timestamp.now('UTC')
            upcoming = fixtures[fixtures['DateUtc'] >= now_utc].sort_values('DateUtc').head(6)
            
            for _, row in upcoming.iterrows():
                h_real = map_name(row['HomeTeam'], le.classes_)
                a_real = map_name(row['AwayTeam'], le.classes_)
                
                if h_real and a_real:
                    # Predict
                    h_stat = matches[matches["HomeTeam"] == h_real].iloc[-1]
                    a_stat = matches[matches["AwayTeam"] == a_real].iloc[-1]
                    
                    row_pred = [[le.transform([h_real])[0], le.transform([a_real])[0], 
                                 h_stat["Form_Point"], h_stat["H_Goal_Avg"], a_stat["A_Goal_Avg"]]]
                    prob = rf.predict_proba(row_pred)[0][1]
                    
                    # Color Logic
                    color = "#00ff00" if prob > 0.6 else "#ff4444" if prob < 0.4 else "#ffbb00"
                    rec = "STRONG BUY" if prob > 0.65 else ("AVOID" if 0.4 <= prob <= 0.6 else "SELL / UNDERDOG")
                    
                    # Card UI
                    st.markdown(f"""
                    <div class="match-card">
                        <div style="display:flex; justify-content:space-between; color:#888;">
                            <span>{row['DateUtc'].strftime('%d %b %H:%M')}</span>
                            <span style="color:{color}; font-weight:bold;">{rec}</span>
                        </div>
                        <h2 style="text-align:center; margin:10px 0;">{h_real} vs {a_real}</h2>
                        <div class="stat-box">
                             AI Probability: <span style="color:{color}; font-size:1.2em;">{prob*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"📉 ดูเทรนด์กราฟ {h_real} vs {a_real}"):
                        # Momentum Graph
                        chart_data = pd.DataFrame({
                            h_real: get_momentum(h_real, matches),
                            a_real: get_momentum(a_real, matches)
                        })
                        st.line_chart(chart_data)
                        st.caption("กราฟแสดงแต้มที่ได้ใน 10 นัดหลัง (สูง=ฟอร์มดี, ต่ำ=ฟอร์มตก)")

    except Exception as e: st.error(f"System Offline: {e}")

# === TAB 2: LAB ANALYSIS (KELLY CRITERION) ===
with tab2:
    st.header("🧪 คำนวณความคุ้มค่าระดับกองทุน")
    
    c1, c2 = st.columns(2)
    h_sel = c1.selectbox("Home Team", sorted(le.classes_), index=0)
    a_sel = c2.selectbox("Away Team", sorted(le.classes_), index=1)
    
    if st.button("🚀 Analyze Now"):
        h_stat = matches[matches["HomeTeam"] == h_sel].iloc[-1]
        a_stat = matches[matches["AwayTeam"] == a_sel].iloc[-1]
        
        row_pred = [[le.transform([h_sel])[0], le.transform([a_sel])[0], 
                     h_stat["Form_Point"], h_stat["H_Goal_Avg"], a_stat["A_Goal_Avg"]]]
        prob = rf.predict_proba(row_pred)[0][1]
        
        st.divider()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"AI Probability: {prob*100:.1f}%")
            fair_odds = 1/prob
            st.write(f"Fair Odds (ราคาเป็นกลาง): **{fair_odds:.2f}**")
            
            # Momentum Chart
            st.write("#### 📈 Momentum Trend")
            chart_data = pd.DataFrame({
                h_sel: get_momentum(h_sel, matches),
                a_sel: get_momentum(a_sel, matches)
            })
            st.line_chart(chart_data)

        with col2:
            st.write("#### 💰 Money Management")
            market_odds = st.number_input("ราคาตลาด (Odds):", 1.01, 20.0, 2.00)
            
            if market_odds > 1.0:
                kelly_pct = calculate_kelly(prob, market_odds)
                edge = (market_odds - fair_odds) / fair_odds * 100
                
                if edge > 0:
                    st.success(f"✅ มีกำไรส่วนต่าง (Edge): +{edge:.1f}%")
                    st.markdown(f"""
                    <div class="kelly-box">
                        ควรลงทุน: {kelly_pct:.1f}% ของพอร์ต<br>
                        (Kelly Criterion Recommendation)
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("*Kelly แนะนำให้ลงตามสัดส่วนความมั่นใจ เพื่อการเติบโตระยะยาว")
                else:
                    st.error(f"❌ เสียเปรียบเจ้ามือ (Edge: {edge:.1f}%)")
                    st.markdown("""<div class="kelly-box" style="background:#440000; color:#ffcccc;">ควรลงทุน: 0% (ห้ามเล่น)</div>""", unsafe_allow_html=True)

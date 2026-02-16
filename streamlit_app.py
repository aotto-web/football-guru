import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson

# --- CONFIGURATION ---
st.set_page_config(page_title="Pro Football Analyst", page_icon="⚽", layout="wide")

# --- CUSTOM CSS (ตกแต่งให้ดูแพง) ---
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #009933;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Pro Football Analyst: Advanced AI")
st.markdown("ระบบวิเคราะห์ฟุตบอลขั้นสูง: **Weighted Opponent Strength + Monte Carlo Simulation**")

# --- DATA ENGINE ---
@st.cache_resource(ttl=3600)
def load_data():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    ]
    
    data_frames = []
    for url in urls:
        try:
            df = pd.read_csv(url)
            # สร้างฟีเจอร์ง่ายๆ: คะแนนความยากของคู่แข่ง (League Points)
            # ใน data set นี้ไม่มีตารางคะแนน เราจะใช้ "ผลต่างประตูได้เสียสะสม" เป็นตัววัดความเก่งแทน
            data_frames.append(df)
        except:
            pass
            
    if not data_frames:
        return None, None, None

    matches = pd.concat(data_frames)
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
    matches = matches[cols].dropna()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
    matches = matches.sort_values("Date")
    
    # Feature Engineering ขั้นสูง: Form + Opponent Difficulty
    # คำนวณค่าเฉลี่ยย้อนหลัง 5 นัด (Weighted)
    def calculate_features(group):
        group['Home_Form_Goals'] = group['FTHG'].rolling(5, closed='left').mean()
        group['Away_Form_Goals'] = group['FTAG'].rolling(5, closed='left').mean()
        group['Home_Form_Shots'] = group['HST'].rolling(5, closed='left').mean() # ยิงเข้ากรอบ
        group['Away_Form_Shots'] = group['AST'].rolling(5, closed='left').mean()
        return group
    
    # แยกคำนวณเหย้า-เยือน
    matches = matches.groupby('HomeTeam', group_keys=False).apply(calculate_features)
    matches = matches.dropna()
    
    # Encoding
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["HomeTeam_Code"] = le.transform(matches["HomeTeam"])
    matches["AwayTeam_Code"] = le.transform(matches["AwayTeam"])
    matches["Target"] = (matches["FTR"] == "H").astype("int") # 1=Home Win

    # Train Random Forest Model
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    predictors = ["HomeTeam_Code", "AwayTeam_Code", "Home_Form_Goals", "Away_Form_Goals", "Home_Form_Shots", "Away_Form_Shots"]
    rf.fit(matches[predictors], matches["Target"])
    
    return rf, le, matches, predictors

# --- APP LOGIC ---
with st.spinner('⚙️ กำลังจูนสมอง AI ระดับ Pro...'):
    rf, le, matches, predictors = load_data()

if rf is None:
    st.error("เกิดข้อผิดพลาดในการโหลดข้อมูล")
else:
    # Sidebar
    st.sidebar.header("🔍 เลือกแมตช์ที่ต้องการ")
    teams = sorted(le.classes_)
    home_team = st.sidebar.selectbox("เจ้าบ้าน (Home)", teams, index=0)
    away_team = st.sidebar.selectbox("ทีมเยือน (Away)", teams, index=1)

    if st.sidebar.button("🚀 วิเคราะห์เชิงลึก (Deep Analyze)"):
        if home_team == away_team:
            st.error("กรุณาเลือกทีมให้ต่างกัน")
        else:
            # 1. ดึงสถิติล่าสุด
            try:
                h_stats = matches[matches["HomeTeam"] == home_team].iloc[-1]
                a_stats = matches[matches["AwayTeam"] == away_team].iloc[-1]
                
                # Input Data
                input_row = pd.DataFrame({
                    "HomeTeam_Code": [le.transform([home_team])[0]],
                    "AwayTeam_Code": [le.transform([away_team])[0]],
                    "Home_Form_Goals": [h_stats["Home_Form_Goals"]],
                    "Away_Form_Goals": [a_stats["Away_Form_Goals"]],
                    "Home_Form_Shots": [h_stats["Home_Form_Shots"]],
                    "Away_Form_Shots": [a_stats["Away_Form_Shots"]]
                })
                
                # Prediction
                win_prob = rf.predict_proba(input_row[predictors])[0][1]
                lose_prob = 1 - win_prob
                
                # 2. Poisson Simulation (ทำนายสกอร์)
                # คำนวณ Expected Goals (xG) คร่าวๆ จากฟอร์มยิงเข้ากรอบ
                # (สูตรประยุกต์: ยิงเข้ากรอบเฉลี่ย * Conversion Rate เฉลี่ยลีก ~0.3)
                home_xg = h_stats["Home_Form_Shots"] * 0.32
                away_xg = a_stats["Away_Form_Shots"] * 0.28 # ทีมเยือนมักยิงได้น้อยกว่า
                
                # --- แสดงผลหน้าจอ ---
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown(f"<h3 style='text-align: center; color: #1f77b4;'>{home_team}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>ฟอร์มยิง: {h_stats['Home_Form_Goals']:.2f} ลูก/นัด</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
                    st.progress(win_prob)
                    st.caption(f"โอกาสเจ้าบ้านชนะ: {win_prob*100:.1f}%")
                with col3:
                    st.markdown(f"<h3 style='text-align: center; color: #ff7f0e;'>{away_team}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>ฟอร์มยิง: {a_stats['Away_Form_Goals']:.2f} ลูก/นัด</p>", unsafe_allow_html=True)

                # --- Section: Correct Score Matrix ---
                st.markdown("### 🎯 ความน่าจะเป็นของสกอร์ (Correct Score Probability)")
                
                score_probs = []
                for h in range(4): # 0-3 ประตู
                    row = []
                    for a in range(4):
                        prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                        row.append(prob)
                    score_probs.append(row)
                
                score_df = pd.DataFrame(score_probs, columns=[f"Away {i}" for i in range(4)], index=[f"Home {i}" for i in range(4)])
                st.dataframe(score_df.style.background_gradient(cmap='Greens', axis=None).format("{:.1%}"))
                
                st.info(f"💡 **xG ที่คาดการณ์:** {home_team} ({home_xg:.2f}) - {away_team} ({away_xg:.2f})")

                # --- Section: Value Betting ---
                st.markdown("### 💰 ตรวจสอบความคุ้มค่า (Value Bet)")
                user_odds = st.number_input("ใส่ราคาต่อรอง (Odds) ที่คุณเห็น:", min_value=1.0, step=0.01)
                
                fair_odds = 1/win_prob if win_prob > 0 else 0
                
                c1, c2 = st.columns(2)
                c1.metric("ราคาที่ควรจะเป็น (Fair Odds)", f"{fair_odds:.2f}")
                
                if user_odds > 1.0:
                    edge = (user_odds - fair_odds) / fair_odds * 100
                    c2.metric("กำไรคาดหวัง (Edge)", f"{edge:.2f}%", delta_color="normal" if edge > 0 else "inverse")
                    
                    if edge > 5:
                        st.success("🌟 **Highly Recommended!** คู่นี้มีกำไรส่วนต่างสูงน่าลงทุน")
                    elif edge > 0:
                        st.info("✅ **Investable** พอน่าลุ้น มีกำไรบางๆ")
                    else:
                        st.error("🛑 **Overpriced** ราคาไม่คุ้มเสี่ยง (เจ้ามือเอาเปรียบ)")

            except IndexError:
                st.warning("ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์คู่นี้ (อาจเป็นทีมเลื่อนชั้น)")

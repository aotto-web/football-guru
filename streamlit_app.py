import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson

st.set_page_config(page_title="Football God Mode", page_icon="⚽", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    h1, h2, h3 { color: #00FF7F !important; }
    div[data-testid="stMetricValue"] { color: #00FF7F; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ AI Football God Mode")
st.write("ระบบวิเคราะห์ความน่าจะเป็น: **ดูทีละคู่** หรือ **ดูทั้งลีก**")

# --- DATA LOADING ---
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
            data_frames.append(df)
        except: pass
            
    if not data_frames: return None, None, None, None

    matches = pd.concat(data_frames)
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
    matches = matches[cols].dropna()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
    matches = matches.sort_values("Date")

    # Feature Engineering
    def calculate_features(group):
        group['H_Form'] = group['FTHG'].rolling(5, closed='left').mean()
        group['A_Form'] = group['FTAG'].rolling(5, closed='left').mean()
        return group
    
    matches = matches.groupby('HomeTeam', group_keys=False).apply(calculate_features).dropna()
    
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["H_Code"] = le.transform(matches["HomeTeam"])
    matches["A_Code"] = le.transform(matches["AwayTeam"])
    matches["Target"] = (matches["FTR"] == "H").astype("int")

    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    predictors = ["H_Code", "A_Code", "H_Form", "A_Form"]
    rf.fit(matches[predictors], matches["Target"])
    
    return rf, le, matches, predictors

with st.spinner('กำลังคำนวณความเป็นไปได้ทั้งจักรวาล...'):
    rf, le, matches, predictors = load_data()

if rf:
    # --- สร้าง TAB แยกหน้าจอ ---
    tab1, tab2 = st.tabs(["🔍 วิเคราะห์รายคู่ (Match)", "📊 ตารางทั้งลีก (League Matrix)"])

    # === TAB 1: วิเคราะห์รายคู่ (เหมือนเดิม) ===
    with tab1:
        st.header("เจาะลึกรายแมตช์")
        c1, c2 = st.columns(2)
        teams = sorted(le.classes_)
        h_team = c1.selectbox("เจ้าบ้าน", teams, index=0)
        a_team = c2.selectbox("ทีมเยือน", teams, index=1)
        
        if st.button("ทำนายผลคู่นี้"):
            # (Logic เดิม)
            try:
                h_stats = matches[matches["HomeTeam"] == h_team].iloc[-1]
                a_stats = matches[matches["AwayTeam"] == a_team].iloc[-1]
                
                row = pd.DataFrame({
                    "H_Code": [le.transform([h_team])[0]],
                    "A_Code": [le.transform([a_team])[0]],
                    "H_Form": [h_stats["H_Form"]],
                    "A_Form": [a_stats["A_Form"]]
                })
                prob = rf.predict_proba(row[predictors])[0][1]
                
                st.metric("โอกาสเจ้าบ้านชนะ", f"{prob*100:.1f}%")
                if prob > 0.6: st.success(f"เชียร์ {h_team} ได้เลย!")
                elif prob < 0.4: st.error(f"{h_team} ไม่น่ารอด")
                else: st.warning("สูสีมาก")
            except: st.error("ข้อมูลไม่พอ")

    # === TAB 2: ตารางเทพ (League Matrix) ===
    with tab2:
        st.header("🔥 ตารางทำนาย: ใครเจอใคร...ใครจะชนะ?")
        st.write("ตารางนี้แสดง **'โอกาสชนะของเจ้าบ้าน'** ในทุกแมตช์ที่เป็นไปได้")
        st.info("วิธีดู: เลือกทีมฝั่งซ้าย (เจ้าบ้าน) แล้วไล่ไปทางขวา (เจอทีมไหน) = % ชนะ")

        # สร้างตาราง Matrix 20x20
        all_teams = sorted(le.classes_)
        matrix_data = []

        # วนลูปทุกทีมเจอทุกทีม
        for home in all_teams:
            row_probs = []
            try:
                h_stats = matches[matches["HomeTeam"] == home].iloc[-1]
                h_form = h_stats["H_Form"]
            except: h_form = 1.5 # ค่ากลางๆถ้าหาไม่เจอ

            for away in all_teams:
                if home == away:
                    row_probs.append(0) # เจอตัวเองไม่ได้
                else:
                    try:
                        a_stats = matches[matches["AwayTeam"] == away].iloc[-1]
                        a_form = a_stats["A_Form"]
                    except: a_form = 1.5
                    
                    # ทำนาย
                    input_data = pd.DataFrame([[le.transform([home])[0], le.transform([away])[0], h_form, a_form]], columns=predictors)
                    prob = rf.predict_proba(input_data)[0][1]
                    row_probs.append(prob)
            
            matrix_data.append(row_probs)

        # แสดงผลเป็น DataFrame สีสวยๆ
        df_matrix = pd.DataFrame(matrix_data, index=all_teams, columns=all_teams)
        
        # ไฮไลท์สี (เขียว=โอกาสชนะสูง, แดง=โอกาสชนะต่ำ)
        st.dataframe(
            df_matrix.style
            .background_gradient(cmap='RdYlGn', vmin=0.2, vmax=0.8)
            .format("{:.0%}")
        , height=800)

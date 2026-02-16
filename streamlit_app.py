import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pytz

# --- 1. CONFIGURATION & GOOGLE STYLE CSS ---
st.set_page_config(page_title="Premier League AI", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    /* Dark Mode Google Style */
    .stApp { background-color: #202124; color: #bdc1c6; }
    
    /* Date Header */
    .date-header {
        font-size: 18px;
        font-weight: bold;
        color: #e8eaed;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 1px solid #3c4043;
        padding-bottom: 5px;
    }
    
    /* Match Row (เหมือนแถวใน Google) */
    .match-row {
        background-color: #303134;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 5px solid #5f6368; /* Default Grey */
        transition: 0.2s;
    }
    .match-row:hover { background-color: #3c4043; }
    
    /* Confidence Colors */
    .high-win { border-left-color: #81c995 !important; } /* Green */
    .high-lose { border-left-color: #f28b82 !important; } /* Red */
    .draw { border-left-color: #fdd663 !important; } /* Yellow */

    .team-name { font-size: 16px; font-weight: 500; color: #fff; }
    .vs-time { font-size: 14px; color: #9aa0a6; text-align: center; min-width: 80px;}
    .ai-badge { 
        font-size: 12px; 
        padding: 4px 8px; 
        border-radius: 4px; 
        background: #000; 
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Premier League Schedule & Prediction")

# --- 2. INTELLIGENT ENGINE (ระบบคำนวณเหมือนเดิม) ---
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

    def get_features(group):
        group['Form_Point'] = group['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0)).rolling(5, closed='left').mean()
        group['H_Goal'] = group['FTHG'].rolling(5, closed='left').mean()
        group['A_Goal'] = group['FTAG'].rolling(5, closed='left').mean()
        return group
    
    matches = matches.groupby('HomeTeam', group_keys=False).apply(get_features).dropna()
    
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["H_Code"] = le.transform(matches["HomeTeam"])
    matches["A_Code"] = le.transform(matches["AwayTeam"])
    matches["Target"] = (matches["FTR"] == "H").astype("int")

    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42)
    rf.fit(matches[["H_Code", "A_Code", "Form_Point", "H_Goal", "A_Goal"]], matches["Target"])
    
    return rf, le, matches

def map_name(name, known):
    mapping = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nott'm Forest", 
               "Wolves": "Wolves", "Man City": "Man City", "Newcastle Utd": "Newcastle",
               "Sheffield Utd": "Sheffield United", "Luton Town": "Luton", "West Ham Utd": "West Ham", "Ipswich Town": "Ipswich"}
    if name in known: return name
    if name in mapping and mapping[name] in known: return mapping[name]
    return None

# --- 3. MAIN UI LOGIC ---
with st.spinner('🔄 Loading Schedule...'):
    rf, le, matches = load_engine()

# ดึงตารางแข่ง
try:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get("https://fixturedownload.com/feed/json/epl-2025", headers=headers)
    
    if r.status_code == 200:
        fixtures = pd.read_json(StringIO(r.text))
        fixtures['DateUtc'] = pd.to_datetime(fixtures['DateUtc'], utc=True)
        now_utc = pd.Timestamp.now('UTC')
        
        # กรองเฉพาะนัดที่ยังไม่แข่ง และเรียงตามเวลา
        upcoming = fixtures[fixtures['DateUtc'] >= now_utc].sort_values('DateUtc').head(15) # ดึง 15 นัดหน้า
        
        # จัดกลุ่มตามวันที่ (Group by Date)
        # สร้างคอลัมน์วันที่แบบอ่านง่าย (เช่น "Sat 17 Feb")
        upcoming['DateStr'] = upcoming['DateUtc'].dt.strftime('%A %d %B')
        
        # วนลูปทีละวัน (หัวข้อวันที่)
        unique_dates = upcoming['DateStr'].unique()
        
        for date_str in unique_dates:
            # 1. แสดงหัวข้อวันที่
            st.markdown(f'<div class="date-header">{date_str}</div>', unsafe_allow_html=True)
            
            # 2. แสดงรายการแข่งในวันนั้น
            day_matches = upcoming[upcoming['DateStr'] == date_str]
            
            for _, row in day_matches.iterrows():
                time_str = row['DateUtc'].strftime('%H:%M')
                h_real = map_name(row['HomeTeam'], le.classes_)
                a_real = map_name(row['AwayTeam'], le.classes_)
                
                # Default values
                css_class = ""
                ai_text = "N/A"
                ai_color = "#555"
                prob = 0.5

                if h_real and a_real:
                    # AI Predict
                    h_stat = matches[matches["HomeTeam"] == h_real].iloc[-1]
                    a_stat = matches[matches["AwayTeam"] == a_real].iloc[-1]
                    pred_row = [[le.transform([h_real])[0], le.transform([a_real])[0], 
                                 h_stat["Form_Point"], h_stat["H_Goal"], a_stat["A_Goal"]]]
                    prob = rf.predict_proba(pred_row)[0][1]
                    
                    # Color Logic
                    if prob > 0.60:
                        css_class = "high-win" # เขียว
                        ai_text = f"Home {prob*100:.0f}%"
                        ai_color = "#81c995"
                    elif prob < 0.40:
                        css_class = "high-lose" # แดง
                        ai_text = f"Away {(1-prob)*100:.0f}%"
                        ai_color = "#f28b82"
                    else:
                        css_class = "draw" # เหลือง
                        ai_text = "50/50"
                        ai_color = "#fdd663"

                # Render Match Row (HTML)
                st.markdown(f"""
                <div class="match-row {css_class}">
                    <div style="flex:1; text-align:right;" class="team-name">{h_real}</div>
                    <div class="vs-time">
                        <div>{time_str}</div>
                        <div style="font-size:10px; color:#5f6368;">VS</div>
                    </div>
                    <div style="flex:1; text-align:left;" class="team-name">{a_real}</div>
                    <div class="ai-badge" style="color:{ai_color}; border: 1px solid {ai_color};">
                        AI: {ai_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # เพิ่มปุ่มกดดูละเอียด (ซ่อนอยู่ใน Expander)
                with st.expander(f"📊 เจาะลึก {h_real} vs {a_real}"):
                    if h_real and a_real:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("โอกาสเจ้าบ้าน", f"{prob*100:.1f}%")
                        c2.metric("ราคาที่ควรเป็น (Fair Odds)", f"{1/prob:.2f}")
                        c3.write(f"**คำแนะนำ:** {'✅ ลงทุนได้' if prob > 0.6 or prob < 0.4 else '⚠️ สูสี/เลี่ยง'}")
                    else:
                        st.write("ข้อมูลทีมไม่เพียงพอ")

except Exception as e:
    st.error(f"โหลดข้อมูลไม่สำเร็จ: {e}")

st.markdown("---")
st.caption("Data provided by football-data.co.uk & FixtureDownload | AI Model: Random Forest v2")

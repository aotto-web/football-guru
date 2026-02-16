import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. ตั้งค่าหน้าเว็บให้ดูง่ายที่สุด ---
st.set_page_config(page_title="Football Simple", page_icon="⚽", layout="centered") # เปลี่ยนเป็น centered ให้เหมือนมือถือ

st.markdown("""
<style>
    /* พื้นหลังสีดำสบายตา */
    .stApp { background-color: #121212; color: #ffffff; }
    
    /* กล่องอธิบายสี (Legend) */
    .legend-box {
        background: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        display: flex;
        justify-content: space-around;
        margin-bottom: 20px;
        font-size: 0.9em;
    }
    
    /* แถวรายการบอล (Card) */
    .match-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 8px solid #555; /* แถบสีจะอยู่ตรงนี้ */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* สีของแถบ (Color Strip) */
    .strip-green { border-left-color: #00e676 !important; } /* เขียวสว่าง */
    .strip-red { border-left-color: #ff5252 !important; }   /* แดงสว่าง */
    .strip-yellow { border-left-color: #ffea00 !important; } /* เหลือง */

    .team { font-size: 16px; font-weight: bold; }
    .vs { color: #888; font-size: 12px; margin: 0 10px; }
    .time { color: #aaa; font-size: 12px; display: block; margin-bottom: 4px;}
    .percent { font-size: 14px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ ตารางบอลสีนำโชค")

# --- 2. ส่วนอธิบายสี (Legend) ---
st.markdown("""
<div class="legend-box">
    <div style="color:#00e676;">🟢 เจ้าบ้านนอนมา</div>
    <div style="color:#ffea00;">🟡 สูสี/ออกเสมอ</div>
    <div style="color:#ff5252;">🔴 ทีมเยือนบุกชนะ</div>
</div>
<div style="font-size:0.8em; color:#666; text-align:center; margin-bottom:20px;">
    *คำเตือน: AI คำนวณจากสถิติย้อนหลัง ไม่รวมข่าวนักเตะเจ็บรายวัน
</div>
""", unsafe_allow_html=True)

# --- 3. ระบบสมองกล (AI Engine) ---
@st.cache_resource(ttl=3600)
def load_ai():
    # ดึงข้อมูลย้อนหลัง 3 ปีมาเทรน
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    ]
    dfs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                dfs.append(pd.read_csv(StringIO(r.text)))
        except: pass
    
    if not dfs: return None, None, None

    matches = pd.concat(dfs)
    matches = matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
    matches = matches.sort_values("Date")
    
    # สูตรคำนวณฟอร์มง่ายๆ (ชนะ=3, เสมอ=1)
    def get_form(g):
        g['Points'] = g['FTR'].apply(lambda x: 3 if x=='H' else (1 if x=='D' else 0))
        g['Form'] = g['Points'].rolling(5, closed='left').mean()
        return g
    
    matches = matches.groupby('HomeTeam', group_keys=False).apply(get_form).fillna(1)
    
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["H_Code"] = le.transform(matches["HomeTeam"])
    matches["A_Code"] = le.transform(matches["AwayTeam"])
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    rf.fit(matches[["H_Code", "A_Code", "Form"]], (matches["FTR"] == "H").astype(int))
    
    return rf, le, matches

def map_name(name, known):
    # แปลชื่อทีมให้ตรงกัน
    mapping = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nott'm Forest", 
               "Wolves": "Wolves", "Man City": "Man City", "Newcastle Utd": "Newcastle",
               "Sheffield Utd": "Sheffield United", "Luton Town": "Luton", "West Ham Utd": "West Ham",
               "Ipswich Town": "Ipswich"} # เพิ่ม Ipswich
    if name in known: return name
    if name in mapping and mapping[name] in known: return mapping[name]
    return None

# --- 4. แสดงผลหน้าจอ ---
rf, le, matches = load_ai()

if rf:
    try:
        # ดึงตารางแข่งจริง
        r = requests.get("https://fixturedownload.com/feed/json/epl-2025", headers={"User-Agent": "Mozilla/5.0"})
        fixtures = pd.read_json(StringIO(r.text))
        fixtures['DateUtc'] = pd.to_datetime(fixtures['DateUtc'], utc=True)
        
        # คัดเฉพาะบอลที่ยังไม่เตะ 10 คู่ถัดไป
        upcoming = fixtures[fixtures['DateUtc'] >= pd.Timestamp.now('UTC')].sort_values('DateUtc').head(10)
        
        # วนลูปแสดงผล
        current_date = ""
        for _, row in upcoming.iterrows():
            date_str = row['DateUtc'].strftime("%d %b (วัน%A)")
            time_str = row['DateUtc'].strftime("%H:%M")
            
            # หัวข้อวันที่ (ถ้าเปลี่ยนวันให้ขึ้นหัวข้อใหม่)
            if current_date != date_str:
                st.markdown(f"<h4 style='color:#888; margin-top:20px; border-bottom:1px solid #333;'>📅 {date_str}</h4>", unsafe_allow_html=True)
                current_date = date_str
            
            h = map_name(row['HomeTeam'], le.classes_)
            a = map_name(row['AwayTeam'], le.classes_)
            
            if h and a:
                # คำนวณ
                h_stat = matches[matches["HomeTeam"] == h].iloc[-1]
                prob = rf.predict_proba([[le.transform([h])[0], le.transform([a])[0], h_stat["Form"]]])[0][1]
                
                # Logic สี (ตามที่คุณขอเป๊ะๆ)
                if prob > 0.60:
                    color_class = "strip-green" # เขียว
                    percent_text = f"<span style='color:#00e676;'>{prob*100:.0f}% เจ้าบ้าน</span>"
                elif prob < 0.40:
                    color_class = "strip-red"   # แดง
                    percent_text = f"<span style='color:#ff5252;'>{(1-prob)*100:.0f}% ทีมเยือน</span>"
                else:
                    color_class = "strip-yellow" # เหลือง
                    percent_text = "<span style='color:#ffea00;'>สูสี 50/50</span>"
                
                # แสดงการ์ด
                st.markdown(f"""
                <div class="match-card {color_class}">
                    <div>
                        <span class="time">{time_str}</span>
                        <div class="team">{h}</div>
                        <div class="team">{a}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="percent">{percent_text}</div>
                        <div style="font-size:10px; color:#666;">AI Confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption(f"⚠️ ข้ามคู่ {row['HomeTeam']} vs {row['AwayTeam']} (ชื่อทีมไม่ตรง)")
                
    except Exception as e:
        st.error(f"โหลดข้อมูลไม่ได้: {e}")

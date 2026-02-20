import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson
from datetime import datetime

# --- 1. SETTINGS & STYLES ---
st.set_page_config(page_title="PREMIER LEAGUE GOD-MODE AI", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1d2129; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .match-card {
        background: linear-gradient(135deg, #1d2129 0%, #111418 100%);
        padding: 20px; border-radius: 15px; border-left: 6px solid #3d195d;
        margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .pl-header { color: #3d195d; background: #00ff88; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
    .status-win { color: #00ff88; font-weight: bold; }
    .status-fail { color: #ff006e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE (Premier League Focus) ---
@st.cache_data(ttl=3600)
def fetch_pl_data():
    # ดึงข้อมูลจาก Football-Data.co.uk (E0 = Premier League)
    url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    try:
        r = requests.get(url)
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        return df
    except:
        return pd.DataFrame()

# --- 3. AI CORE LOGIC (Poisson + Strength Analysis) ---
def get_ultimate_prediction(home, away, df):
    # คำนวณค่าเฉลี่ยประตูทั้งลีค
    avg_home_g = df['FTHG'].mean()
    avg_away_g = df['FTAG'].mean()

    # คำนวณ Attack/Defense Strength (10 นัดล่าสุด)
    h_data = df[df['HomeTeam'] == home].tail(10)
    a_data = df[df['AwayTeam'] == away].tail(10)
    
    h_att = h_data['FTHG'].mean() / avg_home_g
    h_def = h_data['FTAG'].mean() / avg_away_g
    a_att = a_data['FTAG'].mean() / avg_away_g
    a_def = a_data['FTHG'].mean() / avg_home_g

    # Expected Goals (xG)
    exp_h = h_att * a_def * avg_home_g
    exp_a = a_att * h_def * avg_away_g

    # หาผลสกอร์ที่มีความน่าจะเป็นสูงสุด (Most Likely Score)
    probs_h = [poisson.pmf(i, exp_h) for i in range(6)]
    probs_a = [poisson.pmf(i, exp_a) for i in range(6)]
    
    pred_h = np.argmax(probs_h)
    pred_a = np.argmax(probs_a)
    
    # ความมั่นใจ (Confidence)
    confidence = (max(probs_h) * max(probs_a)) * 100
    
    return pred_h, pred_a, exp_h, exp_a, confidence

# --- 4. MAIN INTERFACE ---
st.markdown("<h1 style='text-align: center; color: #00ff88;'>🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE GOD-MODE</h1>", unsafe_allow_html=True)
st.markdown("<div class='pl-header'>วิเคราะห์เจาะลึกทุกลมหายใจพรีเมียร์ลีก</div>", unsafe_allow_html=True)

df_pl = fetch_pl_data()

if not df_pl.empty:
    tab1, tab2, tab3 = st.tabs(["🎯 ทายผลนัดถัดไป", "📉 ตรวจสอบความแม่นยำ (Error)", "📊 สถิติลีค"])

    with tab1:
        st.subheader("วิเคราะห์คู่บิ๊กแมตช์")
        teams = sorted(df_pl['HomeTeam'].unique())
        c1, c2 = st.columns(2)
        with c1: home_sel = st.selectbox("เจ้าบ้าน (Home Team)", teams, index=0)
        with c2: away_sel = st.selectbox("ทีมเยือน (Away Team)", teams, index=1)

        if home_sel == away_sel:
            st.warning("กรุณาเลือกทีมที่แตกต่างกัน")
        else:
            p_h, p_a, x_h, x_a, conf = get_ultimate_prediction(home_sel, away_sel, df_pl)
            
            # การแสดงผลแบบการ์ด "มหาเทพ"
            st.markdown(f"""
            <div class='match-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='text-align: center; flex: 1;'><h3>{home_sel}</h3><p>xG: {x_h:.2f}</p></div>
                    <div style='text-align: center; flex: 1;'><h1 style='color: #00ff88;'>{p_h} - {p_a}</h1></div>
                    <div style='text-align: center; flex: 1;'><h3>{away_sel}</h3><p>xG: {x_a:.2f}</p></div>
                </div>
                <div style='text-align: center; margin-top: 20px;'>
                    <span style='background: #3d195d; padding: 5px 20px; border-radius: 20px;'>
                        ความมั่นใจ AI: {conf:.1f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("เช็คผลงาน AI ย้อนหลัง (Error Tracker)")
        recent_5 = df_pl.tail(10).iloc[::-1] # ดู 10 นัดล่าสุด
        
        comparison = []
        for _, row in recent_5.iterrows():
            # ใช้ข้อมูลก่อนหน้าวันแข่งจริง (ในที่นี้จำลองเพื่อโชว์ผล)
            ph, pa, _, _, _ = get_ultimate_prediction(row['HomeTeam'], row['AwayTeam'], df_pl)
            
            actual = f"{row['FTHG']}-{row['FTAG']}"
            pred = f"{ph}-{pa}"
            diff = abs(ph - row['FTHG']) + abs(pa - row['FTAG'])
            
            if diff == 0: status = "✅ แม่นยำที่สุด"
            elif diff <= 1: status = "🟡 ใกล้เคียง"
            else: status = "❌ พลาด"
            
            comparison.append({
                "วันที่": row['Date'].strftime('%d/%b'),
                "คู่แข่งขัน": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                "ผลจริง": actual,
                "AI ทาย": pred,
                "สถานะ": status
            })
        
        st.table(pd.DataFrame(comparison))

    with tab3:
        # แสดงตารางคะแนนจำลองจากสถิติ
        st.subheader("อันดับความแข็งแกร่ง (AI Power Ranking)")
        ranking = df_pl.groupby('HomeTeam')[['FTHG', 'FTAG']].mean().sort_values('FTHG', ascending=False)
        st.bar_chart(ranking['FTHG'])

# --- 5. PHP & DATABASE INTEGRATION HINT ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=100)
    st.markdown("### 🛠️ Developer Mode")
    if st.checkbox("Show JSON for PHP API"):
        # ตัวอย่าง JSON ที่ PHP จะได้รับ
        sample_json = {
            "match": f"{home_sel} vs {away_sel}",
            "prediction": {"home": int(p_h), "away": int(p_a)},
            "xg": {"home": float(x_h), "away": float(x_a)},
            "timestamp": datetime.now().isoformat()
        }
        st.json(sample_json)
    
    st.info("คำแนะนำ: รันสคริปต์นี้ใน Background แล้วส่งข้อมูลเข้า MySQL เพื่อให้หน้าเว็บ PHP ดึงข้อมูลไปโชว์แบบ Real-time")

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson
from datetime import datetime

# --- 1. SETUP & STYLE ---
st.set_page_config(page_title="PL GOD-MODE PREDICTOR", layout="wide")

st.markdown("""
<style>
    .match-row {
        display: grid; grid-template-columns: 120px 1.5fr 1fr 1.5fr 150px;
        background: #1d2129; padding: 12px; margin-bottom: 5px;
        border-radius: 8px; align-items: center; border-left: 5px solid #3d195d;
    }
    .date-text { color: #888; font-size: 13px; }
    .team-name { font-weight: bold; font-size: 16px; }
    .score-pred { color: #00ff88; font-size: 18px; font-weight: bold; text-align: center; }
    .conf-tag { background: #3d195d; color: white; padding: 2px 10px; border-radius: 15px; font-size: 11px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_data():
    # ข้อมูลสถิติย้อนหลังเพื่อ Train AI
    url_stats = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    # ข้อมูลตารางแข่งล่วงหน้า (Fixtures)
    url_fixtures = "https://fixturedownload.com/feed/json/epl-2025"
    
    try:
        stats_df = pd.read_csv(url_stats)
        fix_res = requests.get(url_fixtures)
        fix_df = pd.DataFrame(fix_res.json())
        return stats_df, fix_df
    except:
        return pd.DataFrame(), pd.DataFrame()

# --- 3. AI PREDICTION LOGIC ---
def predict_match(home, away, stats_df):
    if home not in stats_df['HomeTeam'].values or away not in stats_df['AwayTeam'].values:
        return 0, 0, 0 # กรณีทีมใหม่ไม่มีข้อมูล
        
    avg_h_g = stats_df['FTHG'].mean()
    avg_a_g = stats_df['FTAG'].mean()
    
    h_att = stats_df[stats_df['HomeTeam'] == home]['FTHG'].mean() / avg_h_g
    h_def = stats_df[stats_df['HomeTeam'] == home]['FTAG'].mean() / avg_a_g
    a_att = stats_df[stats_df['AwayTeam'] == away]['FTAG'].mean() / avg_a_g
    a_def = stats_df[stats_df['AwayTeam'] == away]['FTHG'].mean() / avg_h_g
    
    exp_h = h_att * a_def * avg_h_g
    exp_a = a_att * h_def * avg_a_goals = avg_a_g # simplified
    
    p_h = np.argmax([poisson.pmf(i, exp_h) for i in range(6)])
    p_a = np.argmax([poisson.pmf(i, exp_a) for i in range(6)])
    conf = (max([poisson.pmf(i, exp_h) for i in range(6)]) * max([poisson.pmf(i, exp_a) for i in range(6)])) * 100
    
    return p_h, p_a, conf

# --- 4. TEXT LOGGING SYSTEM ---
def save_to_text(log_entry):
    with open("predictions_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- 5. MAIN UI ---
st.title("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League: Full Schedule Prediction")
stats_df, fix_df = load_data()

if not fix_df.empty:
    # กรองเฉพาะคู่ที่ยังไม่เตะ
    fix_df['DateUtc'] = pd.to_datetime(fix_df['DateUtc'])
    upcoming = fix_df[fix_df['DateUtc'] >= datetime.utcnow()].sort_values('DateUtc').head(20)

    if st.button("🚀 วิเคราะห์ทุกล่วงหน้าและบันทึกลง Text File"):
        log_content = f"--- Prediction Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        st.success("บันทึกข้อมูลการทายผลลงไฟล์ predictions_log.txt เรียบร้อยแล้ว!")
        
        for _, row in upcoming.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            p_h, p_a, conf = predict_match(h, a, stats_df)
            date_str = row['DateUtc'].strftime('%d/%m %H:%M')
            
            # เก็บข้อมูลลง Log
            entry = f"[{date_str}] {h} {p_h}-{p_a} {a} (Conf: {conf:.1f}%)"
            save_to_text(entry)

    # แสดงผลบนหน้าเว็บ
    st.markdown("### 📅 ตารางการคาดการณ์ล่วงหน้า")
    for _, row in upcoming.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        p_h, p_a, conf = predict_match(h, a, stats_df)
        
        st.markdown(f"""
        <div class="match-row">
            <div class="date-text">{row['DateUtc'].strftime('%d %b %H:%M')}</div>
            <div class="team-name" style="text-align:right;">{h}</div>
            <div class="score-pred">{p_h} - {p_a}</div>
            <div class="team-name" style="text-align:left;">{a}</div>
            <div class="conf-tag">มั่นใจ {conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# --- 6. DOWNLOAD SECTION ---
st.divider()
try:
    with open("predictions_log.txt", "rb") as file:
        st.download_button(
            label="📥 ดาวน์โหลดไฟล์ข้อมูลการทายผล (.txt)",
            data=file,
            file_name="football_predictions.txt",
            mime="text/plain"
        )
except:
    st.info("กดปุ่ม 'วิเคราะห์' ด้านบนเพื่อสร้างไฟล์ข้อมูล")

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="PL Auto-Predictor 2026", layout="wide")
st.title("🏆 Premier League Predictor (No-Block Version)")

# --- 1. ดึงตารางคะแนนสดจาก Wikipedia (เสถียรและไม่ค่อยบล็อก) ---
@st.cache_data(ttl=3600)
def get_live_stats_wiki():
    try:
        # Wikipedia มีตารางคะแนนพรีเมียร์ลีกที่อัปเดตไวมาก
        url = "https://en.wikipedia.org/wiki/2025%E2%80%9326_Premier_League"
        tables = pd.read_html(url)
        
        # ค้นหาตารางที่มีคำว่า 'Points' หรือ 'Pos'
        df = None
        for t in tables:
            if 'Pts' in t.columns and 'GF' in t.columns:
                df = t
                break
        
        if df is not None:
            # เลือกคอลัมน์: ทีม, แข่ง(Pld), ได้(GF), เสีย(GA), แต้ม(Pts)
            df = df[['Team', 'Pld', 'GF', 'GA', 'Pts']]
            df.columns = ['Team', 'M', 'Scored', 'Conceded', 'Pts']
            
            # ล้างชื่อทีม (บางครั้งมีหมายเหตุ เช่น (C), (R))
            df['Team'] = df['Team'].str.replace(r'\(.*\)', '', regex=True).str.strip()
            
            # คำนวณค่าเฉลี่ยและ Strength
            avg_scored = df['Scored'].astype(float).mean()
            avg_conceded = df['Conceded'].astype(float).mean()
            
            df['Offense'] = df['Scored'].astype(float) / avg_scored
            df['Defense'] = df['Conceded'].astype(float) / avg_conceded
            
            return df, avg_scored / df['M'].astype(float).mean(), avg_conceded / df['M'].astype(float).mean()
    except Exception as e:
        st.error(f"Wikipedia Error: {e}")
    return None, 1.5, 1.3

# --- 2. ฟังก์ชันคำนวณทำนายผล ---
def predict_match(home, away, stats_df, avg_h, avg_a):
    try:
        # ค้นหาชื่อทีมแบบยืดหยุ่น (Fuzzy Match เบื้องต้น)
        h_stat = stats_df[stats_df['Team'].str.contains(home, case=False, na=False)].iloc[0]
        a_stat = stats_df[stats_df['Team'].str.contains(away, case=False, na=False)].iloc[0]
        
        exp_h = h_stat['Offense'] * a_stat['Defense'] * avg_h
        exp_a = a_stat['Offense'] * h_stat['Defense'] * avg_a
        
        h_prob = [poisson.pmf(i, exp_h) for i in range(7)]
        a_prob = [poisson.pmf(i, exp_a) for i in range(7)]
        matrix = np.outer(h_prob, a_prob)
        
        ph = np.sum(np.tril(matrix, -1))
        pd = np.sum(np.diag(matrix))
        pa = np.sum(np.triu(matrix, 1))
        hp, ap = np.unravel_index(matrix.argmax(), matrix.shape)
        
        return exp_h, exp_a, ph, pd, pa, f"{hp}-{ap}"
    except:
        return 0, 0, 0, 0, 0, "N/A"

# --- 3. ส่วนการแสดงผล ---
df_stats, avg_h, avg_a = get_live_stats_wiki()

if df_stats is not None:
    st.sidebar.success("เชื่อมต่อข้อมูลตารางคะแนนสำเร็จ!")
    st.sidebar.dataframe(df_stats[['Team', 'Pts', 'Offense', 'Defense']], hide_index=True)

    st.header("🔮 วิเคราะห์คู่แข่งขันถัดไป")
    
    # เนื่องจากโปรแกรมแข่งดึงยาก เราจะทำระบบ "เลือกคู่เอง" ที่ดึงรายชื่อทีมมาจากตารางคะแนนอัตโนมัติ
    # ทำให้รันได้ทุกคู่ "Auto" ตลอดกาล ไม่ว่าจะเป็นคู่ไหนในลีก
    team_list = sorted(df_stats['Team'].tolist())
    
    col_a, col_b = st.columns(2)
    with col_a:
        h_team = st.selectbox("เลือกเจ้าบ้าน (Home)", team_list, index=0)
    with col_b:
        a_team = st.selectbox("เลือกทีมเยือน (Away)", team_list, index=1)

    if h_team and a_team:
        xh, xa, ph, pd, pa, score = predict_match(h_team, a_team, df_stats, avg_h, avg_a)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{h_team} ชนะ", f"{ph*100:.1f}%")
        c2.metric("เสมอ", f"{pd*100:.1f}%")
        c3.metric(f"{a_team} ชนะ", f"{pa*100:.1f}%")
        
        st.subheader(f"🎯 สกอร์ที่คาดหวัง: {score}")
        st.write(f"ค่าความน่าจะเป็นเชิงสถิติ (xG): {h_team} {xh:.2f} VS {a_team} {xa:.2f}")

else:
    st.error("ไม่สามารถเข้าถึงแหล่งข้อมูลได้ในขณะนี้ โปรดลองใหม่อีกครั้ง")

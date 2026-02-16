import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Football Analyst", page_icon="⚽")

st.title("⚽ Premier League AI Predictor")
st.write("ระบบวิเคราะห์ผลบอลพรีเมียร์ลีกด้วย Machine Learning (อัปเดตข้อมูลอัตโนมัติ)")

# โหลดข้อมูลและเทรนโมเดล (Cache ไว้ 1 ชั่วโมง = 3600 วินาที)
@st.cache_resource(ttl=3600)
def load_data_and_train():
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
        except:
            pass
    
    if not data_frames:
        return None, None, None, None

    matches = pd.concat(data_frames)
    
    # เลือกคอลัมน์และจัดการข้อมูล
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
    matches = matches[cols].dropna()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
    matches["Target"] = (matches["FTR"] == "H").astype("int")

    # แปลงชื่อทีมเป็นตัวเลข
    le = LabelEncoder()
    le.fit(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    matches["HomeTeam_Code"] = le.transform(matches["HomeTeam"])
    matches["AwayTeam_Code"] = le.transform(matches["AwayTeam"])

    # คำนวณค่าเฉลี่ยย้อนหลัง 3 นัด
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("Date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        return group.dropna(subset=new_cols)

    cols_to_roll = ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]
    new_cols = [f"{c}_Rolling" for c in cols_to_roll]
    
    matches_rolling = matches.groupby("HomeTeam").apply(lambda x: rolling_averages(x, cols_to_roll, new_cols))
    matches_rolling = matches_rolling.droplevel('HomeTeam').sort_values("Date")
    
    # เทรน AI
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
    predictors = ["HomeTeam_Code", "AwayTeam_Code"] + new_cols
    rf.fit(matches_rolling[predictors], matches_rolling["Target"])
    
    return rf, le, matches_rolling, predictors

# เรียกใช้ฟังก์ชัน
with st.spinner('กำลังดึงข้อมูลล่าสุดจากอังกฤษ...'):
    rf, le, matches_rolling, predictors = load_data_and_train()

if rf is None:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ โปรดลองใหม่ภายหลัง")
else:
    # ส่วนเลือกทีม
    st.sidebar.header("เลือกคู่แข่งขัน")
    all_teams = sorted(le.classes_)
    home_team = st.sidebar.selectbox("ทีมเจ้าบ้าน", all_teams, index=0)
    away_team = st.sidebar.selectbox("ทีมเยือน", all_teams, index=1)

    if st.sidebar.button("วิเคราะห์ผล"):
        if home_team == away_team:
            st.warning("เลือกทีมซ้ำกันไม่ได้ครับ")
        else:
            try:
                # ดึงสถิติล่าสุด
                home_stats = matches_rolling[matches_rolling["HomeTeam"] == home_team].iloc[-1]
                away_stats = matches_rolling[matches_rolling["AwayTeam"] == away_team].iloc[-1]
                
                # เตรียมข้อมูลทำนาย
                input_data = pd.DataFrame({
                    "HomeTeam_Code": [le.transform([home_team])[0]],
                    "AwayTeam_Code": [le.transform([away_team])[0]],
                    "FTHG_Rolling": [home_stats["FTHG_Rolling"]],
                    "FTAG_Rolling": [home_stats["FTAG_Rolling"]],
                    "HS_Rolling": [home_stats["HS_Rolling"]],
                    "AS_Rolling": [home_stats["AS_Rolling"]],
                    "HST_Rolling": [home_stats["HST_Rolling"]],
                    "AST_Rolling": [home_stats["AST_Rolling"]]
                })
                
                # ทำนาย
                prob = rf.predict_proba(input_data[predictors])[0][1]
                
                # แสดงผล
                st.subheader(f"ผลวิเคราะห์: {home_team} vs {away_team}")
                st.progress(prob)
                st.write(f"โอกาสเจ้าบ้านชนะ: **{prob*100:.1f}%**")
                
                if prob > 0.6:
                    st.success("✅ AI เชียร์เจ้าบ้าน!")
                elif prob < 0.4:
                    st.error("❌ AI คิดว่าเจ้าบ้านไม่รอด (เชียร์ทีมเยือน)")
                else:
                    st.warning("⚖️ สูสีมาก ออกได้ทุกหน้า")
                    
                st.caption(f"ราคาที่ควรจะเป็น (Fair Odds): {1/prob:.2f}")

            except IndexError:
                st.error("ข้อมูลทีมไม่เพียงพอ (อาจเป็นทีมเลื่อนชั้นใหม่)")

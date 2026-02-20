import streamlit as st
from scipy.stats import poisson

# ส่วนหัวของแอป
st.title("⚽ Football Score Predictor (Guru)")
st.subheader("คำนวณโอกาสชนะด้วยหลักการ Poisson Distribution")

# --- ส่วนของการรับค่า Input ---
col1, col2 = st.columns(2)

with col1:
    st.header("Home Team (เจ้าบ้าน)")
    h_att = st.number_input("Home Attack Strength (พลังบุก)", value=1.5)
    h_def = st.number_input("Home Defense Strength (พลังรับ)", value=1.0)
    avg_h_goals = st.number_input("League Avg Home Goals (ค่าเฉลี่ยประตูเจ้าบ้านทั้งลีก)", value=1.3)

with col2:
    st.header("Away Team (ทีมเยือน)")
    a_att = st.number_input("Away Attack Strength (พลังบุก)", value=1.2)
    a_def = st.number_input("Away Defense Strength (พลังรับ)", value=1.1)
    avg_a_goals = st.number_input("League Avg Away Goals (ค่าเฉลี่ยประตูทีมเยือนทั้งลีก)", value=1.1)

# --- ส่วนการคำนวณ Expected Goals (xG) ---
# แก้ไขจุดที่ Error: แยกการคำนวณให้ชัดเจน
exp_h = h_att * a_def * avg_h_goals
exp_a = a_att * h_def * avg_a_goals  # แก้จากบรรทัดที่ 56 เดิมของคุณ

st.divider()
st.write(f"### 🎯 Expected Goals (xG): {exp_h:.2f} - {exp_a:.2f}")

# --- ส่วนการทำนายผลแม่นยำ (Matrix) ---
max_goals = 6
home_probs = [poisson.pmf(i, exp_h) for i in range(max_goals)]
away_probs = [poisson.pmf(i, exp_a) for i in range(max_goals)]

# คำนวณโอกาส ชนะ/เสมอ/แพ้
home_win = 0
draw = 0
away_win = 0

for h in range(max_goals):
    for a in range(max_goals):
        prob = home_probs[h] * away_probs[a]
        if h > a:
            home_win += prob
        elif h < a:
            away_win += prob
        else:
            draw += prob

# --- แสดงผลลัพธ์ ---
c1, c2, c3 = st.columns(3)
c1.metric("เจ้าบ้านชนะ", f"{home_win*100:.1f}%")
c2.metric("เสมอ", f"{draw*100:.1f}%")
c3.metric("ทีมเยือนชนะ", f"{away_win*100:.1f}%")

st.info("💡 หมายเหตุ: นี่เป็นการคำนวณเชิงสถิติเบื้องต้น ไม่รวมปัจจัยเรื่องอาการบาดเจ็บหรือสภาพอากาศ")

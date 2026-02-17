import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from PIL import Image

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Football Simple", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

# --- 2. โหลดโลโก้ + สกัดสีอัตโนมัติ ---
LOGO_PATHS = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]
BASE_DIR = Path(__file__).parent
logo_path = None
for p in LOGO_PATHS:
    fp = BASE_DIR / p
    if fp.exists():
        logo_path = str(fp)
        break

DEFAULT_THEME = {
    "primary": "#00ffff", "secondary": "#ff006e",
    "bg_dark": "#0a0a0f", "bg_mid": "#12121a",
    "primary_rgba": "rgba(0,255,255,", "secondary_rgba": "rgba(255,0,110,"
}

@st.cache_data(ttl=86400)
def extract_colors_from_logo(img_path):
    """สกัดสีหลักจากโลโก้ด้วย KMeans"""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((80, 80))
        pixels = np.array(img).reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        order = np.argsort(-counts)
        sorted_colors = [tuple(colors[i]) for i in order]
        # กรองสีที่มืดหรือสว่างเกินไป
        valid = [c for c in sorted_colors if 25 < np.mean(c) < 245 and max(c) - min(c) > 20]
        if len(valid) >= 2:
            primary = valid[0]
            secondary = valid[1]
        elif len(valid) == 1:
            primary = valid[0]
            secondary = valid[0]
        else:
            primary = (0, 255, 255)
            secondary = (255, 0, 110)
        def to_hex(c): return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
        # สร้างสีพื้นหลังเข้มจาก primary
        bg_dark = (int(primary[0]*0.08), int(primary[1]*0.08), int(primary[2]*0.12))
        bg_dark = (min(20, bg_dark[0]), min(20, bg_dark[1]), min(30, bg_dark[2]))
        bg_mid = (int(primary[0]*0.12), int(primary[1]*0.12), int(primary[2]*0.18))
        bg_mid = (min(25, bg_mid[0]), min(25, bg_mid[1]), min(35, bg_mid[2]))
        return {
            "primary": to_hex(primary),
            "secondary": to_hex(secondary),
            "bg_dark": to_hex(bg_dark),
            "bg_mid": to_hex(bg_mid),
            "primary_rgba": f"rgba({primary[0]},{primary[1]},{primary[2]},",
            "secondary_rgba": f"rgba({secondary[0]},{secondary[1]},{secondary[2]},"
        }
    except Exception:
        return DEFAULT_THEME

theme = extract_colors_from_logo(logo_path) if logo_path else DEFAULT_THEME

# --- 3. CSS ธีม (ปรับตามสีโลโก้) ---
st.markdown(f"""
<style>
    .stApp {{ 
        background: linear-gradient(135deg, {theme['bg_dark']} 0%, {theme['bg_mid']} 50%, {theme['bg_dark']} 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {theme['bg_dark']} 0%, {theme['bg_mid']} 100%) !important;
        border-right: 1px solid {theme['primary_rgba']}0.4) !important;
        box-shadow: 0 0 20px {theme['secondary_rgba']}0.15);
    }}
    [data-testid="stSidebar"] .stMarkdown {{ color: {theme['primary']} !important; }}
    h1 {{ color: {theme['primary']} !important; text-shadow: 0 0 10px {theme['primary']}, 0 0 20px {theme['primary_rgba']}0.5); font-weight: 700 !important; }}
    .legend-box {{
        background: {theme['bg_mid']}ee;
        padding: 14px 20px; border-radius: 12px; display: flex; justify-content: space-around;
        margin-bottom: 20px; font-size: 0.95em;
        border: 1px solid {theme['primary_rgba']}0.3);
        box-shadow: 0 0 15px {theme['primary_rgba']}0.1), inset 0 0 30px rgba(0,0,0,0.3);
    }}
    .match-card {{
        background: linear-gradient(135deg, {theme['bg_mid']}f2 0%, {theme['bg_dark']}fa 100%);
        border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center;
        border-left: 6px solid #555;
        border: 1px solid {theme['secondary_rgba']}0.25);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4), 0 0 15px {theme['primary_rgba']}0.08);
        transition: all 0.3s ease;
    }}
    .match-card:hover {{
        box-shadow: 0 0 25px {theme['primary_rgba']}0.2), 0 0 40px {theme['secondary_rgba']}0.15);
        transform: translateX(4px);
    }}
    .strip-green {{ border-left-color: #00ff88 !important; box-shadow: 0 0 15px rgba(0,255,136,0.2) !important; }}
    .strip-red {{ border-left-color: #ff006e !important; box-shadow: 0 0 15px rgba(255,0,110,0.2) !important; }}
    .strip-yellow {{ border-left-color: #ffdd00 !important; box-shadow: 0 0 15px rgba(255,221,0,0.2) !important; }}
    .team {{ font-size: 16px; font-weight: bold; color: #f0f0f0; }}
    .vs {{ color: {theme['primary']}; font-size: 11px; margin: 0 10px; opacity: 0.8; }}
    .time {{ color: {theme['primary']}; font-size: 12px; display: block; margin-bottom: 4px; opacity: 0.9; }}
    .percent {{ font-size: 15px; font-weight: bold; }}
    .section-date {{
        color: {theme['primary']} !important; margin-top: 24px !important;
        border-bottom: 1px solid {theme['primary_rgba']}0.4) !important;
        padding-bottom: 8px !important; text-shadow: 0 0 8px {theme['primary_rgba']}0.4);
    }}
    .chart-wrapper {{ background: {theme['bg_mid']}cc; border-radius: 12px; padding: 15px; border: 1px solid {theme['primary_rgba']}0.2); margin-bottom: 12px; }}
    .logo-header {{ text-align: center; padding: 10px 0 20px 0; }}
</style>
""", unsafe_allow_html=True)

# --- 4. แสดงโลโก้ตรงหัวเว็บ ---
if logo_path:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, use_container_width=True)

# --- 2. Sidebar เมนูซ้ายมือ ---
with st.sidebar:
    st.markdown("## ⚡ FOOTBALL AI")
    st.markdown("---")
    
    menu = st.radio(
        "เมนู",
        ["หน้าหลัก", "เกี่ยวกับ", "การตั้งค่า"],
        label_visibility="collapsed",
        index=0
    )
    
    st.markdown("---")
    num_matches = st.slider("แสดงจำนวนคู่", min_value=5, max_value=20, value=10)
    st.markdown("---")
    st.caption("🤖 AI คำนวณจากสถิติย้อนหลัง 3 ปี")

# --- 3. ส่วนอธิบายสี (Legend) ---
st.title("⚽ ตารางบอลสีนำโชค")

st.markdown("""
<div class="legend-box">
    <div style="color:#00ff88;">🟢 เจ้าบ้านนอนมา</div>
    <div style="color:#ffdd00;">🟡 สูสี/ออกเสมอ</div>
    <div style="color:#ff006e;">🔴 ทีมเยือนบุกชนะ</div>
</div>
<div style="font-size:0.8em; color:#666; text-align:center; margin-bottom:20px;">
    *คำเตือน: AI คำนวณจากสถิติย้อนหลัง ไม่รวมข่าวนักเตะเจ็บรายวัน
</div>
""", unsafe_allow_html=True)

# --- 4. ระบบสมองกล (AI Engine) ---
@st.cache_resource(ttl=3600)
def load_ai():
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
    mapping = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nott'm Forest", 
               "Wolves": "Wolves", "Man City": "Man City", "Newcastle Utd": "Newcastle",
               "Sheffield Utd": "Sheffield United", "Luton Town": "Luton", "West Ham Utd": "West Ham",
               "Ipswich Town": "Ipswich"}
    if name in known: return name
    if name in mapping and mapping[name] in known: return mapping[name]
    return None

def create_win_prob_chart(home_prob, away_prob, home_team, away_team, accent_color="#00ffff"):
    """สร้างกราฟวงกลมโอกาสชนะ เจ้าบ้าน vs ทีมเยือน"""
    fig = go.Figure(data=[go.Pie(
        labels=[f"เจ้าบ้าน ({home_team})", f"ทีมเยือน ({away_team})"],
        values=[home_prob * 100, away_prob * 100],
        hole=0.6,
        marker=dict(
            colors=['#00ff88', '#ff006e'],
            line=dict(color='#0a0a0f', width=2)
        ),
        textinfo='percent',
        textposition='inside',
        textfont=dict(size=14, color='#ffffff'),
        hovertemplate='<b>%{label}</b><br>โอกาส: %{value:.1f}%<extra></extra>'
    )])
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(color=accent_color, size=12),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=220,
        annotations=[dict(
            text=f'{home_prob*100:.0f}% vs {away_prob*100:.0f}%',
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(color=accent_color)
        )]
    )
    return fig

# --- 5. แสดงผลหน้าจอ ---
if menu == "เกี่ยวกับ":
    st.info("""
    ### เกี่ยวกับแอปนี้
    แอปวิเคราะห์โอกาสชนะบอล EPL ด้วย AI (Random Forest) 
    - ข้อมูลจาก football-data.co.uk ย้อนหลัง 3 ปี
    - คำนวณฟอร์ม 5 นัดล่าสุดของเจ้าบ้าน
    - สีแสดงแนวโน้ม: เขียว=เจ้าบ้านฟอร์มดี, แดง=ทีมเยือนมีโอกาส, เหลือง=สูสี
    """)
    st.stop()
elif menu == "การตั้งค่า":
    st.info("ใช้ Slider ใน Sidebar เพื่อปรับจำนวนคู่ที่แสดง")
    st.stop()

rf, le, matches = load_ai()

if rf:
    try:
        r = requests.get("https://fixturedownload.com/feed/json/epl-2025", headers={"User-Agent": "Mozilla/5.0"})
        fixtures = pd.read_json(StringIO(r.text))
        fixtures['DateUtc'] = pd.to_datetime(fixtures['DateUtc'], utc=True)
        upcoming = fixtures[fixtures['DateUtc'] >= pd.Timestamp.now('UTC')].sort_values('DateUtc').head(num_matches)
        
        current_date = ""
        for idx, (_, row) in enumerate(upcoming.iterrows()):
            date_str = row['DateUtc'].strftime("%d %b (วัน%A)")
            time_str = row['DateUtc'].strftime("%H:%M")
            
            if current_date != date_str:
                st.markdown(f"<h4 class='section-date'>📅 {date_str}</h4>", unsafe_allow_html=True)
                current_date = date_str
            
            h = map_name(row['HomeTeam'], le.classes_)
            a = map_name(row['AwayTeam'], le.classes_)
            
            if h and a:
                h_stat = matches[matches["HomeTeam"] == h].iloc[-1]
                prob = rf.predict_proba([[le.transform([h])[0], le.transform([a])[0], h_stat["Form"]]])[0][1]
                away_prob = 1 - prob
                
                if prob > 0.60:
                    color_class = "strip-green"
                    percent_text = f"<span style='color:#00ff88;'>{prob*100:.0f}% เจ้าบ้าน</span>"
                elif prob < 0.40:
                    color_class = "strip-red"
                    percent_text = f"<span style='color:#ff006e;'>{(1-prob)*100:.0f}% ทีมเยือน</span>"
                else:
                    color_class = "strip-yellow"
                    percent_text = "<span style='color:#ffdd00;'>สูสี 50/50</span>"
                
                # Layout: การ์ด + กราฟวงกลม ข้างๆ กัน
                col_card, col_chart = st.columns([2, 1])
                
                with col_card:
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
                
                with col_chart:
                    fig = create_win_prob_chart(prob, away_prob, h, a, theme["primary"])
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
            else:
                st.caption(f"⚠️ ข้ามคู่ {row['HomeTeam']} vs {row['AwayTeam']} (ชื่อทีมไม่ตรง)")
                
    except Exception as e:
        st.error(f"โหลดข้อมูลไม่ได้: {e}")

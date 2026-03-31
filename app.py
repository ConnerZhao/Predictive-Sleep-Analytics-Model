import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import os

# -- Page Config --
st.set_page_config(page_title="Sleep Analytics", page_icon="🌙", layout="wide")

# -- CSS + Animations --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* === KEYFRAMES === */
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes softPulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.55; }
}
@keyframes growWidth {
    from { width: 0 !important; }
    to   { width: var(--bar-w); }
}
@keyframes popIn {
    from { opacity: 0; transform: translateX(-50%) scale(0.3); }
    to   { opacity: 1; transform: translateX(-50%) scale(1); }
}

/* === BASE === */
.stApp {
    background-color: #07101f;
    color: #f1f5f9;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* === HERO === */
.hero-section {
    background: linear-gradient(135deg, #3730a3 0%, #1e1b4b 45%, #4c1d95 100%);
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite, fadeIn 0.6s ease forwards;
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 24px 64px -12px rgba(67, 56, 202, 0.45);
    border: 1px solid rgba(129, 140, 248, 0.15);
}
.hero-section h1 {
    margin: 0 0 0.45rem 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    line-height: 1.2;
}
.hero-section p {
    opacity: 0.72;
    font-size: 0.97rem;
    margin: 0;
    color: #c7d2fe;
}
.hero-section a {
    color: #a5b4fc;
    text-decoration: none;
    border-bottom: 1px solid rgba(165, 180, 252, 0.35);
    transition: color 0.2s ease, border-color 0.2s ease;
}
.hero-section a:hover {
    color: #e0e7ff;
    border-color: rgba(224, 231, 255, 0.6);
}

/* === METRIC CARDS === */
.metric-card {
    background: rgba(13, 21, 40, 0.85);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    opacity: 0;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                border-color 0.3s ease,
                box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 20px 48px -8px rgba(99, 102, 241, 0.22);
}
.metric-card.c1 { animation: fadeInUp 0.5s ease 0.10s forwards; }
.metric-card.c2 { animation: fadeInUp 0.5s ease 0.20s forwards; }
.metric-card.c3 { animation: fadeInUp 0.5s ease 0.30s forwards; }

.metric-label {
    color: #475569;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.45rem;
}
.metric-value {
    color: #f1f5f9;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.metric-delta {
    color: #6366f1;
    font-size: 0.77rem;
    font-weight: 500;
    margin-bottom: 1rem;
}
.metric-bar-track {
    height: 3px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 99px;
    position: relative;
    overflow: visible;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #4f46e5, #818cf8);
    width: var(--bar-w);
    animation: growWidth 0.9s cubic-bezier(0.4, 0, 0.2, 1) 0.45s both;
}
.metric-bar-avg {
    position: absolute;
    top: -4px;
    width: 2px;
    height: 11px;
    background: rgba(148, 163, 184, 0.55);
    border-radius: 2px;
    transform: translateX(-50%);
    animation: popIn 0.35s ease 1.2s both;
}

/* === STATUS BADGE === */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 1.15rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    animation: fadeIn 0.5s ease 0.3s both;
    margin-top: 0.5rem;
}
.badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    animation: softPulse 2.5s ease infinite;
}
.badge-rested {
    background: rgba(5, 46, 22, 0.55);
    border: 1px solid rgba(52, 211, 153, 0.32);
    color: #34d399;
}
.badge-rested .badge-dot { background: #34d399; }
.badge-tired {
    background: rgba(127, 29, 29, 0.55);
    border: 1px solid rgba(248, 113, 113, 0.32);
    color: #f87171;
}
.badge-tired .badge-dot { background: #f87171; }

/* === GRADIENT DIVIDER === */
.g-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    margin: 1.75rem 0;
    animation: fadeIn 0.4s ease forwards;
}

/* === INSIGHT CARDS === */
.insight-card {
    border-radius: 14px;
    padding: 1.1rem 1.35rem;
    opacity: 0;
    margin-bottom: 0.5rem;
    line-height: 1.55;
}
.insight-win {
    background: rgba(2, 44, 34, 0.45);
    border: 1px solid rgba(16, 185, 129, 0.22);
    border-left: 3px solid #10b981;
    animation: fadeInUp 0.45s ease 0.15s forwards;
}
.insight-focus {
    background: rgba(69, 26, 3, 0.45);
    border: 1px solid rgba(245, 158, 11, 0.22);
    border-left: 3px solid #f59e0b;
    animation: fadeInUp 0.45s ease 0.28s forwards;
}
.insight-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.35rem;
}
.insight-win  .insight-eyebrow { color: #10b981; }
.insight-focus .insight-eyebrow { color: #f59e0b; }
.insight-body {
    color: #cbd5e1;
    font-size: 0.91rem;
}
.insight-body strong { color: #f1f5f9; }

/* === SECTION HEADERS === */
.sec-hdr {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.9rem;
    animation: slideInLeft 0.4s ease forwards;
}
.sec-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.25), transparent);
    margin-left: 0.4rem;
}

/* === CHART CONTAINER === */
.chart-wrap {
    background: rgba(8, 14, 27, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 0.25rem;
    animation: fadeIn 0.5s ease 0.2s both;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13, 21, 40, 0.7) !important;
    border-radius: 10px !important;
    padding: 0.25rem !important;
    gap: 0.2rem !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.18) !important;
    color: #a5b4fc !important;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: #090f1f !important;
    border-right: 1px solid rgba(99, 102, 241, 0.08) !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.28);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }
</style>
""", unsafe_allow_html=True)

# -- Constants --
CONTINUOUS_OUTCOMES = ['sleep_duration_hrs', 'sleep_quality_score', 'deep_sleep_percentage']

OUTCOME_LABELS = {
    'sleep_duration_hrs': 'Duration (hrs)',
    'sleep_quality_score': 'Quality Score',
    'deep_sleep_percentage': 'Deep Sleep %',
}

FEATURE_LABELS = {
    'age': 'Age',
    'bmi': 'BMI',
    'caffeine_mg_before_bed': 'Caffeine (mg)',
    'alcohol_units_before_bed': 'Alcohol (Shots)',
    'screen_time_before_bed_mins': 'Screen Time (mins)',
    'exercise_day': 'Exercised Today',
    'steps_that_day': 'Steps',
    'nap_duration_mins': 'Nap Duration (mins)',
    'stress_score': 'Stress Level',
    'work_hours_that_day': 'Work Hours',
    'sleep_aid_used': 'Sleep Aid Used',
    'shift_work': 'Shift Work',
    'room_temperature_celsius': 'Room Temp (°C)',
    'weekend_sleep_diff_hrs': 'Weekend Sleep Diff (hrs)',
    'sleep_disorder_risk': 'Sleep Disorder Risk',
    'gender_Male': 'Gender: Male',
    'gender_Other': 'Gender: Other',
    'occupation_Driver': 'Occupation: Driver',
    'occupation_Freelancer': 'Occupation: Freelancer',
    'occupation_Homemaker': 'Occupation: Homemaker',
    'occupation_Lawyer': 'Occupation: Lawyer',
    'occupation_Manager': 'Occupation: Manager',
    'occupation_Nurse': 'Occupation: Nurse',
    'occupation_Retired': 'Occupation: Retired',
    'occupation_Sales': 'Occupation: Sales',
    'occupation_Software Engineer': 'Occupation: Software Eng.',
    'occupation_Student': 'Occupation: Student',
    'occupation_Teacher': 'Occupation: Teacher',
    'country_Brazil': 'Country: Brazil',
    'country_Canada': 'Country: Canada',
    'country_France': 'Country: France',
    'country_Germany': 'Country: Germany',
    'country_India': 'Country: India',
    'country_Italy': 'Country: Italy',
    'country_Japan': 'Country: Japan',
    'country_Mexico': 'Country: Mexico',
    'country_Netherlands': 'Country: Netherlands',
    'country_South Korea': 'Country: South Korea',
    'country_Spain': 'Country: Spain',
    'country_Sweden': 'Country: Sweden',
    'country_UK': 'Country: UK',
    'country_USA': 'Country: USA',
    'chronotype_Morning': 'Chronotype: Morning',
    'chronotype_Neutral': 'Chronotype: Neutral',
    'mental_health_condition_Both': 'Mental Health: Both',
    'mental_health_condition_Depression': 'Mental Health: Depression',
    'mental_health_condition_Healthy': 'Mental Health: Healthy',
    'season_Spring': 'Season: Spring',
    'season_Summer': 'Season: Summer',
    'season_Winter': 'Season: Winter',
    'day_type_Weekend': 'Day: Weekend',
}

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(8,14,27,0.5)',
    font=dict(color='#94a3b8', family='Inter, sans-serif'),
)

# -- Model Loading --
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        regressor   = joblib.load(os.path.join(model_dir, 'regressor.pkl'))
        classifier  = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
        ordinal_order = joblib.load(os.path.join(model_dir, 'ordinal_order.pkl'))
        return regressor, classifier, feature_cols, ordinal_order
    except Exception:
        st.error("Model files missing in /models directory.")
        st.stop()

regressor, classifier, feature_cols, ordinal_order = load_models()

@st.cache_resource
def get_shap_explainer(_regressor):
    return shap.TreeExplainer(_regressor)

explainer = get_shap_explainer(regressor)

# -- Hero --
st.markdown("""
<div class="hero-section">
    <h1>🌙 Predictive Sleep Analytics</h1>
    <p>
        Model trained on real-world sleep data &mdash;
        <a href="https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset"
           target="_blank" rel="noopener">Kaggle Sleep Health Dataset</a>
    </p>
</div>
""", unsafe_allow_html=True)

# -- SIDEBAR --
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    with st.expander("Personal Demographics", expanded=True):
        age        = st.slider("Age", 18, 80, 30)
        bmi        = st.slider("BMI", 15.0, 45.0, 24.0)
        gender     = st.selectbox("Gender", ["Female", "Male", "Other"])
        occupation = st.selectbox("Occupation", [
            "Doctor", "Driver", "Freelancer", "Homemaker", "Lawyer",
            "Manager", "Nurse", "Retired", "Sales", "Software Engineer",
            "Student", "Teacher",
        ])
        country = st.selectbox("Country", [
            "Australia", "Brazil", "Canada", "France", "Germany",
            "India", "Italy", "Japan", "Mexico", "Netherlands",
            "South Korea", "Spain", "Sweden", "UK", "USA",
        ])

    with st.expander("Daily Activity", expanded=True):
        steps        = st.number_input("Steps Today", 0, 30000, 8000, step=500)
        exercise_day = st.toggle("Exercised Today", value=True)
        stress_score = st.select_slider("Stress Level", options=range(1, 11), value=5)
        work_hours   = st.slider("Work Hours", 0.0, 18.0, 8.0, step=0.5)
        nap_duration = st.slider("Nap Duration (mins)", 0, 120, 0)

    with st.expander("Habits & Environment", expanded=False):
        caffeine       = st.slider("Caffeine (mg)", 0, 500, 100)
        alcohol        = st.slider("Alcohol (units)", 0.0, 6.0, 0.0)
        room_temp      = st.slider("Temp (°C)", 15.0, 28.0, 20.0)
        screen_time    = st.slider("Screen Time (mins)", 0, 300, 60)
        sleep_aid      = st.toggle("Sleep Aid Used", value=False)
        shift_work_flag = st.toggle("Shift Work", value=False)

    with st.expander("Context", expanded=False):
        chronotype          = st.selectbox("Chronotype", ["Evening", "Morning", "Neutral"])
        mental_health       = st.selectbox("Mental Health", ["Anxiety", "Both", "Depression", "Healthy"])
        sleep_disorder_risk = st.selectbox("Sleep Disorder Risk", ["Healthy", "Mild", "Moderate", "Severe"])
        season   = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])
        day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
        weekend_diff = st.slider("Weekend Sleep Diff (hrs)", -2.0, 4.0, 1.0, step=0.1)

# -- Input Builder --
def build_input_df():
    row = {col: 0 for col in feature_cols}
    row['age']                        = age
    row['bmi']                        = bmi
    row['steps_that_day']             = steps
    row['exercise_day']               = int(exercise_day)
    row['stress_score']               = stress_score
    row['caffeine_mg_before_bed']     = caffeine
    row['alcohol_units_before_bed']   = alcohol
    row['room_temperature_celsius']   = room_temp
    row['screen_time_before_bed_mins']= screen_time
    row['nap_duration_mins']          = nap_duration
    row['work_hours_that_day']        = work_hours
    row['sleep_aid_used']             = int(sleep_aid)
    row['shift_work']                 = int(shift_work_flag)
    row['weekend_sleep_diff_hrs']     = weekend_diff
    row['sleep_disorder_risk']        = ordinal_order.get(sleep_disorder_risk, 0)

    if gender == 'Male':   row['gender_Male']  = 1
    elif gender == 'Other': row['gender_Other'] = 1

    occ_col = f'occupation_{occupation}'
    if occ_col in row: row[occ_col] = 1

    country_col = f'country_{country}'
    if country_col in row: row[country_col] = 1

    if chronotype == 'Morning':  row['chronotype_Morning']  = 1
    elif chronotype == 'Neutral': row['chronotype_Neutral'] = 1

    if mental_health == 'Both':        row['mental_health_condition_Both']       = 1
    elif mental_health == 'Depression': row['mental_health_condition_Depression'] = 1
    elif mental_health == 'Healthy':    row['mental_health_condition_Healthy']    = 1

    if season == 'Spring': row['season_Spring'] = 1
    elif season == 'Summer': row['season_Summer'] = 1
    elif season == 'Winter': row['season_Winter'] = 1

    if day_type == 'Weekend': row['day_type_Weekend'] = 1

    return pd.DataFrame([row])[feature_cols]

input_df = build_input_df()

# -- Predictions --
preds       = regressor.predict(input_df)[0]
prob_rested = classifier.predict_proba(input_df)[0][1]

# -- SHAP --
raw_shap = explainer.shap_values(input_df)
if isinstance(raw_shap, list):
    shap_vals = np.stack(raw_shap, axis=-1)
else:
    shap_vals = raw_shap

# -- Top Section: Gauge + Radar --
col_score, col_radar = st.columns([1, 1.2])

with col_score:
    st.markdown('<p class="sec-hdr">Analysis Summary</p>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_rested * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': '#f1f5f9'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Rest Likelihood", 'font': {'size': 15, 'color': "#64748b"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155",
                     'tickfont': {'color': '#64748b', 'size': 10}},
            'bar': {'color': "#6366f1", 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  40], 'color': 'rgba(220, 38, 38,  0.15)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100],'color': 'rgba(16,  185, 129, 0.15)'},
            ],
            'threshold': {
                'line': {'color': '#818cf8', 'width': 2},
                'thickness': 0.8,
                'value': prob_rested * 100,
            },
        }
    ))
    fig_gauge.update_layout(
        height=260,
        margin=dict(t=50, b=0, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if prob_rested > 0.5:
        st.markdown(
            '<div class="status-badge badge-rested">'
            '<span class="badge-dot"></span>OPTIMAL RECOVERY ANTICIPATED'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge badge-tired">'
            '<span class="badge-dot"></span>RECOVERY DEFICIT LIKELY'
            '</div>',
            unsafe_allow_html=True,
        )

with col_radar:
    st.markdown('<p class="sec-hdr">Sleep Architecture</p>', unsafe_allow_html=True)

    categories = ['Duration', 'Quality', 'Deep Sleep']
    dur_norm  = float(np.clip((preds[0] - 3.0)  / (10.5 - 3.0), 0, 1))
    qual_norm = float(np.clip((preds[1] - 1.0)  / (10.0 - 1.0), 0, 1))
    deep_norm = float(np.clip((preds[2] - 5.0)  / (30.0 - 5.0), 0, 1))
    avg_dur   = (6.42 - 3.0)  / (10.5 - 3.0)
    avg_qual  = (4.87 - 1.0)  / (10.0 - 1.0)
    avg_deep  = (20.25 - 5.0) / (30.0 - 5.0)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[avg_dur, avg_qual, avg_deep],
        theta=categories,
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.07)',
        line=dict(color='rgba(148, 163, 184, 0.4)', width=1, dash='dot'),
        name='Avg Sleeper',
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[dur_norm, qual_norm, deep_norm],
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',
        line=dict(color='#818cf8', width=2.5),
        name='Your Profile',
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False,
                gridcolor="#1e293b",
            ),
            angularaxis=dict(
                gridcolor="#1e293b",
                linecolor="#1e293b",
                tickfont=dict(color='#94a3b8', size=12),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.18,
            font=dict(color='#94a3b8', size=11),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=30, l=40, r=40),
        height=290,
        font=dict(family='Inter, sans-serif'),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -- Gradient Divider --
st.markdown('<div class="g-divider"></div>', unsafe_allow_html=True)

# -- Metric Cards --
st.markdown('<p class="sec-hdr">Predicted Outcomes</p>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

# Progress bar helpers (normalize to %)
dur_pct  = f"{np.clip((preds[0] - 3.0)  / (10.5 - 3.0), 0, 1) * 100:.1f}%"
qual_pct = f"{np.clip((preds[1] - 1.0)  / (10.0 - 1.0), 0, 1) * 100:.1f}%"
deep_pct = f"{np.clip((preds[2] - 5.0)  / (30.0 - 5.0), 0, 1) * 100:.1f}%"

dur_avg_pct  = f"{(6.42 - 3.0)  / (10.5 - 3.0) * 100:.1f}%"
qual_avg_pct = f"{(4.87 - 1.0)  / (10.0 - 1.0) * 100:.1f}%"
deep_avg_pct = f"{(20.25 - 5.0) / (30.0 - 5.0) * 100:.1f}%"

cards = [
    (m1, "c1", "Total Sleep",   f"{preds[0]:.1f}h",  "vs 6.4h avg", dur_pct,  dur_avg_pct),
    (m2, "c2", "Quality Score", f"{preds[1]:.1f}/10", "vs 4.9 avg",  qual_pct, qual_avg_pct),
    (m3, "c3", "Deep Sleep",    f"{preds[2]:.1f}%",   "vs 20% avg",  deep_pct, deep_avg_pct),
]
for col, cls, label, val, delta, bar_w, avg_left in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">{delta}</div>
            <div class="metric-bar-track">
                <div class="metric-bar-fill" style="--bar-w:{bar_w};"></div>
                <div class="metric-bar-avg"  style="left:{avg_left};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -- Gradient Divider --
st.markdown('<div class="g-divider"></div>', unsafe_allow_html=True)

# -- SHAP Section --
st.markdown('<p class="sec-hdr">🔍 Feature Impact Analysis</p>', unsafe_allow_html=True)

t1, t2 = st.tabs(["Visualization", "Raw Data"])

with t1:
    st.markdown(
        "<p style='color:#475569; font-size:0.88rem; margin-bottom:1rem;'>"
        "Top factors influencing your sleep today — "
        "<span style='color:#0d9488;'>■</span> Positive &nbsp;"
        "<span style='color:#e05252;'>■</span> Negative impact"
        "</p>",
        unsafe_allow_html=True,
    )

    inner_tabs = st.tabs(["Quality", "Duration", "Deep Sleep"])
    tab_to_outcome = ['sleep_quality_score', 'sleep_duration_hrs', 'deep_sleep_percentage']

    for i, tab in enumerate(inner_tabs):
        with tab:
            outcome_idx  = CONTINUOUS_OUTCOMES.index(tab_to_outcome[i])
            outcome_shap = shap_vals[0, :, outcome_idx]

            shap_df = pd.DataFrame({
                'Feature': [FEATURE_LABELS.get(f, f) for f in feature_cols],
                'Impact':  outcome_shap,
            })
            shap_df['Abs_Impact'] = shap_df['Impact'].abs()
            top_features = shap_df.nlargest(8, 'Abs_Impact').sort_values('Impact', ascending=True)

            colors = ['#0d9488' if v > 0 else '#e05252' for v in top_features['Impact']]

            fig_impact = go.Figure(go.Bar(
                x=top_features['Impact'],
                y=top_features['Feature'],
                orientation='h',
                marker=dict(
                    color=colors,
                    opacity=0.85,
                    line=dict(width=0),
                ),
                text=[f"{v:+.3f}" for v in top_features['Impact']],
                textposition='outside',
                textfont=dict(color='#94a3b8', size=11),
            ))
            fig_impact.update_layout(
                **CHART_LAYOUT,
                height=330,
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#1e293b",
                    zerolinecolor="#334155",
                    zerolinewidth=1.5,
                    tickfont=dict(color='#64748b', size=10),
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(color='#94a3b8', size=11),
                ),
            )
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_impact, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with t2:
    st.markdown(
        "<p style='color:#475569; font-size:0.88rem; margin-bottom:0.5rem;'>Raw SHAP values per feature and outcome</p>",
        unsafe_allow_html=True,
    )
    raw_shap_df = pd.DataFrame(
        shap_vals[0],
        index=feature_cols,
        columns=[OUTCOME_LABELS[o] for o in CONTINUOUS_OUTCOMES],
    )
    st.dataframe(
        raw_shap_df.style.background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True,
    )

# -- Gradient Divider --
st.markdown('<div class="g-divider"></div>', unsafe_allow_html=True)

# -- Personalized Insights --
st.markdown('<p class="sec-hdr">💡 Personalized Insights</p>', unsafe_allow_html=True)

quality_idx  = CONTINUOUS_OUTCOMES.index('sleep_quality_score')
quality_shap = pd.Series(shap_vals[0, :, quality_idx], index=feature_cols)
worst_factor = quality_shap.idxmin()
best_factor  = quality_shap.idxmax()

best_label  = FEATURE_LABELS.get(best_factor,  best_factor)
worst_label = FEATURE_LABELS.get(worst_factor, worst_factor)

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"""
    <div class="insight-card insight-win">
        <div class="insight-eyebrow">Quick Win</div>
        <div class="insight-body">
            Your <strong>{best_label}</strong> is currently your strongest sleep booster.
            Keep it up — this is working in your favour tonight.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="insight-card insight-focus">
        <div class="insight-eyebrow">Focus Area</div>
        <div class="insight-body">
            Your <strong>{worst_label}</strong> is having the most negative impact
            on sleep quality. Consider adjusting this for a better night.
        </div>
    </div>
    """, unsafe_allow_html=True)

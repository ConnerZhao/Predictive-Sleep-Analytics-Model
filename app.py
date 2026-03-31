import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import os

# -- Page Config --
st.set_page_config(page_title="Predictive Sleep Analytics Model", page_icon="", layout="wide")

# -- Enhanced CSS --
st.markdown("""
<style>
    /* Main Background and Font */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Modern Glassmorphism Cards */
    .metric-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        transition: transform 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #4338ca 0%, #1e1b4b 100%);
        padding: 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }

    /* Status Badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-rested { background: #065f46; color: #34d399; }
    .badge-tired { background: #7f1d1d; color: #f87171; }

    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# -- Header --
st.markdown("""
<div class="hero-section">
    <h1 style='margin:0; font-size: 2.5rem;'>Predictive Sleep Analytics Model</h1>
    <p style='opacity: 0.8; font-size: 1.1rem;'>Model trained with real-world sleep data.</p>
from <a href='https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset' target='_blank'>Kaggle Sleep Health Dataset</a>
</div>
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
    plot_bgcolor='rgba(15,23,42,0.5)',
    font=dict(color='#94a3b8'),
)

# -- Model Loading --
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        regressor = joblib.load(os.path.join(model_dir, 'regressor.pkl'))
        classifier = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
        ordinal_order = joblib.load(os.path.join(model_dir, 'ordinal_order.pkl'))
        return regressor, classifier, feature_cols, ordinal_order
    except:
        st.error("Model files missing in /models directory.")
        st.stop()

regressor, classifier, feature_cols, ordinal_order = load_models()

@st.cache_resource
def get_shap_explainer(_regressor):
    return shap.TreeExplainer(_regressor)

explainer = get_shap_explainer(regressor)

# -- SIDEBAR: All Inputs --
with st.sidebar:
    st.header("Parameters")

    with st.expander("Personal Demographics", expanded=True):
        age = st.slider("Age", 18, 80, 30)
        bmi = st.slider("BMI", 15.0, 45.0, 24.0)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        occupation = st.selectbox("Occupation", [
            "Doctor", "Driver", "Freelancer", "Homemaker", "Lawyer",
            "Manager", "Nurse", "Retired", "Sales", "Software Engineer",
            "Student", "Teacher"
        ])
        country = st.selectbox("Country", [
            "Australia", "Brazil", "Canada", "France", "Germany",
            "India", "Italy", "Japan", "Mexico", "Netherlands",
            "South Korea", "Spain", "Sweden", "UK", "USA"
        ])

    with st.expander("Daily Activity", expanded=True):
        steps = st.number_input("Steps Today", 0, 30000, 8000, step=500)
        exercise_day = st.toggle("Exercised Today", value=True)
        stress_score = st.select_slider("Stress Level", options=range(1, 11), value=5)
        work_hours = st.slider("Work Hours", 0.0, 18.0, 8.0, step=0.5)
        nap_duration = st.slider("Nap Duration (mins)", 0, 120, 0)

    with st.expander("Personal Habits and Environment", expanded=False):
        caffeine = st.slider("Caffeine (mg)", 0, 500, 100)
        alcohol = st.slider("Alcohol (units)", 0.0, 6.0, 0.0)
        room_temp = st.slider("Temp (°C)", 15.0, 28.0, 20.0)
        screen_time = st.slider("Screen Time (mins)", 0, 300, 60)
        sleep_aid = st.toggle("Sleep Aid Used", value=False)
        shift_work_flag = st.toggle("Shift Work", value=False)

    with st.expander("Context", expanded=False):
        chronotype = st.selectbox("Chronotype", ["Evening", "Morning", "Neutral"])
        mental_health = st.selectbox("Mental Health", ["Anxiety", "Both", "Depression", "Healthy"])
        sleep_disorder_risk = st.selectbox("Sleep Disorder Risk", ["Healthy", "Mild", "Moderate", "Severe"])
        season = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])
        day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
        weekend_diff = st.slider("Weekend Sleep Diff (hrs)", -2.0, 4.0, 1.0, step=0.1)

# -- Data Processing --
def build_input_df():
    row = {col: 0 for col in feature_cols}

    # Numeric / binary features
    row['age'] = age
    row['bmi'] = bmi
    row['steps_that_day'] = steps
    row['exercise_day'] = int(exercise_day)
    row['stress_score'] = stress_score
    row['caffeine_mg_before_bed'] = caffeine
    row['alcohol_units_before_bed'] = alcohol
    row['room_temperature_celsius'] = room_temp
    row['screen_time_before_bed_mins'] = screen_time
    row['nap_duration_mins'] = nap_duration
    row['work_hours_that_day'] = work_hours
    row['sleep_aid_used'] = int(sleep_aid)
    row['shift_work'] = int(shift_work_flag)
    row['weekend_sleep_diff_hrs'] = weekend_diff

    # Ordinal: sleep_disorder_risk
    row['sleep_disorder_risk'] = ordinal_order.get(sleep_disorder_risk, 0)

    # One-hot: gender (baseline = Female)
    if gender == 'Male':
        row['gender_Male'] = 1
    elif gender == 'Other':
        row['gender_Other'] = 1

    # One-hot: occupation (baseline = Doctor)
    occ_col = f'occupation_{occupation}'
    if occ_col in row:
        row[occ_col] = 1

    # One-hot: country (baseline = Australia)
    country_col = f'country_{country}'
    if country_col in row:
        row[country_col] = 1

    # One-hot: chronotype (baseline = Evening)
    if chronotype == 'Morning':
        row['chronotype_Morning'] = 1
    elif chronotype == 'Neutral':
        row['chronotype_Neutral'] = 1

    # One-hot: mental_health_condition (baseline = Anxiety)
    if mental_health == 'Both':
        row['mental_health_condition_Both'] = 1
    elif mental_health == 'Depression':
        row['mental_health_condition_Depression'] = 1
    elif mental_health == 'Healthy':
        row['mental_health_condition_Healthy'] = 1

    # One-hot: season (baseline = Autumn)
    if season == 'Spring':
        row['season_Spring'] = 1
    elif season == 'Summer':
        row['season_Summer'] = 1
    elif season == 'Winter':
        row['season_Winter'] = 1

    # One-hot: day_type (baseline = Weekday)
    if day_type == 'Weekend':
        row['day_type_Weekend'] = 1

    return pd.DataFrame([row])[feature_cols]

input_df = build_input_df()

# -- Prediction Engine --
preds = regressor.predict(input_df)[0]
# preds[0] = sleep_duration_hrs, preds[1] = sleep_quality_score, preds[2] = deep_sleep_percentage
prob_rested = classifier.predict_proba(input_df)[0][1]

# -- SHAP Values --
raw_shap = explainer.shap_values(input_df)
if isinstance(raw_shap, list):
    # list of (n_samples, n_features) per output → (n_samples, n_features, n_outputs)
    shap_vals = np.stack(raw_shap, axis=-1)
else:
    shap_vals = raw_shap  # already (n_samples, n_features, n_outputs)

# -- MAIN LAYOUT --
col_score, col_radar = st.columns([1, 1.2])

with col_score:
    st.subheader("Analysis Summary")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_rested * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Rest Likelihood", 'font': {'size': 18, 'color': "#94a3b8"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#6366f1"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(220, 38, 38, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'},
            ],
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=50, b=0, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prob_rested > 0.5:
        st.markdown('<div class="status-badge badge-rested">OPTIMAL RECOVERY ANTICIPATED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge badge-tired">RECOVERY DEFICIT LIKELY</div>', unsafe_allow_html=True)

with col_radar:
    st.subheader("Sleep Architecture")

    categories = ['Duration', 'Quality', 'Deep Sleep']

    # Normalize predictions to [0, 1] using dataset min/max
    dur_norm  = np.clip((preds[0] - 3.0)  / (10.5 - 3.0),  0, 1)
    qual_norm = np.clip((preds[1] - 1.0)  / (10.0 - 1.0),  0, 1)
    deep_norm = np.clip((preds[2] - 5.0)  / (30.0 - 5.0),  0, 1)

    # Dataset averages normalized
    avg_dur  = (6.42 - 3.0)  / (10.5 - 3.0)   # ≈ 0.46
    avg_qual = (4.87 - 1.0)  / (10.0 - 1.0)   # ≈ 0.43
    avg_deep = (20.25 - 5.0) / (30.0 - 5.0)   # ≈ 0.61

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=[avg_dur, avg_qual, avg_deep],
        theta=categories,
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.1)',
        line=dict(color='rgba(148, 163, 184, 0.5)', width=1, dash='dot'),
        name='Avg Sleeper'
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=[dur_norm, qual_norm, deep_norm],
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#818cf8', width=3),
        name='Your Profile'
    ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#334155"),
            angularaxis=dict(gridcolor="#334155", linecolor="#334155")
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.2),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=40, r=40),
        height=300
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# -- Metrics Row --
m1, m2, m3 = st.columns(3)
metrics = [
    ("Total Sleep",   f"{preds[0]:.1f}h",  "vs 6.4h avg"),
    ("Quality Score", f"{preds[1]:.1f}/10", "vs 4.9 avg"),
    ("Deep Sleep",    f"{preds[2]:.1f}%",   "vs 20% avg"),
]

for i, (label, val, delta) in enumerate(metrics):
    with [m1, m2, m3][i]:
        st.markdown(f"""
        <div class="metric-container">
            <p style='color: #94a3b8; font-size: 0.9rem; margin:0;'>{label}</p>
            <h2 style='color: #f8fafc; margin: 0.2rem 0;'>{val}</h2>
            <p style='color: #6366f1; font-size: 0.8rem; margin:0;'>{delta}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# -- SHAP Interpretability --
st.subheader("🔍 Feature Impact Analysis")
t1, t2 = st.tabs(["Visualization", "Raw Data"])

with t1:
    st.markdown("<p style='color: #94a3b8; margin-bottom: 1rem;'>Top factors " \
    "influencing your sleep today (Green = Positive impact, Red = Negative)</p>", 
    unsafe_allow_html=True)

    # Tab order: Quality, Duration, Deep Sleep
    # Model output order (CONTINUOUS_OUTCOMES): duration=0, quality=1, deep_sleep=2
    inner_tabs = st.tabs(["Quality", "Duration", "Deep Sleep"])
    tab_to_outcome = ['sleep_quality_score', 'sleep_duration_hrs', 'deep_sleep_percentage']

    for i, tab in enumerate(inner_tabs):
        with tab:
            outcome_idx = CONTINUOUS_OUTCOMES.index(tab_to_outcome[i])
            outcome_shap = shap_vals[0, :, outcome_idx]

            shap_df = pd.DataFrame({
                'Feature': [FEATURE_LABELS.get(f, f) for f in feature_cols],
                'Impact': outcome_shap
            })

            shap_df['Abs_Impact'] = shap_df['Impact'].abs()
            top_features = shap_df.nlargest(8, 'Abs_Impact').sort_values('Impact', ascending=True)

            fig_impact = go.Figure(go.Bar(
                x=top_features['Impact'],
                y=top_features['Feature'],
                orientation='h',
                marker_color=['#0d9488' if val > 0 else '#dc2626' for val in top_features['Impact']],
                text=[f"{v:+.2f}" for v in top_features['Impact']],
                textposition='auto',
            ))

            fig_impact.update_layout(
                **CHART_LAYOUT,
                height=350,
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="#334155", zerolinecolor="#94a3b8"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_impact, use_container_width=True)

with t2:
    st.markdown("### Feature Importance Table")
    raw_shap_df = pd.DataFrame(
        shap_vals[0],
        index=feature_cols,
        columns=[OUTCOME_LABELS[o] for o in CONTINUOUS_OUTCOMES]
    )
    st.dataframe(raw_shap_df.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)

st.divider()

st.subheader("Personalized Improvements")

# Most impactful features for Sleep Quality (CONTINUOUS_OUTCOMES index 1)
quality_idx = CONTINUOUS_OUTCOMES.index('sleep_quality_score')
quality_shap = pd.Series(shap_vals[0, :, quality_idx], index=feature_cols)
worst_factor = quality_shap.idxmin()
best_factor = quality_shap.idxmax()

c1, c2 = st.columns(2)

with c1:
    st.info(f"**Quick Win:** Your **{FEATURE_LABELS.get(best_factor, best_factor)}** is currently your strongest sleep booster. Keep it up!")

with c2:
    st.warning(f"**Focus Area:** Your **{FEATURE_LABELS.get(worst_factor, worst_factor)}** is hurting your sleep quality the most right now. Consider adjusting this for a better night.")

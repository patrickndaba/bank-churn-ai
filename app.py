import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="Churn AI | Intelligence Portal", layout="wide", page_icon="📈")

# --- CUSTOM PROFESSIONAL "MIDNIGHT EMERALD" UI ---
st.markdown("""
    <style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    :root {
        --primary: #10b981; /* Emerald */
        --primary-dark: #059669;
        --bg-deep: #0f172a; /* Deep Navy */
        --bg-card: rgba(30, 41, 59, 0.7); /* Slate Glass */
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    html, body, [class*="css"], .stMarkdown, label, p, div {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 13px !important;
        color: var(--text-main);
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }

    /* --- SIDEBAR REDESIGN --- */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem !important;
    }

    /* Sidebar Label Styling - High Contrast Tags */
    section[data-testid="stSidebar"] label {
        color: #0f172a !important; /* Deep Navy Font */
        background-color: #10b981 !important; /* Emerald Background */
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 10px !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 8px !important;
        display: inline-block !important;
        text-transform: uppercase !important;
    }

    /* Sidebar Selectbox & Inputs spacing */
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 12px !important;
    }
    
    /* Title & Typography */
    h1 {
        background: linear-gradient(90deg, #f8fafc, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: var(--text-dim);
        font-size: 13px !important;
        margin-bottom: 25px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Premium Glass Cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Inputs - Emerald Glow */
    .stNumberInput input, .stSelectbox div[role="combobox"], .stTextInput input {
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    
    /* Dropbox Hover - Black */
    .stSelectbox div[role="combobox"]:hover {
        background-color: #000000 !important;
        border-color: var(--primary) !important;
        cursor: pointer;
    }

    /* Gray Background for specific area (Active Member) */
    .active-member-container {
        background-color: #334155 !important; /* Slate Gray */
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .stNumberInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-weight: 800 !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #059669, #10b981) !important;
    }

    /* Unique Footer */
    .footer {
        margin-top: 60px;
        padding: 30px;
        text-align: center;
        color: var(--text-dim);
        font-size: 11px !important;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_stacking_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return model, scaler, feature_cols

def preprocess_input(input_data, scaler, feature_cols):
    df = pd.DataFrame([input_data])
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
    df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
    df['Is_Senior'] = (df['Age'] > 50).astype(int)
    df['Active_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df = pd.get_dummies(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    scaled_df = scaler.transform(df)
    return scaled_df, df

# Sidebar Header
with st.sidebar:
    st.markdown("<h2 style='color:#10b981; margin-bottom:0;'>ENGINE 01</h2>", unsafe_allow_html=True)
    st.caption("PREDICTIVE INTELLIGENCE UNIT")
    st.markdown("---")
    
    input_data = {
        'CreditScore': st.slider("CREDIT SCORE", 300, 850, 600),
        'Age': st.slider("CUSTOMER AGE", 18, 100, 35),
        'Tenure': st.slider("TENURE (YEARS)", 0, 10, 3),
        'Balance': st.number_input("ACCOUNT BALANCE", min_value=0.0, value=50000.0),
        'NumOfProducts': st.selectbox("PRODUCTS USED", [1, 2, 3, 4]),
        'HasCrCard': st.checkbox("CREDIT CARD HOLDER", value=True),
    }
    
    # Active Member with Gray Background
    with st.container():
        st.markdown('<div class="active-member-container">', unsafe_allow_html=True)
        input_data['IsActiveMember'] = st.checkbox("ACTIVE MEMBER", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Remaining inputs
    input_data.update({
        'EstimatedSalary': st.number_input("ESTIMATED SALARY", min_value=0.0, value=45000.0),
        'Geography': st.selectbox("REGION", ["France", "Germany", "Spain"]),
        'Gender': st.selectbox("GENDER", ["Female", "Male"])
    })

# Data Mapping
input_data[f"Geography_{input_data['Geography']}"] = 1
input_data[f"Gender_{input_data['Gender']}"] = 1
del input_data['Geography']
del input_data['Gender']

# Model Ops
model, scaler, feature_cols = load_artifacts()
scaled_input, raw_input_df = preprocess_input(input_data, scaler, feature_cols)
churn_prob = model.predict_proba(scaled_input)[:, 1][0]
risk_score = churn_prob * 100

# --- DASHBOARD HEADER ---
st.markdown("<h1>PREDICTIVE INTELLIGENCE PORTAL</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Customer Retention & Risk Mitigation AI</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 0.7])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>CORE ANALYTICS</h3>", unsafe_allow_html=True)
    
    if risk_score > 70:
        st.error(f"CRITICAL RISK: {risk_score:.1f}% PROBABILITY")
    elif risk_score > 40:
        st.warning(f"ELEVATED RISK: {risk_score:.1f}% PROBABILITY")
    else:
        st.success(f"STABLE ASSET: {risk_score:.1f}% PROBABILITY")
        
    st.progress(churn_prob)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>KPI UNIT</h3>", unsafe_allow_html=True)
    st.metric("RISK INDEX", f"{risk_score:.1f}%", f"{risk_score-25:.1f}%", delta_color="inverse")
    st.caption("Real-time variance based on current inputs.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- AI INSIGHT ENGINE (SHAP) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧠 STRATEGIC INTELLIGENCE: NEURAL DECOMPOSITION")

try:
    xgb_base_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_base_model)
    shap_vals = explainer.shap_values(scaled_input)
    
    # Visual Guide for Dark Theme
    guide_col1, guide_col2 = st.columns(2)
    with guide_col1:
        st.markdown("""
            <div style="background-color: rgba(239, 68, 68, 0.15); padding: 15px; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.3);">
                <b style="color: #f87171;">🔴 RISK ACCELERATORS</b><br>
                <span style="font-size: 12px; color: #cbd5e1;">Features pushing the prediction towards churn.</span>
            </div>
        """, unsafe_allow_html=True)
    with guide_col2:
        st.markdown("""
            <div style="background-color: rgba(16, 185, 129, 0.15); padding: 15px; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
                <b style="color: #34d399;">🟢 RETENTION ANCHORS</b><br>
                <span style="font-size: 12px; color: #cbd5e1;">Features stabilizing the customer relationship.</span>
            </div>
        """, unsafe_allow_html=True)

    # The Plot
    st_shap(shap.force_plot(explainer.expected_value, shap_vals[0], raw_input_df, link="logit", text_rotation=0, plot_cmap=["#10b981", "#ef4444"]), height=200)

    # Dynamic Insights
    feature_names = raw_input_df.columns
    contributions = pd.DataFrame({'Feature': feature_names, 'Influence': shap_vals[0]}).sort_values(by='Influence', ascending=False)

    top_risk = contributions.iloc[0]
    top_strength = contributions.iloc[-1]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### ⚡ ACTIONABLE INTELLIGENCE")
    i_col1, i_col2 = st.columns(2)
    with i_col1:
        st.write(f"🚩 **CRITICAL DRIVER:** `{top_risk['Feature']}`")
        st.caption("Primary catalyst for churn risk in this profile.")
    with i_col2:
        if top_strength['Influence'] < 0:
            st.write(f"🛡️ **RETENTION ANCHOR:** `{top_strength['Feature']}`")
            st.caption("Strongest factor preventing profile escalation.")
    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.info("System initializing decision intelligence...")

# FOOTER
st.markdown(f"""
    <div class="footer">
        PREDICTIVE INTELLIGENCE PORTAL v1.0 | Developed by Patrick Ndabarishye 
    </div>
""", unsafe_allow_html=True)

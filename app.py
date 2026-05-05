import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="Churn AI | Intelligence Portal", layout="wide", page_icon="📈")

# --- CUSTOM PROFESSIONAL "IVORY & ARCTIC" LIGHT UI ---
st.markdown("""
    <style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    :root {
        --primary: #0f172a; /* Deep Navy */
        --accent: #10b981; /* Emerald */
        --bg-main: #f8fafc; /* Arctic White */
        --bg-sidebar: #ffffff;
        --text-main: #1e293b;
        --text-dim: #64748b;
        --card-bg: rgba(255, 255, 255, 0.8);
    }

    html, body, [class*="css"], .stMarkdown, label, p, div {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 13px !important;
        color: var(--text-main);
    }

    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }

    /* --- SIDEBAR REDESIGN (LIGHT MODE) --- */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem !important;
    }

    /* Sidebar Label Styling - Professional Slate */
    section[data-testid="stSidebar"] label {
        color: var(--primary) !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        letter-spacing: 1px !important;
        margin-bottom: 6px !important;
        display: block !important;
        text-transform: uppercase !important;
    }

    /* Inputs - Clean Ivory Style */
    .stNumberInput input, .stTextInput input, .stSelectbox div[role="combobox"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 13px !important;
        padding: 8px !important;
        transition: all 0.2s ease;
    }
    
    .stSelectbox div[role="combobox"]:hover {
        background-color: #f8fafc !important;
        border-color: var(--accent) !important;
    }

    /* Active Member Container - Light Integration */
    .active-member-container {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--accent);
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 15px;
    }

    /* Title & Typography */
    h1 {
        color: var(--primary);
        font-size: 30px !important;
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

    /* Premium Glass Cards (Light) */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-weight: 800 !important;
    }

    /* Footer */
    .footer {
        margin-top: 60px;
        padding: 30px;
        text-align: center;
        color: var(--text-dim);
        font-size: 11px !important;
        border-top: 1px solid #e2e8f0;
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
    st.markdown("<h2 style='color:#0f172a; margin-bottom:0;'>ENGINE 01</h2>", unsafe_allow_html=True)
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
    
    # Active Member with Professional Integration
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
    st.markdown("<h3 style='margin-top:0; color:#0f172a;'>CORE ANALYTICS</h3>", unsafe_allow_html=True)
    
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
    st.markdown("<h3 style='margin-top:0; color:#0f172a;'>KPI UNIT</h3>", unsafe_allow_html=True)
    st.metric("RISK INDEX", f"{risk_score:.1f}%", f"{risk_score-25:.1f}%", delta_color="inverse")
    st.caption("Real-time variance based on current profile.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- AI INSIGHT ENGINE (SHAP) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#0f172a;'>🧠 STRATEGIC INTELLIGENCE: NEURAL DECOMPOSITION</h3>", unsafe_allow_html=True)

try:
    xgb_base_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_base_model)
    shap_vals = explainer.shap_values(scaled_input)
    
    if isinstance(shap_vals, list):
        display_shap_vals = shap_vals[1]
        base_value = explainer.expected_value[1]
    else:
        display_shap_vals = shap_vals
        base_value = explainer.expected_value

    # Visual Guide for Light Theme
    guide_col1, guide_col2 = st.columns(2)
    with guide_col1:
        st.markdown("""
            <div style="background-color: #fee2e2; padding: 15px; border-radius: 12px; border-left: 5px solid #ef4444;">
                <b style="color: #b91c1c;">🔴 RISK ACCELERATORS</b><br>
                <span style="font-size: 12px; color: #7f1d1d;">Factors increasing churn probability.</span>
            </div>
        """, unsafe_allow_html=True)
    with guide_col2:
        st.markdown("""
            <div style="background-color: #dcfce7; padding: 15px; border-radius: 12px; border-left: 5px solid #10b981;">
                <b style="color: #14532d;">🟢 RETENTION ANCHORS</b><br>
                <span style="font-size: 12px; color: #14532d;">Factors stabilizing the relationship.</span>
            </div>
        """, unsafe_allow_html=True)

    # The Plot
    st_shap(shap.force_plot(base_value, display_shap_vals[0], raw_input_df, link="logit", text_rotation=0), height=200)

    # Dynamic Insights
    feature_names = raw_input_df.columns
    contributions = pd.DataFrame({'Feature': feature_names, 'Influence': display_shap_vals[0]}).sort_values(by='Influence', ascending=False)
    top_risk = contributions.iloc[0]
    top_strength = contributions.iloc[-1]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#0f172a; margin-top:0;'>⚡ ACTIONABLE INTELLIGENCE</h4>", unsafe_allow_html=True)
    i_col1, i_col2 = st.columns(2)
    with i_col1:
        st.write(f"🚩 **CRITICAL DRIVER:** `{top_risk['Feature']}`")
        st.caption("Primary factor currently driving risk upward.")
    with i_col2:
        if top_strength['Influence'] < 0:
            st.write(f"🛡️ **RETENTION ANCHOR:** `{top_strength['Feature']}`")
            st.caption("Strongest factor currently maintaining stability.")
    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.info("System initializing decision intelligence...")

# FOOTER
st.markdown(f"""
    <div class="footer">
        PREDICTIVE INTELLIGENCE PORTAL v2.0 | Developed by Patrick Ndabarishye 
    </div>
""", unsafe_allow_html=True)

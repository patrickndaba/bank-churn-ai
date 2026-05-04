import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="Churn AI | Intelligence Dashboard", layout="wide", page_icon="📈")

# --- CUSTOM HIGH-END SEAMLESS GLASSMORPHISM UI ---
st.markdown("""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"], .stMarkdown, label, p, div {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: #1e293b; /* Deep Slate Text */
    }

    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }

    /* --- SIDEBAR (SLIDING PART) - SEAMLESS & LIGHT --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebarUserContent"] {
        background-color: transparent !important;
        padding-top: 2rem !important;
    }

    /* Dark Slate Text for Sidebar (Light Theme) */
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: #1e293b !important;
    }
    
    /* Title Styling */
    h1 {
        color: #0f172a;
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0px !important;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 14px !important;
        margin-bottom: 20px;
    }

    /* Premium Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }

    /* Inputs - Emerald Accents */
    .stNumberInput input, .stSelectbox div[role="combobox"], .stTextInput input {
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        background-color: white !important;
    }

    /* Progress Bar - Emerald */
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }

    /* Unique Footer */
    .footer {
        margin-top: 50px;
        padding: 20px;
        text-align: center;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-weight: 500;
        font-size: 12px !important;
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
    st.markdown("### 📊 Churn Engine")
    st.caption("Predictive Intelligence Portal")
    st.markdown("---")
    
    input_data = {
        'CreditScore': st.slider("Credit Score", 300, 850, 600),
        'Age': st.slider("Customer Age", 18, 100, 35),
        'Tenure': st.slider("Tenure (Years)", 0, 10, 3),
        'Balance': st.number_input("Account Balance", min_value=0.0, value=50000.0),
        'NumOfProducts': st.selectbox("Products Used", [1, 2, 3, 4]),
        'HasCrCard': st.checkbox("Credit Card Holder", value=True),
        'IsActiveMember': st.checkbox("Active Member", value=True),
        'EstimatedSalary': st.number_input("Estimated Salary", min_value=0.0, value=45000.0),
        'Geography': st.selectbox("Region", ["France", "Germany", "Spain"]),
        'Gender': st.selectbox("Gender", ["Female", "Male"])
    }

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
st.markdown("<h1>BANK CHURN AI | Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Predictive Analytics for Customer Retention</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 0.7])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Analysis")
    
    if risk_score > 70:
        st.error(f"⚠️ **CRITICAL RISK PROFILE**: {risk_score:.1f}% Churn Probability")
    elif risk_score > 40:
        st.warning(f"🔍 **MODERATE RISK PROFILE**: {risk_score:.1f}% Churn Probability")
    else:
        st.success(f"✨ **STABLE PROFILE**: {risk_score:.1f}% Churn Probability")
        
    st.progress(churn_prob)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Metrics")
    st.metric("Risk Index", f"{risk_score:.1f}%", f"{risk_score-25:.1f}%", delta_color="inverse")
    st.caption("Drivers: Account activity and balance ratios.")
    st.markdown('</div>', unsafe_allow_html=True)

# SHAP SECTION
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧠 AI Neural Insights (SHAP)")
st.info("Technical Explanation: Red features increase risk, Blue features decrease risk. Powered by Stacking Ensemble (XGB+RF+Cat).")

try:
    xgb_base_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_base_model)
    shap_values = explainer.shap_values(scaled_input)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], raw_input_df), height=180)
except Exception as e:
    st.error("AI Insight engine initializing...")

# FOOTER
st.markdown(f"""
    <div class="footer">
        Developed by Patrick Ndabarishye 
    </div>
""", unsafe_allow_html=True)

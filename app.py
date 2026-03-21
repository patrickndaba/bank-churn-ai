import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="BK Churn AI | Customer Retention", layout="wide", page_icon="🏦")

# --- CUSTOM COMPACT CSS FOR BANK OF KIGALI BRANDING ---
st.markdown("""
    <style>
    /* Global Font & Size Settings */
    html, body, [class*="css"], .stMarkdown, label, p, div {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 14px !important; /* Reduced from default */
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #005DAA !important;
        color: white !important;
        min-width: 250px !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: white !important;
        font-size: 13px !important;
    }
    
    /* White Sliders Styling */
    div[data-testid="stThumbValue"] {
        color: #005DAA !important;
        font-weight: bold;
    }
    div[data-baseweb="slider"] > div > div {
        background-color: #ffffff !important; /* White bar */
        border: 1px solid #005DAA;
        height: 8px;
    }

    /* Small Input Boxes with Colored Borders */
    .stNumberInput input, .stSelectbox div[role="combobox"], .stTextInput input {
        border: 1.5px solid #005DAA !important;
        border-radius: 5px !important;
        padding: 5px !important;
        height: 35px !important; /* Compact height */
        font-size: 13px !important;
    }

    /* Header & Title */
    h1 {
        color: #005DAA;
        font-size: 24px !important; /* Smaller Header */
        font-weight: 800;
        border-bottom: 2px solid #FFC72C;
        padding-bottom: 5px;
    }
    
    h2 { font-size: 20px !important; color: #005DAA; }
    h3 { font-size: 18px !important; color: #005DAA; }

    /* Prediction Card Styling (Compact) */
    .prediction-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 6px solid #005DAA;
        margin-bottom: 10px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 10px;
        font-size: 11px !important;
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

# Sidebar Interface
with st.sidebar:
    st.image("https://www.bk.rw/themes/bk/logo.png", width=150)
    st.markdown("### Customer Intelligence Portal")
    st.markdown("---")
    
    input_data = {
        'CreditScore': st.slider("Credit Score", 300, 850, 600),
        'Age': st.slider("Customer Age", 18, 100, 35),
        'Tenure': st.slider("Years with Bank", 0, 10, 3),
        'Balance': st.number_input("Account Balance (RWF)", min_value=0.0, value=1500000.0),
        'NumOfProducts': st.selectbox("Products Used", [1, 2, 3, 4]),
        'HasCrCard': st.checkbox("Credit Card Holder", value=True),
        'IsActiveMember': st.checkbox("Active Member Account", value=True),
        'EstimatedSalary': st.number_input("Est. Monthly Income (RWF)", min_value=0.0, value=800000.0),
        'Geography': st.selectbox("Market Region", ["France", "Germany", "Spain"]),
        'Gender': st.selectbox("Gender", ["Female", "Male"])
    }

input_data[f"Geography_{input_data['Geography']}"] = 1
input_data[f"Gender_{input_data['Gender']}"] = 1
del input_data['Geography']
del input_data['Gender']

model, scaler, feature_cols = load_artifacts()
scaled_input, raw_input_df = preprocess_input(input_data, scaler, feature_cols)
churn_prob = model.predict_proba(scaled_input)[:, 1][0]
risk_score = churn_prob * 100

# --- MAIN DASHBOARD ---
st.title("🏦 BANK OF KIGALI | AI Churn Risk Dashboard")
st.markdown("#### Predictive Customer Intelligence System")

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("AI Analysis: Probability")
    
    if risk_score > 70:
        st.error(f"🚨 **CRITICAL RISK**: {risk_score:.1f}%")
    elif risk_score > 40:
        st.warning(f"⚠️ **MODERATE RISK**: {risk_score:.1f}%")
    else:
        st.success(f"✅ **LOW RISK**: {risk_score:.1f}%")
        
    st.progress(churn_prob)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Customer Metrics")
    st.metric("Risk Score", f"{risk_score:.1f}%", delta=f"{risk_score-20:.1f}%", delta_color="inverse")
    st.caption("Primary drivers: Balance-to-Salary ratio and Engagement levels.")

# SHAP SECTION
st.markdown("---")
st.subheader("🔍 AI Intelligence (SHAP)")
try:
    xgb_base_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_base_model)
    shap_values = explainer.shap_values(scaled_input)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], raw_input_df), height=180)
except Exception as e:
    st.error("AI Insight temporarily unavailable.")

st.markdown('<div class="footer">© 2026 Bank of Kigali AI Labs | Sem 3 Project</div>', unsafe_allow_html=True)

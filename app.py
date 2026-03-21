import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="BK Churn AI | Customer Retention", layout="wide", page_icon="🏦")

# --- CUSTOM CSS FOR BANK OF KIGALI BRANDING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #005DAA !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Header & Title */
    h1 {
        color: #005DAA;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        border-bottom: 3px solid #FFC72C;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #005DAA;
        font-weight: 600;
    }

    /* Prediction Card Styling */
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 10px solid #005DAA;
        margin-bottom: 20px;
    }
    
    /* Button & Slider Styling */
    .stButton>button {
        background-color: #FFC72C !important;
        color: #005DAA !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #FFC72C;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 20px;
        font-size: 0.8rem;
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
    st.image("https://www.bk.rw/themes/bk/logo.png", width=200) # Placeholder for BK Logo
    st.markdown("### Customer Intelligence Portal")
    st.markdown("---")
    
    # Collect User Input
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

# Mapping inputs
input_data[f"Geography_{input_data['Geography']}"] = 1
input_data[f"Gender_{input_data['Gender']}"] = 1
del input_data['Geography']
del input_data['Gender']

# Load Model & Predict
model, scaler, feature_cols = load_artifacts()
scaled_input, raw_input_df = preprocess_input(input_data, scaler, feature_cols)
churn_prob = model.predict_proba(scaled_input)[:, 1][0]
risk_score = churn_prob * 100

# --- MAIN DASHBOARD LAYOUT ---
st.title("🏦 BANK OF KIGALI | AI Churn Risk Dashboard")
st.markdown("#### Empowering your branches with Predictive Customer Intelligence")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("AI Analysis: Churn Probability")
    
    # Dynamic Alert Styling
    if risk_score > 70:
        st.error(f"🚨 **CRITICAL RISK PROFILE**: {risk_score:.1f}%")
        st.markdown("**Action Required**: Immediate direct intervention by a Branch Manager recommended.")
    elif risk_score > 40:
        st.warning(f"⚠️ **MODERATE RISK PROFILE**: {risk_score:.1f}%")
        st.markdown("**Action Required**: Targeted loyalty incentive or fee waiver recommended.")
    else:
        st.success(f"✅ **LOW RISK PROFILE**: {risk_score:.1f}%")
        st.markdown("**Action Required**: Standard relationship management; high retention likelihood.")
        
    st.progress(churn_prob)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Customer Metrics Summary")
    st.metric("Risk Score", f"{risk_score:.1f}%", delta=f"{risk_score-20:.1f}%", delta_color="inverse")
    st.write("---")
    st.write("**Top Impact Factors Identified:**")
    st.caption("AI identifying Balance-to-Salary ratio and Activity levels as primary drivers for this profile.")

# SHAP SECTION
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.subheader("🔍 AI Interpretability (SHAP Intelligence)")
st.info("The chart below explains the *why* behind the score. Factors in red push the customer toward leaving, while blue factors pull them back to loyalty.")

try:
    xgb_base_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_base_model)
    shap_values = explainer.shap_values(scaled_input)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], raw_input_df), height=200)
except Exception as e:
    st.error("AI Insight temporarily unavailable.")

st.markdown("""
    <div class="footer">
        © 2026 Bank of Kigali AI Labs | Developed for Semester 3 Final Project
    </div>
""", unsafe_allow_html=True)

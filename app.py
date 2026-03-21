import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="Bank Churn AI Dashboard", layout="wide")

# Helper Functions
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_stacking_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return model, scaler, feature_cols

def preprocess_input(input_data, scaler, feature_cols):
    # Data preprocessing logic to align with training
    df = pd.DataFrame([input_data])
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
    df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
    df['Is_Senior'] = (df['Age'] > 50).astype(int)
    df['Active_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    
    # Feature One-Hot Encoding
    df = pd.get_dummies(df)
    
    # Align columns with training data (add missing columns with 0s)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Ensure correct column order
    df = df[feature_cols]
    
    # Standardize
    scaled_df = scaler.transform(df)
    return scaled_df, df

# Sidebar Header
st.sidebar.header("Customer Information Input")

# Load Model
model, scaler, feature_cols = load_artifacts()

# Collect User Input
input_data = {
    'CreditScore': st.sidebar.slider("Credit Score", 300, 850, 600),
    'Age': st.sidebar.slider("Age", 18, 100, 40),
    'Tenure': st.sidebar.slider("Tenure (Years)", 0, 10, 5),
    'Balance': st.sidebar.number_input("Balance ($)", min_value=0.0, value=50000.0),
    'NumOfProducts': st.sidebar.selectbox("Number of Products", [1, 2, 3, 4]),
    'HasCrCard': st.sidebar.checkbox("Has Credit Card?", value=True),
    'IsActiveMember': st.sidebar.checkbox("Is Active Member?", value=True),
    'EstimatedSalary': st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0),
    'Geography': st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"]),
    'Gender': st.sidebar.selectbox("Gender", ["Female", "Male"])
}

# Mapping Geography/Gender to match encoded names
input_data[f"Geography_{input_data['Geography']}"] = 1
input_data[f"Gender_{input_data['Gender']}"] = 1
del input_data['Geography']
del input_data['Gender']

# Preprocess and Predict
scaled_input, raw_input_df = preprocess_input(input_data, scaler, feature_cols)
churn_prob = model.predict_proba(scaled_input)[:, 1][0]
risk_score = churn_prob * 100

# App Layout
st.title("🏦 Bank Customer Churn Prediction AI")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Prediction Result")
    
    # Risk Scoring Logic
    if risk_score > 70:
        st.error(f"**CRITICAL RISK**: {risk_score:.2f}% Probability of Churn")
    elif risk_score > 40:
        st.warning(f"**MODERATE RISK**: {risk_score:.2f}% Probability of Churn")
    else:
        st.success(f"**LOW RISK**: {risk_score:.2f}% Probability of Churn")
        
    st.progress(churn_prob)
    
    st.info("Recommendations: Targeted retention offers, personalized support, or loyalty rewards for high-risk customers.")

with col2:
    st.subheader("Key Predictive Features")
    st.write(raw_input_df)

# SHAP Interpretability
st.markdown("---")
st.subheader("🔍 Explainability (SHAP Value)")

# We use the XGBoost component for SHAP as meta-models (Stacking) are hard to explain directly
# but XGBoost provides the strongest signal.
try:
    xgb_base_model = model.named_estimators_['xgb']
    explainer = joblib.load("xgb_explainer.joblib") if False else shap.TreeExplainer(xgb_base_model)
    shap_values = explainer.shap_values(scaled_input)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], raw_input_df), height=200)
    st.caption("SHAP Force Plot: Factors in red push the risk higher, blue factors pull the risk lower.")
except Exception as e:
    st.error(f"Error generating SHAP plot: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys
sys.path.append("src")
from preprocess import engineer_features, encode_data

st.set_page_config(
    page_title="Churn Predictor | Vinod Tenagi",
    page_icon="📉",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center; color:#00D9FF'>📉 Customer Churn Predictor</h1>
    <p style='text-align:center; color:gray'>Real-time churn risk scoring powered by XGBoost</p>
    <hr>
""", unsafe_allow_html=True)

st.sidebar.header("🔧 Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20, 150, 70)
total_charges = monthly_charges * tenure
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
support_tickets = st.sidebar.slider("Support Tickets (last 3 months)", 0, 10, 1)

input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "PaymentMethod": payment,
    "TechSupport": tech_support,
    "NumSupportTickets": support_tickets
}])

input_data = engineer_features(input_data)
input_data = encode_data(input_data)

if "Churn" in input_data.columns:
    input_data.drop(columns=["Churn"], inplace=True)

col1, col2, col3 = st.columns(3)

try:
    model = joblib.load("models/xgboost_churn.pkl")
    churn_prob = model.predict_proba(input_data)[0][1]
    churn_pred = model.predict(input_data)[0]

    with col1:
        st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
    with col2:
        st.metric("Monthly Charges", f"${monthly_charges}")
    with col3:
        st.metric("Tenure", f"{tenure} months")

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(churn_prob * 100, 1),
        title={"text": "Churn Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00D9FF"},
            "steps": [
                {"range": [0, 30], "color": "#1a472a"},
                {"range": [30, 60], "color": "#f4a261"},
                {"range": [60, 100], "color": "#e63946"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    if churn_pred == 1:
        st.error("⚠️ HIGH RISK — This customer is likely to churn! Immediate retention action recommended.")
    else:
        st.success("✅ LOW RISK — This customer is likely to stay.")

except Exception as e:
    st.warning("⚠️ Model not found. Please run src/train.py first to generate the model.")
    st.info("Run: `python src/train.py`")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center; color:gray'>
    Built by <a href='https://www.linkedin.com/in/vinod-tenagi-a485ba392/' target='_blank'>Vinod Tenagi</a>
    | Data Analytics Engineer | Bangalore
    </p>
""", unsafe_allow_html=True)

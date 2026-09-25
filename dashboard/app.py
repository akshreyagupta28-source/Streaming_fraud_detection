import streamlit as st
import pandas as pd
import joblib
import os
 
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="",
    layout="wide"
)
 
DATA_PATH = "dataset/processed/paysim_processed.csv"
MODEL_PATH = "models/xgboost.pkl"
CM_IMAGE_PATH = "result/xgboost_confusion_matrix.png"
 
model = joblib.load(MODEL_PATH)
 
df = pd.read_csv(
    DATA_PATH,
    nrows=1000
)
 
feature_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change_orig",
    "balance_change_dest",
    "amount_to_orig_balance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]
 
st.title("Real-Time Fraud Detection System")
st.caption("Streaming-based financial transaction fraud detection using XGBoost.")
 
# --- Sidebar controls ---
st.sidebar.header("Controls")
 
threshold = st.sidebar.slider(
    "Fraud probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)
 
show_fraud_only = st.sidebar.checkbox("Show only fraud transactions")
 
# --- Run predictions ---
X = df[feature_columns]
 
df["fraud_probability"] = model.predict_proba(X)[:, 1]
 
df["decision"] = df["fraud_probability"].apply(
    lambda p: "FRAUD" if p >= threshold else "LEGITIMATE"
)
 
# --- Top metrics ---
total_transactions = len(df)
fraud_count = (df["decision"] == "FRAUD").sum()
legit_count = total_transactions - fraud_count
fraud_pct = (fraud_count / total_transactions) * 100
 
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total_transactions)
col2.metric("Fraud Detected", fraud_count)
col3.metric("Legitimate", legit_count)
col4.metric("Fraud %", f"{fraud_pct:.2f}%")
 
# --- Charts ---
chart_col1, chart_col2 = st.columns(2)
 
with chart_col1:
    st.subheader("Decision Breakdown")
    st.bar_chart(df["decision"].value_counts())
 
with chart_col2:
    st.subheader("Fraud Count by Step")
    fraud_by_step = df[df["decision"] == "FRAUD"].groupby("step").size()
    st.line_chart(fraud_by_step)
 
# --- Transactions table ---
st.subheader("Recent Transactions")
 
display_df = df[["step", "amount", "fraud_probability", "decision"]]
 
if show_fraud_only:
    display_df = display_df[display_df["decision"] == "FRAUD"]
 
 
def highlight_fraud(row):
    color = "background-color: #ff4b4b" if row["decision"] == "FRAUD" else ""
    return [color] * len(row)
 
 
st.dataframe(
    display_df.style.apply(highlight_fraud, axis=1),
    use_container_width=True
)
 
# --- Model performance ---
st.subheader("Model Performance")
 
if os.path.exists(CM_IMAGE_PATH):
    st.image(CM_IMAGE_PATH, caption="XGBoost Confusion Matrix", width=500)
else:
    st.info("Confusion matrix image not found at result/xgboost_confusion_matrix.png")
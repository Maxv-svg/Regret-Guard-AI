import streamlit as st
import joblib
import pandas as pd
import time

# UI CONFIGURATION
st.set_page_config(page_title="Revolut AI Guard", layout="wide")

# Revolut Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stButton>button { border-radius: 20px; height: 3em; background-color: #0075FF; color: white; border: none; font-weight: bold; }
    .stMetric { background-color: #1c1c1e; padding: 20px; border-radius: 15px; border: 1px solid #2c2c2e; }
    </style>
    """, unsafe_allow_html=True)

# LOAD PRE-TRAINED BRAIN
bundle = joblib.load('data/regret_bundle.pkl')
model, features = bundle['model'], bundle['features']

if 'vault' not in st.session_state: st.session_state.vault = []

# SIDEBAR: REVOLUT NAV
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/Revolut_logo.png", width=120)
    st.write("### Maximilian")
    st.divider()
    balance = st.number_input("Balance (€)", value=2100.0, step=100.0)
    st.caption(f"AI Integrity: {100-bundle['mae']:.1f}%")

# MAIN FEATURE: THE CHECKOUT INTERVENTION
st.title("🛡️ AI Transaction Guard")
st.write("Transaction pending: **Sneakers / StockX**")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Checkout Context")
    with st.container(border=True):
        price = st.number_input("Amount (€)", value=299.0)
        risk = st.select_slider("Merchant Return Risk", options=[0.05, 0.15, 0.30, 0.55], value=0.15)
        fomo = st.toggle("Flash Sale / Drop")
        mood = st.slider("Mood Stability", 1, 10, 4)
        sleep = st.slider("Sleep (Hours)", 3.0, 12.0, 5.5)

with col_right:
    st.subheader("Security Analysis")
    if st.button("Run AI Verification", type="primary", use_container_width=True):
        with st.spinner('Calculating emotional and financial impact...'):
            time.sleep(1)
            
            # Match engineered features from CSV
            rel_price = price / balance
            imp_index = ((11 - mood) * 0.3 + (11 - sleep) * 0.3 + (int(fomo) * 1.5))
            
            inputs = pd.DataFrame([[price, balance, mood, int(fomo), sleep, risk, rel_price, imp_index]], columns=features)
            score = model.predict(inputs)[0]
            
            st.metric("Probability of Regret", f"{score:.1f}%")
            
            # XAI: Feature Importance
            st.write("**Why did AI intervene?**")
            st.bar_chart(pd.Series(model.feature_importances_, index=features))

            if score > 65:
                st.error("🚨 **High Risk Intercepted**")
                st.info(f"Reason: This purchase impacts {rel_price*100:.1f}% of your liquid funds.")
                if st.button("Move to Cooling Vault"):
                    st.session_state.vault.append({"Item": "Sneakers", "Price": price, "Risk": f"{score:.1f}%"})
                    st.toast("Security pause active.")
            else:
                st.success("✅ **Approved by Guard**")

if st.session_state.vault:
    st.divider()
    st.subheader("📂 Cooling Vault")
    st.table(st.session_state.vault)
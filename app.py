import streamlit as st
import joblib
import pandas as pd
import time

# 1. PAGE CONFIG & PREMIUM STYLING
st.set_page_config(page_title="Revolut", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Revolut Dark Mode
st.markdown("""
    <style>
    .main { background-color: #000000; color: #FFFFFF; }
    .stButton>button { border-radius: 12px; height: 3.5em; background-color: #0666EB; color: white; border: none; font-weight: bold; width: 100%; }
    .revolut-card { background-color: #1c1c1e; padding: 25px; border-radius: 20px; border: 1px solid #2c2c2e; margin-bottom: 20px; }
    .stMetric { background-color: #1c1c1e; padding: 15px; border-radius: 15px; border: 1px solid #2c2c2e; }
    div[data-testid="stExpander"] { border: none !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD THE AI BUNDLE
# This contains the model, feature list, and accuracy (MAE)
bundle = joblib.load('data/regret_bundle.pkl')
model, features = bundle['model'], bundle['features']

# 3. STATE MANAGEMENT
# We use st.session_state to handle the transaction flow
if 'app_state' not in st.session_state: st.session_state.app_state = 'dashboard'
if 'vault' not in st.session_state: st.session_state.vault = []

# SIDEBAR (Simulating the Profile/Settings)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/Revolut_logo.png", width=120)
    st.write("### Maximilian")
    balance = st.number_input("Current Balance (€)", value=2850.0)
    st.caption(f"AI Integrity: {100-bundle['mae']:.1f}%")

# --- DASHBOARD VIEW ---
if st.session_state.app_state == 'dashboard':
    st.title("Home")
    
    # Big Revolut-style Card
    st.markdown(f'<div class="revolut-card"><h3>Main Account</h3><h1>€ {balance:,.2f}</h1></div>', unsafe_allow_html=True)
    
    col_pay, col_vault = st.columns(2)
    with col_pay:
        if st.button("💸 New Online Payment"):
            st.session_state.app_state = 'checkout'
            st.rerun()
    with col_vault:
        if st.button("🛡️ Cooling Vault"):
            st.session_state.app_state = 'vault'
            st.rerun()

# --- CHECKOUT FLOW (The Feature) ---
elif st.session_state.app_state == 'checkout':
    st.title("Verify Transaction")
    st.write("Requested by: **StockX / Sneakers**")
    
    with st.container():
        st.markdown('<div class="revolut-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Payment Details")
            price = st.number_input("Amount (€)", value=320.0)
            merchant_risk = st.select_slider("Category Risk (Returns)", options=[0.05, 0.15, 0.30, 0.55], value=0.15)
            fomo = st.toggle("Limited Drop / FOMO?")
        
        with col2:
            st.subheader("Your Context")
            mood = st.slider("Mood Stability (1-10)", 1, 10, 4)
            sleep = st.slider("Sleep last night (h)", 3, 12, 6)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Confirm and Analyze", type="primary"):
        with st.spinner('Regret Guard AI is analyzing...'):
            time.sleep(1) # Simulate real-world API processing
            
            # FEATURE ENGINEERING (Matches train_model logic)
            rel_price = price / balance
            imp_index = ((11 - mood) * 0.3 + (11 - sleep) * 0.3 + (int(fomo) * 1.5))
            
            # Create input for the model
            inputs = pd.DataFrame([[price, balance, mood, int(fomo), sleep, merchant_risk, rel_price, imp_index]], columns=features)
            score = model.predict(inputs)[0]
            
            # Display Result (Evaluator Mode)
            st.divider()
            st.metric("Probability of Purchase Regret", f"{score:.1f}%")
            
            if score > 60:
                st.error("🚨 **High Risk Intervention**")
                st.write(f"This purchase accounts for **{rel_price*100:.1f}%** of your balance. Your current mood and lack of sleep suggest an impulsive pattern.")
                
                # XAI: Explain why
                st.bar_chart(pd.Series(model.feature_importances_, index=features))
                
                if st.button("❄️ Block & Move to Vault"):
                    st.session_state.vault.append({"Merchant": "StockX", "Price": f"{price}€", "Risk": f"{score:.1f}%"})
                    st.toast("Security pause active.")
                    st.session_state.app_state = 'dashboard'
                    st.rerun()
            else:
                st.success("✅ **Verified**")
                if st.button("Complete Transaction"):
                    st.session_state.app_state = 'dashboard'
                    st.rerun()
    
    if st.button("← Cancel Payment"):
        st.session_state.app_state = 'dashboard'
        st.rerun()

# --- VAULT VIEW ---
elif st.session_state.app_state == 'vault':
    st.title("Cooling Vault")
    st.write("Items stored for 24 hours to prevent impulsive decisions.")
    if st.session_state.vault:
        st.table(st.session_state.vault)
        if st.button("Clear Vault"):
            st.session_state.vault = []
            st.rerun()
    else:
        st.info("Vault is currently empty.")
    
    if st.button("← Back to Dashboard"):
        st.session_state.app_state = 'dashboard'
        st.rerun()
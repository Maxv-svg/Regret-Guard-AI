import streamlit as st
import joblib
import pandas as pd
import time

# --- 1. PREMIUM REVOLUT STYLING ---
st.set_page_config(page_title="Revolut Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .stButton>button { border-radius: 25px; height: 3.5em; background: linear-gradient(90deg, #0666EB 0%, #0047AB 100%); color: white; border: none; font-weight: 700; width: 100%; }
    .revolut-card { background-color: #1c1c1e; padding: 25px; border-radius: 24px; border: 1px solid #2c2c2e; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .metric-box { background-color: #1c1c1e; padding: 20px; border-radius: 18px; border: 1px solid #3a3a3c; text-align: center; }
    h1, h2, h3 { color: #FFFFFF; font-weight: 800; }
    .stSlider [data-baseweb="slider"] { margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATA ---
try:
    bundle = joblib.load('data/regret_bundle.pkl')
    model, features = bundle['model'], bundle['features']
except:
    st.error("Please run 'python train_model.py' first.")
    st.stop()

if 'step' not in st.session_state: st.session_state.step = 'dashboard'
if 'vault' not in st.session_state: st.session_state.vault = []

# --- 3. REVOLUT SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/Revolut_logo.png", width=120)
    st.write("### Maximilian")
    st.caption("Premium Member")
    balance = st.number_input("Current Balance (€)", value=2850.0)

# --- 4. STEP 1: DASHBOARD ---
if st.session_state.step == 'dashboard':
    st.title("Home")
    st.markdown(f'<div class="revolut-card"><p style="color:#8e8e93; margin-bottom:5px;">Main Account</p><h1>€ {balance:,.2f}</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if st.button("💸 New Online Payment"):
            st.session_state.step = 'checkout'
            st.rerun()
    with col2:
        if st.button("🛡️ Cooling Vault"):
            st.session_state.step = 'vault'
            st.rerun()

# --- 5. STEP 2: TRANSACTION VERIFICATION (The Simulation) ---
elif st.session_state.step == 'checkout':
    st.title("Verify Transaction")
    st.write("Merchant Request: **StockX / Sneakers**")
    
    with st.container():
        st.markdown('<div class="revolut-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("### Payment Details")
            price = st.number_input("Amount (€)", value=350.0)
            merchant_risk = st.select_slider("Category Reliability", options=[0.05, 0.15, 0.30, 0.55], value=0.15, format_func=lambda x: f"{int(x*100)}% Risk")
            fomo = st.toggle("Limited Release / Flash Sale")
        with c2:
            st.markdown("### Personal Context")
            mood = st.slider("Mood (1-10)", 1, 10, 4)
            sleep = st.slider("Sleep last night (h)", 3.0, 12.0, 5.5)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyze & Verify", type="primary"):
        with st.spinner('Analyzing financial health...'):
            time.sleep(1.2)
            rel_price = price / balance
            imp_index = ((11 - mood) * 0.3 + (11 - sleep) * 0.3 + (int(fomo) * 1.5))
            inputs = pd.DataFrame([[price, balance, mood, int(fomo), sleep, merchant_risk, rel_price, imp_index]], columns=features)
            score = model.predict(inputs)[0]
            
            st.divider()
            st.markdown(f'<div class="metric-box"><p style="color:#8e8e93;">Probability of Regret</p><h2 style="color:{"#FF4B4B" if score > 60 else "#00CC96"}">{score:.1f}%</h2></div>', unsafe_allow_html=True)
            
            if score > 60:
                st.error("🚨 **High Risk Intervention**")
                st.info(f"**AI Insight:** This purchase consumes {rel_price*100:.1f}% of your liquid balance. Combined with your current fatigue ({sleep}h sleep), our model detects a high probability of impulsive regret.")
                st.write("**Risk Drivers:**")
                st.bar_chart(pd.Series(model.feature_importances_, index=features))
                
                if st.button("❄️ Block & Save to Cooling Vault"):
                    st.session_state.vault.append({"Merchant": "StockX", "Amount": f"{price}€", "Risk": f"{score:.1f}%"})
                    st.session_state.step = 'dashboard'
                    st.rerun()
            else:
                st.success("✅ **Verified: Low Risk**")
                if st.button("Finalize Payment"):
                    st.session_state.step = 'dashboard'
                    st.rerun()
    
    if st.button("← Cancel Payment", key="back"): st.session_state.step = 'dashboard'; st.rerun()

elif st.session_state.step == 'vault':
    st.title("Cooling Vault")
    if st.session_state.vault:
        st.dataframe(pd.DataFrame(st.session_state.vault), use_container_width=True)
    else: st.info("Vault is empty.")
    if st.button("← Back to Dashboard"): st.session_state.step = 'dashboard'; st.rerun()
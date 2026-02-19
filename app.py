import streamlit as st
import joblib
import pandas as pd
import time

# 1. NATIVE MOBILE APP STYLING (The "Revolut" Skin)
st.set_page_config(page_title="Revolut Pro", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Main Background & Font */
    .main { background-color: #000000; color: #FFFFFF; font-family: 'Inter', -apple-system, sans-serif; }
    
    /* Hide Streamlit Header/Footer */
    header, footer {visibility: hidden;}
    
    /* Revolut Premium Card */
    .rev-card {
        background: linear-gradient(135deg, #1c1c1e 0%, #0a0a0a 100%);
        padding: 30px;
        border-radius: 28px;
        border: 1px solid #2c2c2e;
        margin-bottom: 25px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    }
    
    /* Action Buttons */
    .stButton>button {
        border-radius: 16px;
        height: 4em;
        background-color: #1977F3;
        color: white;
        border: none;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    /* Transaction List Item */
    .tx-item {
        display: flex;
        justify-content: space-between;
        padding: 15px 0;
        border-bottom: 0.5px solid #2c2c2e;
    }

    /* AI Risk Alert Overlay */
    .ai-alert {
        background-color: #1c1c1e;
        border-left: 5px solid #FF4B4B;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD THE "BRAIN"
try:
    bundle = joblib.load('data/regret_bundle.pkl')
    model, features = bundle['model'], bundle['features']
except:
    st.error("AI Brain not found. Please run 'python train_model.py' first.")
    st.stop()

# 3. STATE CONTROLLER
if 'ui_state' not in st.session_state: st.session_state.ui_state = 'home'
if 'vault' not in st.session_state: st.session_state.vault = []

# --- VIEW 1: PREMIUM HOME DASHBOARD ---
if st.session_state.ui_state == 'home':
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Home")
    
    # Hero Card
    st.markdown("""
        <div class="rev-card">
            <p style="color: #8E8E93; font-size: 14px; margin-bottom: 5px;">Main Account • EUR</p>
            <h1 style="font-size: 42px; margin: 0;">€ 2,840.50</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Action Grid
    col1, col2, col3, col4 = st.columns(4)
    col1.button("➕") # Add money
    if col2.button("💸"): # Transfer (Trigger Feature)
        st.session_state.ui_state = 'payment_input'
        st.rerun()
    col3.button("📊") # Insights
    if col4.button("🛡️"): # Vault
        st.session_state.ui_state = 'vault'
        st.rerun()

    # Transaction History
    st.markdown("<h3 style='margin-top:30px;'>Transactions</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class="tx-item"><span>🍕 Uber Eats</span><span style="font-weight:700;">-€22.40</span></div>
        <div class="tx-item"><span>🛍️ Amazon.com</span><span style="font-weight:700;">-€45.00</span></div>
        <div class="tx-item"><span>🏢 Salary Payment</span><span style="color:#00CC96; font-weight:700;">+€3,100.00</span></div>
    """, unsafe_allow_html=True)

# --- VIEW 2: SMART PAYMENT INTERVENTION ---
elif st.session_state.ui_state == 'payment_input':
    st.title("Send Money")
    st.markdown("<p style='color:#8E8E93;'>Merchant: <b>ASOS London / StockX</b></p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="rev-card">', unsafe_allow_html=True)
        # We put the "features" here as if the user is filling out checkout details
        price = st.number_input("Amount to pay (€)", value=199.0, step=10.0)
        st.markdown("<p style='font-size:12px; color:#8E8E93;'>AI Behavioral Check active</p>", unsafe_allow_html=True)
        
        # Simulated "Sensors" (The sliders are integrated context)
        st.divider()
        st.caption("Financial Context Check")
        mood = st.select_slider("Current Mood", options=range(1,11), value=5)
        sleep = st.slider("Sleep (Hours)", 3.0, 11.0, 7.0)
        risk = st.select_slider("Category Risk", options=[0.05, 0.15, 0.30, 0.55], value=0.15)
        fomo = st.toggle("Flash Sale / FOMO Trigger")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Verify & Send", type="primary"):
        with st.spinner('AI Guard is evaluating...'):
            time.sleep(1.5)
            
            # RUN LOGIC (Engineering happenning inside UI to stay responsive)
            rel_price = price / 2840.50
            imp_index = ((11 - mood) * 0.3 + (11 - sleep) * 0.3 + (int(fomo) * 1.5))
            
            inputs = pd.DataFrame([[price, 2840.50, mood, int(fomo), sleep, risk, rel_price, imp_index]], columns=features)
            score = model.predict(inputs)[0]
            
            st.session_state.last_score = score
            st.session_state.last_price = price
            st.session_state.ui_state = 'ai_evaluator'
            st.rerun()
    
    if st.button("← Cancel"): st.session_state.ui_state = 'home'; st.rerun()

# --- VIEW 3: THE AI EVALUATOR (Decision Screen) ---
elif st.session_state.ui_state == 'ai_evaluator':
    score = st.session_state.last_score
    st.title("🛡️ AI Security Check")
    
    st.markdown(f"""
        <div class="rev-card" style="border: 2px solid {'#FF4B4B' if score > 60 else '#00CC96'}">
            <p style="text-align:center; color:#8E8E93;">Risk of Future Regret</p>
            <h1 style="text-align:center; font-size: 64px; color:{'#FF4B4B' if score > 60 else '#00CC96'}">{score:.1f}%</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if score > 60:
        st.markdown(f"""
            <div class="ai-alert">
                <b>Risk Detected:</b> This purchase represents a high emotional risk. 
                Our model indicates that your low mood and limited sleep may be driving an impulsive decision.
            </div>
        """, unsafe_allow_html=True)
        
        st.write("**AI Risk Drivers (XAI):**")
        st.bar_chart(pd.Series(model.feature_importances_, index=features))
        
        if st.button("❄️ Block and Move to Cooling Vault"):
            st.session_state.vault.append({"Item": "ASOS / StockX", "Amount": f"{st.session_state.last_price}€", "Risk": f"{score:.1f}%"})
            st.toast("Payment blocked. Funds safe.")
            st.session_state.ui_state = 'home'
            st.rerun()
    else:
        st.success("Analysis complete: This purchase is well-balanced with your habits.")
        if st.button("Complete Payment Now"):
            st.session_state.ui_state = 'home'
            st.rerun()
            
    if st.button("← Return Home"): st.session_state.ui_state = 'home'; st.rerun()

# --- VIEW 4: COOLING VAULT ---
elif st.session_state.ui_state == 'vault':
    st.title("Cooling Vault")
    if st.session_state.vault:
        st.table(st.session_state.vault)
        if st.button("Empty Vault"): st.session_state.vault = []; st.rerun()
    else:
        st.info("No impulsive items currently blocked.")
    if st.button("← Back"): st.session_state.ui_state = 'home'; st.rerun()
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Initialize Runtime: Load the pre-trained bundle
bundle = joblib.load('data/regret_bundle.pkl')
model = bundle['model']
feature_names = bundle['features']

# 2. State Management for the 'Cooling List' (Exploration requirement)
if 'cooling_list' not in st.session_state:
    st.session_state.cooling_list = []

# 3. Page Layout (Appearance / UX)
st.set_page_config(page_title="Regret Guard Pro", page_icon="🛡️", layout="centered")
st.title("🛡️ Regret Guard Pro")
st.markdown("Prevent impulsive shopping by visualizing your 'Regret Risk' in real-time.")

# Use Tabs for organized interaction
tab1, tab2 = st.tabs(["🔍 Purchase Check", "⏳ My Cooling List"])

with tab1:
    st.subheader("Current Transaction Context")
    
    # Sidebar for simulated Banking context
    with st.sidebar:
        st.header("💳 Bank Status")
        balance = st.number_input("Current Balance (€)", value=1200.0, step=50.0)
        st.caption(f"Model Accuracy (MAE): {bundle['accuracy_mae']:.2f}")

    # Input widgets for the current purchase
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Item Price (€)", min_value=0.0, value=89.0)
        merchant = st.select_slider("Merchant History Risk", 
                                     options=[0.05, 0.15, 0.25, 0.45], 
                                     value=0.15, 
                                     format_func=lambda x: f"{int(x*100)}% Returns")
    with col2:
        mood = st.slider("Mood (1: Low, 10: Balanced)", 1, 10, 6)
        sleep = st.slider("Sleep (h)", 3.0, 12.0, 7.5)
        fomo = st.toggle("Is this a flash sale?")

    # Execution & Evaluator Mode
    if st.button("Evaluate Purchase Risk", use_container_width=True):
        # Prepare data for model inference
        input_data = pd.DataFrame([[price, balance, mood, int(fomo), sleep, merchant]], 
                                  columns=feature_names)
        
        risk_score = model.predict(input_data)[0]
        
        st.divider()
        st.metric("Predicted Regret Risk", f"{risk_score:.1f}%")
        
        if risk_score > 65:
            st.error("🚨 High Risk: This purchase matches your historical regret pattern.")
            if st.button("Add to Cooling List"):
                st.session_state.cooling_list.append({
                    "Item": "Pending Check", 
                    "Price": f"{price}€", 
                    "Risk": f"{risk_score:.1f}%"
                })
                st.toast("Item saved for 24h review!")
        elif risk_score > 35:
            st.warning("⚠️ Caution: You might be shopping under stress or fatigue.")
        else:
            st.success("✅ Looks safe: This purchase fits your stable financial habits.")

with tab2:
    st.subheader("Items under 24h Review")
    if st.session_state.cooling_list:
        st.table(st.session_state.cooling_list)
        if st.button("Clear List"):
            st.session_state.cooling_list = []
            st.rerun()
    else:
        st.info("Your list is currently empty. Shop wisely!")
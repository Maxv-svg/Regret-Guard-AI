import streamlit as st
import joblib
import pandas as pd

# Load Bundle
bundle = joblib.load('data/regret_bundle.pkl')
model = bundle['model']
feature_names = bundle['features']

if 'vault' not in st.session_state:
    st.session_state.vault = []

st.set_page_config(page_title="NeoBank | AI Guard", layout="wide")

# Revolut-style Navigation
with st.sidebar:
    st.title("NeoBank 🏦")
    st.write("Welcome, **Maximilian**")
    st.divider()
    page = st.radio("Navigation", ["Home", "Cards", "Regret Guard AI", "Payments"])
    st.divider()
    balance = st.number_input("Total Balance (€)", value=1250.0, step=100.0)

if page == "Regret Guard AI":
    st.title("🛡️ AI Regret Guard")
    tab1, tab2 = st.tabs(["🔍 New Assessment", "📂 Cooling Vault"])

    with tab1:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.subheader("Details")
            price = st.number_input("Price (€)", min_value=0.0, value=150.0)
            merchant = st.select_slider("Category Risk", options=[0.05, 0.15, 0.30, 0.50], value=0.15)
            fomo = st.toggle("Flash Sale?")
            mood = st.slider("Mood (1-10)", 1, 10, 6)
            sleep = st.slider("Sleep (h)", 3.0, 12.0, 7.5)

        with c2:
            st.subheader("AI Analysis")
            if st.button("Analyze Transaction", type="primary", use_container_width=True):
                # Calculate derived features for prediction
                rel_price = price / balance
                impulsivity = ((11 - mood) * 0.4 + (11 - sleep) * 0.4 + (int(fomo) * 2))
                
                input_df = pd.DataFrame([[price, balance, mood, int(fomo), sleep, merchant, rel_price, impulsivity]], 
                                        columns=feature_names)
                score = model.predict(input_df)[0]
                
                st.metric("Regret Risk", f"{score:.1f}%")
                st.write("**Risk Drivers (XAI):**")
                st.bar_chart(pd.Series(model.feature_importances_, index=feature_names))

                if score > 60:
                    st.error("🚨 High Regret Risk! Lack of sleep and price impact detected.")
                    if st.button("Move to Cooling Vault"):
                        st.session_state.vault.append({"Item": "Purchase", "Price": f"{price}€", "Risk": f"{score:.1f}%"})
                        st.toast("Saved to Vault!")
                else:
                    st.success("✅ Safe Purchase. Enjoy!")

    with tab2:
        st.subheader("Cooling Vault (24h Review)")
        if st.session_state.vault: st.table(st.session_state.vault)
        else: st.info("Vault is empty.")
else:
    st.title(f"{page}")
    st.write("Standard banking features go here.")
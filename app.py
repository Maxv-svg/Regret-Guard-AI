import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the advanced bundle
bundle = joblib.load('data/regret_bundle.pkl')
model = bundle['model']
feature_names = bundle['features']

# Functional State Management for the Cooling List
if 'cooling_list' not in st.session_state:
    st.session_state.cooling_list = []

st.set_page_config(page_title="Regret Guard Pro", layout="wide")
st.title("🛡️ Regret Guard Pro")
st.markdown("Datengestützte Kauf-Evaluation zur Vermeidung von impulsivem Verhalten.")

# Layout with Tabs
tab1, tab2 = st.tabs(["🔍 Analyse-Tool", "⏳ Meine Cooling-Liste"])

with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("Eingabedaten")
        price = st.number_input("Kaufpreis (€)", value=149.0)
        balance = st.sidebar.number_input("Kontostand (€)", value=1000.0) # In Sidebar for context
        mood = st.select_slider("Stimmung (1-10)", options=range(1,11), value=5)
        sleep = st.slider("Schlaf (Stunden)", 3.0, 12.0, 7.0)
        merchant = st.select_slider("Händler-Risiko", options=[0.05, 0.15, 0.25, 0.45], 
                                    format_func=lambda x: f"{int(x*100)}% Retouren", value=0.15)
        fomo = st.toggle("Limitierter Sale?")

    with col_out:
        st.subheader("KI-Ergebnis")
        if st.button("Kauf prüfen", use_container_width=True):
            input_df = pd.DataFrame([[price, balance, mood, int(fomo), sleep, merchant]], columns=feature_names)
            score = model.predict(input_df)[0]
            
            st.metric("Regret Risk", f"{score:.1f}%", delta=f"{score-50:.1f}%", delta_color="inverse")
            
            # XAI: Explain why the score is high
            st.write("**Warum dieser Score?**")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            st.bar_chart(feat_imp)
            
            if score > 65:
                st.error("🚨 Achtung: Hohes Risiko für einen Fehlkauf!")
                if st.button("Auf Cooling-Liste setzen"):
                    st.session_state.cooling_list.append({"Item": "Kaufanfrage", "Preis": f"{price}€", "Risiko": f"{score:.1f}%"})
                    st.toast("In Warteliste gespeichert!")
            else:
                st.success("✅ Dieser Kauf scheint vernünftig zu sein.")

with tab2:
    st.subheader("Warteliste (24h Review)")
    if st.session_state.cooling_list:
        st.table(st.session_state.cooling_list)
        if st.button("Liste leeren"):
            st.session_state.cooling_list = []
            st.rerun()
    else:
        st.info("Deine Liste ist aktuell leer. Bleib diszipliniert!")
import streamlit as st
import joblib
import pandas as pd
import time

# 1. NATIVE MOBILE APP STYLING (The "Revolut" Skin)
st.set_page_config(page_title="Regret Guard • Revolut", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Global app background and font */
    .stApp {
        background: radial-gradient(circle at top, #1e1e2e 0, #050509 50%, #000000 100%);
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }

    /* Hide Streamlit default chrome */
    header, footer {visibility: hidden;}

    /* Centered phone shell */
    .block-container {
        max-width: 480px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .phone-shell {
        margin: auto;
        max-width: 420px;
        border-radius: 32px;
        padding: 16px 14px 18px 14px;
        background: radial-gradient(circle at top left, #18181f 0, #050509 40%, #000000 100%);
        box-shadow:
            0 24px 60px rgba(0, 0, 0, 0.85),
            0 0 0 1px rgba(255, 255, 255, 0.04);
        position: relative;
    }

    /* Fake phone notch / status bar */
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 10px 6px 10px;
        font-size: 11px;
        color: #8E8E93;
    }
    .status-pill {
        width: 110px;
        height: 6px;
        border-radius: 999px;
        background: #1f1f23;
        margin: 0 auto 4px auto;
    }

    /* Top app bar */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 4px 12px 4px;
    }
    .app-title {
        font-size: 17px;
        font-weight: 600;
    }
    .tag-pill {
        font-size: 11px;
        padding: 2px 10px;
        border-radius: 999px;
        background: rgba(25, 119, 243, 0.16);
        border: 1px solid rgba(25, 119, 243, 0.45);
        color: #d0e0ff;
    }

    /* Revolut-style balance card */
    .rev-card {
        background: radial-gradient(circle at top left, #26263a 0%, #111118 35%, #050509 100%);
        padding: 22px 20px;
        border-radius: 24px;
        border: 1px solid #272739;
        margin-bottom: 18px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.8);
    }

    /* Action buttons (primary) */
    .stButton>button {
        border-radius: 16px;
        height: 3.2em;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        color: #ffffff;
        border: none;
        font-weight: 600;
        letter-spacing: 0.2px;
        font-size: 14px;
        transition: all 0.22s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.55);
    }

    /* Ghost / secondary buttons */
    .ghost-btn>button {
        border-radius: 16px !important;
        height: 3.2em !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(120, 120, 128, 0.35) !important;
        color: #f5f5f7 !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }

    /* Transaction list items */
    .tx-list {
        margin-top: 4px;
        background: rgba(20, 20, 24, 0.9);
        border-radius: 18px;
        padding: 10px 16px 4px 16px;
        border: 1px solid rgba(39, 39, 57, 0.85);
    }
    .tx-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 0.5px solid #2c2c2e;
        font-size: 13px;
    }
    .tx-item:last-child {
        border-bottom: none;
        padding-bottom: 4px;
    }

    /* AI Risk Alert block */
    .ai-alert {
        background: linear-gradient(135deg, #2b1114 0%, #1c1c1e 40%, #111111 100%);
        border-left: 4px solid #FF4B4B;
        padding: 16px 16px 14px 16px;
        border-radius: 16px;
        margin: 16px 0 8px 0;
        font-size: 13px;
    }

    /* Bottom nav mimicking Revolut */
    .bottom-nav {
        position: relative;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid rgba(39, 39, 57, 0.85);
        font-size: 11px;
        color: #8E8E93;
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

# 3. STATE CONTROLLER (simple state machine)
if "ui_state" not in st.session_state:
    st.session_state.ui_state = "home"
if "vault" not in st.session_state:
    st.session_state.vault = []
if "current_item" not in st.session_state:
    # Simulated e‑commerce checkout item
    st.session_state.current_item = {
        "name": "ASOS • Oversized hoodie",
        "emoji": "🧥",
        "category_risk": 0.30,
        "suggested_price": 89.0,
    }

# Phone frame wrapper (all views live inside)
st.markdown('<div class="phone-shell">', unsafe_allow_html=True)

# Fake phone status bar + header always visible
st.markdown(
    """
    <div class="status-pill"></div>
    <div class="status-bar">
        <span>09:41</span>
        <span>Regret Guard</span>
        <span>🔋 82%</span>
    </div>
    <div class="app-header">
        <div class="app-title">Revolut • Guard AI</div>
        <div class="tag-pill">Behavioral Safety</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- VIEW 1: PREMIUM HOME / SIMULATION DASHBOARD ---
if st.session_state.ui_state == "home":
    # Hero Card
    st.markdown(
        """
        <div class="rev-card">
            <p style="color: #8E8E93; font-size: 12px; margin-bottom: 2px;">Main Account • EUR</p>
            <h1 style="font-size: 40px; margin: 0 0 4px 0;">€ 2,840.50</h1>
            <p style="color:#4ade80; font-size:12px; margin:0;">Regret Guard is <b>on</b> for online checkouts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simulation card: new online checkout waiting for verification
    item = st.session_state.current_item
    st.markdown(
        f"""
        <div class="tx-list" style="margin-top:6px; margin-bottom:10px;">
            <div class="tx-item" style="border-bottom:none; padding-bottom:6px;">
                <div>
                    <div style="font-size:13px; color:#8E8E93; margin-bottom:2px;">Pending online checkout</div>
                    <div style="font-size:15px; font-weight:600;">
                        {item['emoji']} {item['name']}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-weight:600;">‑€{item['suggested_price']:.2f}</div>
                    <div style="font-size:11px; color:#f97373;">Tap to verify</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Review this purchase with AI"):
        st.session_state.ui_state = "payment_input"
        st.rerun()

    # Transaction History inside phone card (purely contextual)
    st.markdown(
        "<p style='margin-top:18px; font-size:13px; color:#8E8E93;'>Recent activity</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="tx-list">
            <div class="tx-item">
                <span>🍕 Uber Eats</span>
                <span style="font-weight:600;">-€22.40</span>
            </div>
            <div class="tx-item">
                <span>🛍️ Amazon.com</span>
                <span style="font-weight:600;">-€45.00</span>
            </div>
            <div class="tx-item">
                <span>🎮 Playstation Store</span>
                <span style="font-weight:600;">-€59.99</span>
            </div>
            <div class="tx-item">
                <span style="color:#00CC96;">🏢 Salary Payment</span>
                <span style="color:#00CC96; font-weight:600;">+€3,100.00</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- VIEW 2: SMART PAYMENT INTERVENTION (CHECKOUT SCREEN) ---
elif st.session_state.ui_state == "payment_input":
    item = st.session_state.current_item

    st.markdown(
        "<p style='font-size:14px; color:#8E8E93; margin-bottom:4px;'>Step 1 · Verify this purchase</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h3 style='margin-top:0; margin-bottom:8px;'>{item['emoji']} {item['name']}</h3>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="rev-card">', unsafe_allow_html=True)
        price = st.number_input(
            "Amount at checkout (€)",
            value=float(item["suggested_price"]),
            step=5.0,
        )
        st.markdown(
            "<p style='font-size:12px; color:#8E8E93; margin-top:-4px;'>"
            "Before Revolut sends this money, Regret Guard runs a quick behavioral check."
            "</p>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("How are you right now? (this shapes the AI's decision)")
        mood = st.select_slider(
            "Current mood (1 = low, 10 = great)", options=range(1, 11), value=5
        )
        sleep = st.slider("Sleep last night (hours)", 3.0, 11.0, 7.0)
        risk = st.select_slider(
            "Merchant / category risk", options=[0.05, 0.15, 0.30, 0.55], value=item["category_risk"]
        )
        fomo = st.toggle("This feels like a limited‑time deal / flash sale")
        st.markdown("</div>", unsafe_allow_html=True)

    # Primary call to action: run the AI check
    if st.button("Run Regret Guard check", type="primary"):
        with st.spinner("Analyzing your situation with the AI model..."):
            time.sleep(1.5)

            rel_price = price / 2840.50
            imp_index = (11 - mood) * 0.3 + (11 - sleep) * 0.3 + (int(fomo) * 1.5)

            inputs = pd.DataFrame(
                [[price, 2840.50, mood, int(fomo), sleep, risk, rel_price, imp_index]],
                columns=features,
            )
            score = model.predict(inputs)[0]

            st.session_state.last_score = score
            st.session_state.last_price = price
            st.session_state.ui_state = "ai_evaluator"
            st.rerun()

    cancel_col, _ = st.columns([1, 2])
    if cancel_col.button("← Back to account"):
        st.session_state.ui_state = "home"
        st.rerun()

# --- VIEW 3: THE AI EVALUATOR (Decision Screen) ---
elif st.session_state.ui_state == "ai_evaluator":
    score = st.session_state.last_score
    item = st.session_state.current_item

    st.markdown(
        "<p style='font-size:14px; color:#8E8E93; margin-bottom:4px;'>Step 2 · AI evaluates your decision</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h3 style='margin-top:0; margin-bottom:8px;'>Regret risk for {item['emoji']} {item['name']}</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rev-card" style="border: 2px solid {'#FF4B4B' if score > 60 else '#22c55e'}">
            <p style="text-align:center; color:#8E8E93; font-size:13px; margin-bottom:6px;">
                For this simulated payment of <b>€{st.session_state.last_price:.2f}</b>
            </p>
            <h1 style="text-align:center; font-size: 56px; margin: 0; color:{'#FF4B4B' if score > 60 else '#22c55e'};">
                {score:.1f}%
            </h1>
            <p style="text-align:center; color:#8E8E93; font-size:12px; margin-top:4px;">
                chance that you'll regret this purchase later, based on your past data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if score > 60:
        st.markdown(
            """
            <div class="ai-alert">
                <b>High emotional risk detected.</b><br/>
                Your current mood, recent sleep and FOMO pattern look similar to past purchases
                that you later regretted. Regret Guard recommends <b>waiting</b> before buying.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="ai-alert" style="border-left-color:#22c55e; background:linear-gradient(135deg,#102318 0%,#111111 60%);">
                <b>Low regret risk.</b><br/>
                This purchase looks consistent with your usual behavior and financial buffer.
                You can safely complete it, or still park it in the Cooling Vault if you prefer a pause.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("**Why the AI thinks this (feature importance):**")
    st.bar_chart(pd.Series(model.feature_importances_, index=features))

    # Explicit decision: Continue vs move to vault, regardless of risk level
    st.markdown("**Step 3 · What do you want to do?**")
    primary_col, secondary_col = st.columns(2)

    with primary_col:
        if st.button("Buy it anyway"):
            st.toast("In this simulation, the hoodie is bought. In the real app, funds would leave your account.")
            st.session_state.ui_state = "home"
            st.rerun()

    with secondary_col:
        if st.container().button("Save in Cooling Vault", key="vault_from_ai"):
            st.session_state.vault.append(
                {
                    "Item": item["name"],
                    "Amount": f"{st.session_state.last_price:.2f}€",
                    "Risk": f"{score:.1f}%",
                }
            )
            st.toast("Stored in Cooling Vault. Real app would remind you in 24h.")
            st.session_state.ui_state = "home"
            st.rerun()

    back_col, _ = st.columns([1, 2])
    if back_col.button("← Change how you feel / context"):
        st.session_state.ui_state = "payment_input"
        st.rerun()

# --- VIEW 4: COOLING VAULT ---
elif st.session_state.ui_state == 'vault':
    st.markdown(
        "<h3 style='margin-top:4px; margin-bottom:4px;'>Cooling Vault</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:13px; color:#8E8E93;'>Items you stopped at the last minute to review with a cool head.</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.vault:
        st.table(st.session_state.vault)
        clear_col, _ = st.columns([1, 2])
        if clear_col.button("Empty Vault"):
            st.session_state.vault = []
            st.rerun()
    else:
        st.info("No impulsive items currently blocked. Regret Guard is quiet. 💤")

    back_col, _ = st.columns([1, 2])
    if back_col.button("← Back to home"):
        st.session_state.ui_state = "home"
        st.rerun()

# Bottom nav always visible inside phone shell to mimic mobile app
nav_home, nav_vault = st.columns(2)
with nav_home:
    if st.button("🏠 Home", key="nav_home"):
        st.session_state.ui_state = "home"
        st.rerun()
with nav_vault:
    if st.button("🛡️ Vault", key="nav_vault"):
        st.session_state.ui_state = "vault"
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
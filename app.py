import streamlit as st
import joblib
import pandas as pd
import time

from utils.data_processor import get_product_insights
from utils.llm_chains import run_regret_chain

# 1. NATIVE MOBILE APP STYLING (The "Revolut" Skin)
st.set_page_config(page_title="Regret Guard • Revolut", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Global app background and font */
    .stApp {
        background: radial-gradient(circle at top, #1e1e2e 0, #050509 50%, #000000 100%);
        color: #f5f5f7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        font-size: 14px;
    }

    /* Form field labels (sliders, inputs, etc.) */
    .stApp label {
        color: #f9fafb !important;
        font-size: 13px;
        font-weight: 500;
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

    /* Step labels for the 3-step flow */
    .step-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #a5b4fc;
        margin-bottom: 3px;
        text-align: left;
    }
    .step-title {
        font-size: 20px;
        font-weight: 600;
        color: #f9fafb;
        margin-bottom: 4px;
        text-align: left;
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
        height: 3.0em;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 60%, #111827 100%);
        color: #ffffff;
        border: none;
        font-weight: 600;
        letter-spacing: 0.1px;
        font-size: 13px;
        transition: all 0.16s ease;
        box-shadow: none;
    }
    .stButton>button:hover {
        filter: brightness(1.05);
        transform: translateY(-0.5px);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.6);
    }

    /* Primary CTA in checkout screen */
    .primary-cta .stButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 45%, #166534 100%);
    }
    .primary-cta .stButton>button:hover {
        filter: brightness(1.04);
        box-shadow: 0 6px 18px rgba(22, 163, 74, 0.55);
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
        background: linear-gradient(135deg, #2b1114 0%, #19191d 40%, #0d0d10 100%);
        border-left: 4px solid #FF4B4B;
        padding: 14px 14px 12px 14px;
        border-radius: 14px;
        margin: 14px 0 6px 0;
        font-size: 13px;
        line-height: 1.4;
    }

    /* Context panel for "How you feel" section (visual container turned off to avoid pill bar) */
    .context-panel {
        margin-top: 4px;
        padding: 0;
        border-radius: 0;
        background: transparent;
        border: none;
    }

    /* Bottom nav was removed for a cleaner layout */

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
    # Default: typical Amazon product (matches amazon_reviews.csv context)
    st.session_state.current_item = {
        "name": "Amazon • Wireless Earbuds",
        "emoji": "",
        "category_risk": 0.25,
        "suggested_price": 59.99,
    }
if "last_outcome" not in st.session_state:
    st.session_state.last_outcome = None
if "rag_regret_probability" not in st.session_state:
    st.session_state.rag_regret_probability = None
    st.session_state.rag_core_failure_points = ""
    st.session_state.rag_counter_argument = ""
    st.session_state.rag_raw_reviews = []
    st.session_state.rag_matched_query = ""
    st.session_state.rag_rating_band = "1-2"  # "1-2", "1-3", or "2-3"
    st.session_state.rag_empty_reason = None

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
            <p style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#a5b4fc; margin-bottom:4px;">
                Main account • EUR
            </p>
            <h1 style="font-size: 32px; margin: 0 0 2px 0;">€ 2,840.50</h1>
            <p style="color:#4ade80; font-size:11px; margin:0;">Regret Guard is <b>on</b> for online checkouts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simulation card: new online checkout waiting for verification
    item = st.session_state.current_item
    st.markdown(
        f"""
        <div class="tx-list" style="margin-top:8px; margin-bottom:10px;">
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#a5b4fc; margin-bottom:4px;">
                Pending online checkout
            </div>
            <div class="tx-item" style="border-bottom:none; padding-bottom:4px; padding-top:4px;">
                <div>
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
        "<p style='margin-top:18px; font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#a5b4fc; margin-bottom:4px;'>Recent activity</p>",
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
        "<div class='step-label'>Step 1 · Checkout details</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='step-title'>{item['name']}</div>",
        unsafe_allow_html=True,
    )

    with st.container():
        price = float(item["suggested_price"])
        st.markdown(
            f"<p style='font-size:13px; letter-spacing:0.08em; text-transform:uppercase; color:#a5b4fc; margin-bottom:0;'>Amount at checkout</p>"
            f"<p style='font-size:26px; font-weight:700; margin-top:2px; color:#f9fafb;'>€ {price:.2f}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:13px; color:#e5e7eb; margin-top:4px;'>"
            "Before Revolut sends this money, Regret Guard runs a quick behavioral check."
            "</p>",
            unsafe_allow_html=True,
        )

        # Lightweight RAG input section
        st.markdown(
            "<div class='step-label' style='margin-top:10px; margin-bottom:2px;'>Product context</div>",
            unsafe_allow_html=True,
        )
        product_query = st.text_input(
            "Which product are you about to buy? (for review lookup)",
            value="earbuds",
            placeholder="e.g. earbuds, headphones, wireless earbuds",
            key="product_query",
            help="Words here are matched against 1–3★ Amazon reviews (1–2★ preferred; 2–3★ used when scarce). Default 'earbuds' is set so you get review results; change if your data has other product keywords.",
        )
        st.caption(
            "Default is **earbuds** so you’ll get real Amazon review evidence in the result. Change to *headphones*, *wireless earbuds*, or other product words that appear in your review data."
        )
        user_reason = st.text_area(
            "Why do you want to buy this?",
            placeholder="Be honest – what is the main reason you want this purchase right now?",
            key="user_reason",
        )

        st.divider()
        st.markdown(
            "<div class='step-label' style='margin-bottom:2px;'>Step 2 · Your current state</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='step-title' style='font-size:18px; margin-bottom:4px;'>How you feel now shapes the AI decision</div>",
            unsafe_allow_html=True,
        )

        mood = st.select_slider(
            "Mood right now (1 = low, 10 = great)", options=range(1, 11), value=5
        )
        sleep = st.slider("Sleep last night (hours)", 3.0, 11.0, 7.0)
        risk = st.select_slider(
            "Merchant / category risk",
            options=[0.05, 0.15, 0.25, 0.30, 0.55],
            value=item["category_risk"] if item["category_risk"] in [0.05, 0.15, 0.25, 0.30, 0.55] else 0.25,
        )
        st.markdown(
            "<p style='margin-top:10px; margin-bottom:2px; color:#f9fafb; font-size:14px; font-weight:600;'>"
            "This feels like a limited‑time deal / flash sale"
            "</p>",
            unsafe_allow_html=True,
        )
        fomo = st.toggle("", label_visibility="collapsed")

    # Primary call to action: run the AI check
    st.markdown('<div class="primary-cta">', unsafe_allow_html=True)
    run_clicked = st.button("Run Regret Guard check", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    back_clicked = st.button("← Back", key="back_from_checkout")

    if run_clicked:
        with st.spinner("Analyzing your situation with the AI model..."):
            time.sleep(1.5)

            rel_price = price / 2840.50
            imp_index = (11 - mood) * 0.3 + (11 - sleep) * 0.3 + (int(fomo) * 1.5)

            inputs = pd.DataFrame(
                [[price, 2840.50, mood, int(fomo), sleep, risk, rel_price, imp_index]],
                columns=features,
            )
            score = model.predict(inputs)[0]

            # --- RAG‑light pipeline: fetch real complaints and run the LLM chain ---
            st.session_state.rag_regret_probability = None
            st.session_state.rag_core_failure_points = ""
            st.session_state.rag_counter_argument = ""
            st.session_state.rag_raw_reviews = []
            st.session_state.rag_matched_query = ""
            st.session_state.rag_rating_band = "1-2"
            st.session_state.rag_empty_reason = None

            try:
                insights = get_product_insights(product_query)
                complaints_text = insights.get("complaints_text", "")
                if not complaints_text.strip():
                    st.session_state.rag_empty_reason = insights.get("empty_reason")
                if complaints_text.strip():
                    chain_result = run_regret_chain(
                        product_query=insights.get("matched_query", product_query),
                        user_reason=user_reason or "No explicit reason provided.",
                        complaints_text=complaints_text,
                    )
                    st.session_state.rag_regret_probability = chain_result.get(
                        "regret_probability"
                    )
                    st.session_state.rag_core_failure_points = chain_result.get(
                        "core_failure_points", ""
                    )
                    st.session_state.rag_counter_argument = chain_result.get(
                        "counter_argument", ""
                    )
                    st.session_state.rag_raw_reviews = insights.get(
                        "raw_reviews", []
                    )
                    st.session_state.rag_matched_query = insights.get(
                        "matched_query", product_query
                    )
                    st.session_state.rag_rating_band = insights.get("rating_band") or "1-2"
            except Exception as e:
                # Fail gracefully – keep the core ML guard working even if RAG fails.
                st.warning(
                    f"Regret Guard evidence lookup is temporarily unavailable ({e}). "
                    "Core risk score is still shown below."
                )

            st.session_state.last_score = score
            st.session_state.last_price = price
            st.session_state.ui_state = "ai_evaluator"
            st.rerun()

    if back_clicked:
        st.session_state.ui_state = "home"
        st.rerun()

# --- VIEW 3: THE AI EVALUATOR (Decision Screen) ---
elif st.session_state.ui_state == "ai_evaluator":
    score = st.session_state.last_score
    rag_score = st.session_state.rag_regret_probability
    # When we have both ML and RAG, use a combined score for the "final rating"
    if rag_score is not None:
        combined_score = (score + rag_score) / 2.0
        score_for_display = combined_score
        score_source_note = "combined (your behavior + real Amazon reviews)"
    else:
        score_for_display = score
        score_source_note = "based on your past data"

    item = st.session_state.current_item
    importances = pd.Series(model.feature_importances_, index=features)
    top_features = ", ".join(importances.sort_values(ascending=False).head(3).index.tolist())

    st.markdown(
        "<div class='step-label'>Step 2 · AI review</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='step-title'>Regret risk for {item['emoji']} {item['name']}</div>",
        unsafe_allow_html=True,
    )

    # Risk bands: 0–20 low (green), 20–40 medium (yellow), >40 high (red)
    if score_for_display <= 20:
        band_color = "#22c55e"
        headline_text = "Low regret risk."
        explanation = (
            "This purchase looks consistent with your usual behavior and financial buffer. "
            "You can safely complete it or still stop the transaction if you prefer a pause."
        )
    elif score_for_display <= 40:
        band_color = "#eab308"
        headline_text = "Medium regret risk."
        explanation = (
            "This purchase is borderline for your usual behavior. If you are unsure, consider stopping "
            "the transaction and revisiting it later."
        )
    else:
        band_color = "#ef4444"
        headline_text = "High regret risk."
        explanation = (
            "Your current mood, recent sleep and FOMO pattern look similar to past purchases that you later regretted. "
            "Regret Guard recommends waiting before buying."
        )

    # Rating band label for RAG copy (1-2, 1-3, or 2-3)
    _band = getattr(st.session_state, "rag_rating_band", "1-2")
    _band_short = "2–3★" if _band == "2-3" else ("1–3★" if _band == "1-3" else "1–2★")
    _band_long = (
        "2–3★" if _band == "2-3" else
        "1–3★ (2–3★ used where 1–2★ was scarce)" if _band == "1-3" else
        "1–2★"
    )

    # Build reasoning text: ML factors + optional Cohere/review evidence
    reasoning_ml = f"The behavioral model pays most attention to: <b>{top_features}</b>."
    reasoning_reviews = ""
    if st.session_state.rag_regret_probability is not None:
        # Always include review evidence in the main description when we have a RAG score
        parts = []
        if st.session_state.rag_core_failure_points and st.session_state.rag_core_failure_points.strip():
            parts.append(
                f" From real {_band_short} Amazon reviews, the main concerns are: "
                + st.session_state.rag_core_failure_points.strip()[:300]
                + ("…" if len(st.session_state.rag_core_failure_points.strip()) > 300 else "")
                + "."
            )
        parts.append(
            f" The evidence-based regret estimate from these reviews is <b>{st.session_state.rag_regret_probability:.1f}%</b>. "
            "The score above combines this with your behavioral risk."
        )
        reasoning_reviews = " ".join(parts)

    st.markdown(
        f"""
        <div class="rev-card" style="border: 2px solid {band_color};">
            <p style="text-align:center; color:#f9fafb; font-size:13px; margin-bottom:4px;">
                For this simulated payment of <b>€{st.session_state.last_price:.2f}</b>
            </p>
            <h1 style="text-align:center; font-size: 56px; margin: 0; color:{band_color};">
                {score_for_display:.1f}%
            </h1>
            <p style="text-align:center; color:#f9fafb; font-size:13px; margin-top:2px;">
                chance that you'll regret this purchase later ({score_source_note})
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ai-alert" style="border-left-color:{band_color};">
            <p style="margin:0 0 4px 0;"><b>{headline_text}</b></p>
            <p style="margin:0 0 4px 0; font-size:13px;">
                {explanation}
            </p>
            <p style="margin:0; font-size:13px;">
                {reasoning_ml}{reasoning_reviews}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- RAG‑light visualisation (or why it's missing) ---
    if getattr(st.session_state, "rag_empty_reason", None):
        st.info(
            "**Review evidence not loaded.** "
            + str(st.session_state.rag_empty_reason)
        )
    if st.session_state.rag_regret_probability is not None:
        st.markdown(
            "<div class='step-label' style='margin-top:12px;'>Evidence from similar buyers</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"*Regret Guard uses real, anonymised **{_band_long}** reviews from the Amazon dataset: "
            "buyers who were unhappy or mixed. The AI summarises common "
            "complaints and weighs them against your reason to buy.*"
        )
        st.write(
            "Based on these reviews, Regret Guard estimates a "
            f"**{st.session_state.rag_regret_probability:.1f}%** "
            "chance that you would feel buyer's remorse."
        )
        progress = min(
            max(st.session_state.rag_regret_probability / 100.0, 0.0), 1.0
        )
        st.progress(progress, text="Evidence‑based regret probability")

        if st.session_state.rag_core_failure_points:
            st.markdown("**Core failure points from real reviews:**")
            st.markdown(st.session_state.rag_core_failure_points)

        if st.session_state.rag_counter_argument:
            st.markdown("**Regret Guard's critical counter‑argument:**")
            st.info(st.session_state.rag_counter_argument)

        if st.session_state.rag_raw_reviews:
            st.markdown(
                f"**Real‑World Evidence:** Below are the **{_band_short}** review texts from the "
                "Amazon dataset that were used for this assessment. They show what went wrong "
                "for other buyers."
            )
            with st.expander(
                f"Show {len(st.session_state.rag_raw_reviews)} raw {_band_short} reviews "
                f"for “{st.session_state.rag_matched_query or 'this purchase'}”"
            ):
                for i, review in enumerate(st.session_state.rag_raw_reviews, start=1):
                    st.markdown(f"**#{i}**  \n{review}")

    # Explicit decision: Continue vs move to vault, regardless of risk level
    st.markdown(
        "<div class='step-label'>Step 3 · Your decision</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="primary-cta">', unsafe_allow_html=True)
    buy_clicked = st.button("✅ Buy it anyway", use_container_width=True)
    save_clicked = st.button("⛔ Stop transaction", key="vault_from_ai_decision", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if buy_clicked:
        st.session_state.last_outcome = {
            "action": "bought",
            "item": item["name"],
            "amount": f"{st.session_state.last_price:.2f}€",
            "risk": f"{score_for_display:.1f}%",
        }
        st.session_state.ui_state = "success"
        st.rerun()

    if save_clicked:
        st.session_state.vault.append(
            {
                "Item": item["name"],
                "Amount": f"{st.session_state.last_price:.2f}€",
                "Risk": f"{score_for_display:.1f}%",
            }
        )
        st.toast("Transaction stopped and kept on file.")
        st.session_state.last_outcome = {
            "action": "stopped",
            "item": item["name"],
            "amount": f"{st.session_state.last_price:.2f}€",
            "risk": f"{score_for_display:.1f}%",
        }
        st.session_state.ui_state = "home"
        st.rerun()

    if st.button("← Return"):
        st.session_state.ui_state = "payment_input"
        st.rerun()

# --- VIEW 4: TRANSACTION SUMMARY / SUCCESS LAYER ---
elif st.session_state.ui_state == "success":
    outcome = st.session_state.last_outcome or {}
    action = outcome.get("action", "bought")

    st.markdown(
        "<div class='step-label'>Summary</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='step-title'>Transaction result</div>",
        unsafe_allow_html=True,
    )

    bg = "#0f172a" if action == "bought" else "#111827"
    border = "#22c55e" if action == "bought" else "#38bdf8"
    headline = "Payment completed" if action == "bought" else "Transaction stopped"
    st.markdown(
        f"""
        <div style="background:{bg}; border-radius:20px; padding:16px 18px; border:1px solid {border}; margin-bottom:12px;">
            <p style="font-size:15px; font-weight:600; margin:0 0 4px 0;">{headline}</p>
            <p style="font-size:22px; font-weight:700; text-align:center; margin:4px 0 10px 0;">
                -{outcome.get("amount","€ 89.00")}
            </p>
            <p style="font-size:13px; margin:0;">
                <b>Item:</b> {outcome.get("item","Amazon • Wireless Earbuds")}<br/>
                <b>Regret risk at decision time:</b> {outcome.get("risk","--")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Back to home"):
        st.session_state.ui_state = "home"
        st.rerun()

# --- VIEW 5: STOPPED TRANSACTIONS (previously Cooling Vault) ---
elif st.session_state.ui_state == 'vault':
    st.markdown(
        "<h3 style='margin-top:4px; margin-bottom:4px;'>Stopped transactions</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:13px; color:#8E8E93;'>Transactions you decided to stop at the last moment.</p>",
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

st.markdown("</div>", unsafe_allow_html=True)
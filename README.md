# Regret Guard

In an era of one-click checkouts and hyper-targeted advertising, the friction between wanting and owning has vanished. This convenience often leads to impulsive "hot-state" decision-making while users are tired, stressed, or pressured by artificial scarcity.

Regret Guard acts as a digital conscience for fintech environments. Instead of just processing a payment, it leverages AI to evaluate whether a transaction is likely a "regret buy" before the money leaves the account, forcing a moment of data-driven reflection to improve user wellbeing.

---

## Key Features

* Predictive Regret Scoring: Utilizes a Random Forest Regressor to identify patterns between behavioral inputs and potential buyer's remorse.
* Impulsivity Index: A custom feature set that mathematically combines user variables such as sleep deprivation, mood, and FOMO triggers.
* Positive Friction: Deliberate AI analysis phases designed to break the "dopamine hit" cycle and force psychological pauses during checkout.
* Dynamic Risk Visualization: Immediate emotional impact through conditional color-coding (Green/Yellow/Red) based on real-time risk assessments.

---

## Prototype-test

Test the prototype on the streamlit cloud via the following link: https://regret-guard-ai.streamlit.app/

---

## Technical Methodology

Regret Guard moves beyond simple transaction limits by analyzing the psychological state of the user at the point of sale.

The core logic is driven by a behavioral pipeline:
1. Data Input: Collection of "hot-state" indicators (Stress, Sleep, Scarcity).
2. Processing: The Random Forest model evaluates the Impulsivity Index.
3. Mitigation: If the Regret Score exceeds the threshold, the UI injects mandatory reflection time.

---

## How the model works (current architecture)

The app combines **two** sources of “regret” signal:

### 1. ML model (offline-trained)

- **Data:** `data/transaction_history.csv` (single source of truth).
- **Training:** `train_model.py` builds engineered features (`relative_price`, `impulsivity_index` from price, balance, mood, sleep, limited offer) and trains a **RandomForestRegressor** to predict `regret_score`.
- **Output:** `data/regret_bundle.pkl` (model + feature list + MAE).
- **At runtime:** When you click “Run Regret Guard check”, the app builds the same features from your inputs (amount, balance, mood, sleep, FOMO, category risk) and the model predicts a **regret %**. That score is shown in the main card and drives the 0–20% (green), 20–40% (yellow), >40% (red) bands.

### 2. RAG-light chain (Cohere + real reviews)

- **Data:** `data/amazon_reviews.csv` (1–2★ reviews).
- **Retrieval:** `utils/data_processor.py` → `get_product_insights(product_query)` loads the CSV, keeps 1–2★ reviews, keyword-matches the product query, ranks by “most descriptive” (length), and returns the top 5 complaints as text + raw review list.
- **LLM chain:** `utils/llm_chains.py` → `run_regret_chain(product_query, user_reason, complaints_text)`:
  - **Step 1 (Analyst):** Cohere summarises the complaints into “Core Failure Points”.
  - **Step 2 (Negotiator):** Cohere takes your “reason to buy” + those failure points and returns a **Regret Probability (0–100)** and a **counter-argument** in Regret Guard’s voice (few-shot prompted).
- **In the UI:** If the RAG pipeline runs successfully, the AI evaluator screen also shows an evidence-based regret % (with a progress bar), core failure points, the counter-argument, and a “Real-World Evidence” expander with the raw 1–2★ review text.

**Summary:** The **main number and bands** come from the Random Forest on transaction data; the **“Evidence from similar buyers”** block (and its score) comes from the Cohere RAG chain on Amazon reviews. Both can be shown together.

---

## Run locally

1. Install: `pip install -r requirements.txt`
2. Train the ML model (once): `python train_model.py` (requires `data/transaction_history.csv`).
3. Set `COHERE_API_KEY` in `.streamlit/secrets.toml` or your environment (optional; needed for the RAG “Evidence from similar buyers” section).
4. Run: `streamlit run app.py`

---

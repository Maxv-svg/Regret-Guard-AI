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

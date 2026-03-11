import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st

try:
    from cohere import Client as CohereClient
except ImportError:
    CohereClient = None  # type: ignore[misc, assignment]


def _get_cohere_client() -> Any:
    """
    Create a Cohere client using a key loaded from Streamlit secrets or the environment.
    """
    if CohereClient is None:
        raise ImportError(
            "The 'cohere' package is not installed. Install it with:  pip install cohere"
        )
    api_key = None
    try:
        api_key = st.secrets.get("COHERE_API_KEY", None)
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY not found. Please add it to st.secrets or your environment."
        )

    return CohereClient(api_key=api_key)


@dataclass
class RegretAssessment:
    core_failure_points: str
    regret_probability: float
    counter_argument: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "core_failure_points": self.core_failure_points,
            "regret_probability": self.regret_probability,
            "counter_argument": self.counter_argument,
        }


def _cohere_generate(client: Any, prompt: str, temperature: float = 0.2) -> str:
    """
    Call Cohere and return the generated text.
    Tries generate() first (v1), then chat() (v1 or v2).
    """
    # 1) Legacy generate (v1 Client)
    try:
        response = client.generate(
            model="command-r-plus",
            prompt=prompt,
            temperature=temperature,
            max_tokens=1024,
        )
        return response.generations[0].text.strip()
    except (AttributeError, TypeError, Exception):
        pass

    # 2) v1 chat (message=, response.text)
    try:
        response = client.chat(
            model="command-r-plus",
            message=prompt,
            temperature=temperature,
        )
        return getattr(response, "text", None) or str(response).strip()
    except Exception:
        pass

    # 3) ClientV2-style: if client has chat(messages=...)
    try:
        response = client.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        if hasattr(response, "text"):
            return response.text.strip()
        if hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content
            if isinstance(content, list) and len(content) > 0:
                return getattr(content[0], "text", str(content[0])).strip()
            return str(content).strip()
    except Exception:
        pass

    raise RuntimeError(
        "Cohere API call failed. Ensure COHERE_API_KEY is set and the cohere package supports generate or chat."
    )


def _analyst_step(complaints_text: str) -> str:
    """
    Step 1 – The Analyst:
    Summarise the core failure points appearing in the complaints.
    """
    client = _get_cohere_client()

    prompt = (
        "You are The Analyst inside the Regret Guard AI team. "
        "Your job is to read raw customer complaints and extract the 3–5 "
        "most important 'Core Failure Points' as bullet points, grouped by theme. "
        "Focus on durability, price/value, reliability, delivery, customer support, "
        "and expectation mismatch.\n\n"
        "Example input:\n"
        "Complaints:\n"
        "- The headphones broke after two weeks.\n"
        "- Sound is tinny and cheap compared to the price.\n"
        "- Customer support refused to help when the left ear stopped working.\n"
        "- The advertised battery life is 30h but I barely get 8h.\n\n"
        "Example output:\n"
        "Core Failure Points:\n"
        "• Durability & build quality: several users report the headphones breaking within weeks.\n"
        "• Audio quality vs price: perceived as cheap and tinny for the advertised premium price.\n"
        "• Battery life mismatch: real‑world battery life is much lower than advertised.\n"
        "• Customer support: unhelpful response when defects appear, amplifying frustration.\n\n"
        "Now process these complaints:\n\n"
        f"Complaints:\n{complaints_text}\n\n"
        "Summarise the Core Failure Points."
    )

    return _cohere_generate(client, prompt, temperature=0.2)


def _negotiator_step(
    product_query: str, user_reason: str, core_failure_points: str
) -> RegretAssessment:
    """
    Step 2 – The Negotiator:
    Combines the user's reason to buy + failure points into a structured
    regret probability and counter‑argument.
    """
    client = _get_cohere_client()

    prompt = (
        "You are The Negotiator inside Regret Guard AI. "
        "Your tone is calm, protective, and slightly skeptical – like a financial therapist "
        "who wants to prevent future regret.\n\n"
        "You MUST respond with ONLY valid JSON (no markdown, no extra text) with these keys:\n"
        '"regret_probability": <0-100 integer>, '
        '"core_failure_points": "short summary", '
        '"counter_argument": "short but sharp warning"\n\n'
        "regret_probability = likelihood the user will regret this purchase "
        "given their motivation and the extracted failure points.\n\n"
        "Example 1 input:\n"
        '{"product": "Impulse gaming console in a flash sale", '
        '"user_reason": "All my friends are buying it tonight and it\'s 40% off.", '
        '"core_failure_points": "Low long‑term usage, repetitive experience, budget strain."}\n\n'
        "Example 1 output:\n"
        '{"regret_probability": 78, '
        '"core_failure_points": "Low long‑term usage, repetitive experience, and clear budget strain.", '
        '"counter_argument": "Right now you are driven by FOMO and a time‑limited discount. '
        'Most people like you stopped using this console quickly. If you wait a month and still want it, the decision will be calmer and safer."}\n\n'
        "Example 2 input:\n"
        '{"product": "Ergonomic office chair", '
        '"user_reason": "My back hurts and my current chair is very old.", '
        '"core_failure_points": "Minor delivery delays, unclear assembly."}\n\n'
        "Example 2 output:\n"
        '{"regret_probability": 18, '
        '"core_failure_points": "Minor friction around delivery and assembly.", '
        '"counter_argument": "Your motivation is health‑driven and stable. Complaints are mostly one‑time hassles. This looks like a low‑regret improvement."}\n\n'
        "Now respond with JSON only for this input:\n"
        + json.dumps(
            {
                "product": product_query,
                "user_reason": user_reason,
                "core_failure_points": core_failure_points,
            },
            indent=2,
        )
    )

    raw = _cohere_generate(client, prompt, temperature=0.25)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(raw[start : end + 1])
        else:
            raise

    return RegretAssessment(
        core_failure_points=parsed.get("core_failure_points", core_failure_points),
        regret_probability=float(parsed.get("regret_probability", 50)),
        counter_argument=parsed.get("counter_argument", ""),
    )


def run_regret_chain(
    product_query: str, user_reason: str, complaints_text: str
) -> Dict[str, object]:
    """
    Public entry point for the two‑step Regret Guard chain.

    1. The Analyst → summarises complaints into core failure points.
    2. The Negotiator → combines user context + failure points and
       outputs a regret probability and counter‑argument.
    """
    if not complaints_text.strip():
        return {
            "core_failure_points": "",
            "regret_probability": None,
            "counter_argument": "",
        }

    core_failure_points = _analyst_step(complaints_text)
    assessment = _negotiator_step(product_query, user_reason, core_failure_points)
    return assessment.to_dict()

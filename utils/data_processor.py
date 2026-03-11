import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_REVIEWS_CACHE: Optional[pd.DataFrame] = None


def _load_reviews() -> pd.DataFrame:
    """
    Lazy-load and cache the amazon_reviews.csv file.
    """
    global _REVIEWS_CACHE

    if _REVIEWS_CACHE is None:
        base_dir = Path(__file__).resolve().parents[1]
        csv_path = base_dir / "data" / "amazon_reviews.csv"
        df = pd.read_csv(csv_path)

        # Extract numeric rating from strings like "Rated 1 out of 5 stars"
        df["rating_value"] = (
            df["Rating"]
            .astype(str)
            .str.extract(r"Rated\s+(\d)\s+out", expand=False)
            .astype(float)
        )

        # Normalised helper columns for simple keyword search
        df["review_title_lower"] = df["Review Title"].astype(str).str.lower()
        df["review_text_lower"] = df["Review Text"].astype(str).str.lower()

        _REVIEWS_CACHE = df

    return _REVIEWS_CACHE


def _score_match(row: pd.Series, query_tokens: List[str]) -> int:
    """
    Simple keyword match score: counts how many query tokens appear
    in the concatenated title + text for the row.
    """
    haystack = f"{row['review_title_lower']} {row['review_text_lower']}"
    return sum(1 for token in query_tokens if token in haystack)


def get_product_insights(product_query: str, max_reviews: int = 5) -> Dict[str, object]:
    """
    Retrieve complaint evidence for a given product/query string.

    1. Loads amazon_reviews.csv (cached).
    2. Filters to 1★ and 2★ reviews only.
    3. Uses simple keyword matching over title + text to find rows
       most related to the `product_query`.
    4. Returns:
       - matched_query: the original query (for display).
       - complaints_text: concatenated string of the most descriptive complaints.
       - raw_reviews: list of individual complaint texts (for "Real‑World Evidence").
    """
    query = (product_query or "").strip()
    df = _load_reviews()

    # Only low ratings: 1★ and 2★
    low_df = df[df["rating_value"].isin([1, 2])].copy()
    if low_df.empty:
        return {"matched_query": query, "complaints_text": "", "raw_reviews": []}

    if query:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        if tokens:
            low_df["match_score"] = low_df.apply(_score_match, axis=1, query_tokens=tokens)
            # Keep rows with at least one token match; fall back to all lows if none
            matched = low_df[low_df["match_score"] > 0]
            if matched.empty:
                matched = low_df
        else:
            matched = low_df
    else:
        matched = low_df

    # Rank by a simple proxy for "descriptive": length of review text
    matched = matched.copy()
    matched["text_len"] = matched["Review Text"].astype(str).str.len()
    matched = matched.sort_values(by=["match_score", "text_len"], ascending=[False, False])

    top = matched.head(max_reviews)
    raw_reviews = top["Review Text"].astype(str).tolist()

    # Concatenate into a single text block for the LLM, with clear separators
    complaints_text = "\n\n---\n\n".join(textwrap.fill(r, width=400) for r in raw_reviews)

    return {
        "matched_query": query or "General Amazon experience",
        "complaints_text": complaints_text,
        "raw_reviews": raw_reviews,
    }


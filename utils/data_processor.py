import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_REVIEWS_CACHE: Optional[pd.DataFrame] = None
_LOAD_ERROR: Optional[str] = None  # Hint when file missing/empty/wrong shape


def _find_reviews_csv(base_dir: Path) -> Path:
    """Try amazon_reviews.csv then amazon_Reviews.csv (case variants)."""
    for name in ("amazon_reviews.csv", "amazon_Reviews.csv"):
        p = base_dir / "data" / name
        if p.exists():
            return p
    return base_dir / "data" / "amazon_reviews.csv"  # may not exist


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names to Rating, Review Title, Review Text."""
    col_map = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("rating", "overall", "stars"):
            col_map[col] = "Rating"
        elif c in ("review title", "review_title", "title", "summary", "summary_"):
            col_map[col] = "Review Title"
        elif c in ("review text", "review_text", "body", "review_body", "content", "review"):
            col_map[col] = "Review Text"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def _load_reviews() -> pd.DataFrame:
    """
    Lazy-load and cache the amazon_reviews.csv file.
    Tries amazon_reviews.csv and amazon_Reviews.csv. Handles empty file and column variants.
    """
    global _REVIEWS_CACHE, _LOAD_ERROR
    _LOAD_ERROR = None

    if _REVIEWS_CACHE is not None:
        return _REVIEWS_CACHE

    base_dir = Path(__file__).resolve().parents[1]
    csv_path = _find_reviews_csv(base_dir)
    if not csv_path.exists():
        _LOAD_ERROR = "File not found: data/amazon_reviews.csv (or data/amazon_Reviews.csv). Add a CSV with columns: Rating, Review Title, Review Text."
        _REVIEWS_CACHE = pd.DataFrame()
        return _REVIEWS_CACHE

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        err = str(e).strip().lower()
        # If main file is empty or unreadable, try sample file so the user gets a result
        sample_path = base_dir / "data" / "amazon_reviews_sample.csv"
        if sample_path.exists():
            try:
                df = pd.read_csv(sample_path)
                if not df.empty and len(df.columns) >= 3:
                    _REVIEWS_CACHE = None  # allow processing below
                    # Fall through to normalization (we'll set cache at the end)
                else:
                    df = None
            except Exception:
                df = None
        else:
            df = None
        if df is None:
            if "no columns to parse" in err or "empty" in err:
                _LOAD_ERROR = (
                    "The CSV file is empty or has no header row. Add a header: "
                    "Rating, Review Title, Review Text. See data/amazon_reviews_sample.csv for an example."
                )
            else:
                _LOAD_ERROR = f"Could not read CSV: {e}"
            _REVIEWS_CACHE = pd.DataFrame()
            return _REVIEWS_CACHE
    else:
        if df.empty or (hasattr(df, "columns") and len(df.columns) == 0):
            sample_path = base_dir / "data" / "amazon_reviews_sample.csv"
            if sample_path.exists():
                try:
                    df = pd.read_csv(sample_path)
                    if not df.empty and len(df.columns) >= 3:
                        pass  # use sample, fall through
                    else:
                        _LOAD_ERROR = "data/amazon_reviews.csv is empty. Add 1–3★ reviews with columns: Rating, Review Title, Review Text."
                        _REVIEWS_CACHE = pd.DataFrame()
                        return _REVIEWS_CACHE
                except Exception:
                    _LOAD_ERROR = "data/amazon_reviews.csv is empty. Add 1–3★ reviews with columns: Rating, Review Title, Review Text."
                    _REVIEWS_CACHE = pd.DataFrame()
                    return _REVIEWS_CACHE
            else:
                _LOAD_ERROR = "data/amazon_reviews.csv is empty. Add 1–3★ reviews with columns: Rating, Review Title, Review Text."
                _REVIEWS_CACHE = pd.DataFrame()
                return _REVIEWS_CACHE

    if df.empty or len(df) == 0:
        _LOAD_ERROR = "data/amazon_reviews.csv is empty. Add 1–3★ reviews with columns: Rating, Review Title, Review Text."
        _REVIEWS_CACHE = pd.DataFrame()
        return _REVIEWS_CACHE

    df = _normalize_columns(df)
    for required in ("Rating", "Review Title", "Review Text"):
        if required not in df.columns:
            _LOAD_ERROR = f"CSV must have columns: Rating, Review Title, Review Text. Found: {list(df.columns)}."
            _REVIEWS_CACHE = pd.DataFrame()
            return _REVIEWS_CACHE

    # Rating: numeric 1–5 or "Rated X out of 5 stars"
    raw = df["Rating"].astype(str)
    numeric = raw.str.extract(r"Rated\s+(\d)\s+out", expand=False)
    if numeric.notna().any():
        df["rating_value"] = numeric.astype(float)
    else:
        # Try plain numbers
        df["rating_value"] = pd.to_numeric(raw.replace("", float("nan")), errors="coerce")
    df["rating_value"] = df["rating_value"].clip(1, 5)

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


def get_product_insights(
    product_query: str, max_reviews: int = 5, min_low_rating_count: int = 3
) -> Dict[str, object]:
    """
    Retrieve complaint evidence for a given product/query string.

    1. Loads amazon_reviews.csv (cached).
    2. Prefers 1–2★ reviews; if there are fewer than min_low_rating_count, also
       includes 3★ reviews for more data. If there are no 1–2★ at all, uses 2–3★.
    3. Uses simple keyword matching over title + text to find rows
       most related to the `product_query`.
    4. Returns:
       - matched_query: the original query (for display).
       - complaints_text: concatenated string of the most descriptive complaints.
       - raw_reviews: list of individual complaint texts (for "Real‑World Evidence").
       - rating_band: "1-2", "1-3", or "2-3" (which star band was used).
    """
    query = (product_query or "").strip()
    df = _load_reviews()
    if df.empty or "rating_value" not in df.columns:
        reason = _LOAD_ERROR or "No review data loaded. Add data/amazon_reviews.csv with columns: Rating, Review Title, Review Text."
        return {"matched_query": query, "complaints_text": "", "raw_reviews": [], "empty_reason": reason, "rating_band": None}

    low_12 = df[df["rating_value"].isin([1.0, 2.0])].copy()
    mid_23 = df[df["rating_value"].isin([2.0, 3.0])].copy()

    # Prefer 1–2★; if not enough, use 1–3★ (add 3★). If no 1–2★ at all, use 2–3★.
    if len(low_12) >= min_low_rating_count:
        use_df = low_12
        rating_band = "1-2"
    elif not low_12.empty:
        # Few 1–2★: add 3★ for more data (1–3★ band)
        use_df = df[df["rating_value"].isin([1.0, 2.0, 3.0])].copy()
        rating_band = "1-3"
    elif not mid_23.empty:
        use_df = mid_23
        rating_band = "2-3"
    else:
        reason = _LOAD_ERROR or "No 1–2★ or 2–3★ reviews in the file. Add rows with Rating 1, 2, or 3."
        return {
            "matched_query": query,
            "complaints_text": "",
            "raw_reviews": [],
            "empty_reason": reason,
            "rating_band": None,
        }

    if query:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        if tokens:
            use_df["match_score"] = use_df.apply(_score_match, axis=1, query_tokens=tokens)
            matched = use_df[use_df["match_score"] > 0]
            if matched.empty:
                matched = use_df
        else:
            matched = use_df
    else:
        matched = use_df

    # Rank by match score then by length (most descriptive)
    matched = matched.copy()
    matched["text_len"] = matched["Review Text"].astype(str).str.len()
    matched = matched.sort_values(by=["match_score", "text_len"], ascending=[False, False])

    top = matched.head(max_reviews)
    raw_reviews = top["Review Text"].astype(str).tolist()

    complaints_text = "\n\n---\n\n".join(textwrap.fill(r, width=400) for r in raw_reviews)

    return {
        "matched_query": query or "General Amazon experience",
        "complaints_text": complaints_text,
        "raw_reviews": raw_reviews,
        "empty_reason": None,
        "rating_band": rating_band,
    }


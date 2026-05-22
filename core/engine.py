"""
Recommendation engine — hybrid content-based + preference scoring + diversity re-ranking.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ─── Cosine similarity ─────────────────────────────────────────────────────────
def compute_cosine_scores(user_vec: np.ndarray, feature_matrix: np.ndarray) -> np.ndarray:
    """Returns cosine similarity score (0-1) for every item."""
    user_norm = np.linalg.norm(user_vec)
    if user_norm == 0:
        return np.zeros(len(feature_matrix))
    scores = cosine_similarity(user_vec.reshape(1, -1), feature_matrix)[0]
    return np.clip(scores, 0, 1)


# ─── Preference matching score ─────────────────────────────────────────────────
def compute_preference_score(df: pd.DataFrame, preferences: dict) -> np.ndarray:
    """
    Explicit preference matching on genre, mood, domain.
    Returns a 0-1 boost per item.
    """
    scores = np.zeros(len(df))

    genres  = [g.lower() for g in preferences.get("genres",  [])]
    moods   = [m.lower() for m in preferences.get("moods",   [])]
    domains = [d.lower() for d in preferences.get("domains", [])]

    for i, row in df.iterrows():
        match = 0.0
        total = 0.0
        if genres:
            total += 1
            if row["genre"].lower() in genres or row["subgenre"].lower() in genres:
                match += 1
        if moods:
            total += 1
            if row["mood"].lower() in moods:
                match += 1
        if domains:
            total += 1
            if row["domain"].lower() in domains:
                match += 1
        scores[i] = (match / total) if total > 0 else 0.5

    return scores


# ─── Weighted hybrid score ─────────────────────────────────────────────────────
def compute_hybrid_score(
    cosine_scores: np.ndarray,
    pref_scores: np.ndarray,
    df: pd.DataFrame,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Combines cosine similarity, preference match, item rating, and popularity
    into a final hybrid recommendation score.
    """
    w = weights or {"cosine": 0.45, "pref": 0.30, "rating": 0.15, "popularity": 0.10}

    rating_scores     = df["rating_norm"].values
    popularity_scores = df["popularity_norm"].values

    hybrid = (
        w["cosine"]     * cosine_scores +
        w["pref"]       * pref_scores +
        w["rating"]     * rating_scores +
        w["popularity"] * popularity_scores
    )
    return np.clip(hybrid, 0, 1)


# ─── Diversity re-ranking (MMR) ────────────────────────────────────────────────
def diversity_rerank(
    df: pd.DataFrame,
    hybrid_scores: np.ndarray,
    feature_matrix: np.ndarray,
    top_n: int = 10,
    lambda_: float = 0.7,
) -> pd.DataFrame:
    """
    Maximal Marginal Relevance re-ranking to balance relevance and diversity.
    lambda_=1.0 → pure relevance; 0.0 → pure diversity.
    """
    selected_indices = []
    remaining = list(range(len(df)))

    while len(selected_indices) < min(top_n, len(df)):
        best_idx, best_score = -1, -np.inf
        for idx in remaining:
            relevance = hybrid_scores[idx]
            if selected_indices:
                sim_to_selected = cosine_similarity(
                    feature_matrix[idx].reshape(1, -1),
                    feature_matrix[selected_indices],
                ).max()
            else:
                sim_to_selected = 0.0
            mmr = lambda_ * relevance - (1 - lambda_) * sim_to_selected
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    result = df.iloc[selected_indices].copy()
    result["cosine_score"]  = hybrid_scores[selected_indices]   # reuse for display
    result["hybrid_score"]  = hybrid_scores[selected_indices]
    result["confidence_pct"] = (hybrid_scores[selected_indices] * 100).round(1)
    result["rank"] = range(1, len(result) + 1)
    return result.reset_index(drop=True)


# ─── Cold-start handler ────────────────────────────────────────────────────────
def cold_start_recommendations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    For brand-new users with no preferences: return top-rated popular items,
    ensuring domain diversity.
    """
    df = df.copy()
    df["cold_score"] = 0.6 * df["rating_norm"] + 0.4 * df["popularity_norm"]
    result = (
        df.sort_values("cold_score", ascending=False)
          .groupby("domain")
          .head(3)
          .sort_values("cold_score", ascending=False)
          .head(top_n)
    )
    result["hybrid_score"]  = result["cold_score"]
    result["confidence_pct"] = (result["cold_score"] * 100).round(1)
    result["rank"] = range(1, len(result) + 1)
    return result.reset_index(drop=True)
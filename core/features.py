"""
Feature engineering — builds TF-IDF + numeric feature matrix for cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


def build_feature_matrix(df: pd.DataFrame):
    """
    Combines:
      • TF-IDF on composite text (title + genre + subgenre + mood + tags)
      • One-hot encoded domain
      • Normalized rating, popularity, year
    Returns the feature matrix and the vectorizer for query projection.
    """
    # ── Text corpus ──────────────────────────────────────────────────────────
    df = df.copy()
    df["corpus"] = (
        df["title"] + " " + df["genre"] + " " + df["subgenre"] + " " +
        df["mood"] + " " + df["tags"] + " " + df["domain"]
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=300, sublinear_tf=True)
    text_matrix = tfidf.fit_transform(df["corpus"]).toarray()

    # ── Domain one-hot ────────────────────────────────────────────────────────
    domain_dummies = pd.get_dummies(df["domain"], prefix="dom").values.astype(float)

    # ── Numeric features ──────────────────────────────────────────────────────
    numeric = df[["rating_norm", "popularity_norm", "year_norm"]].values

    # ── Weighted combination ──────────────────────────────────────────────────
    # Text features carry most signal; domain adds domain boundary;
    # numerics add quality & recency bias.
    feature_matrix = np.hstack([
        text_matrix * 1.0,
        domain_dummies * 0.6,
        numeric * 0.4,
    ])

    return feature_matrix, tfidf, df


def build_user_vector(preferences: dict, tfidf: TfidfVectorizer, n_domain: int) -> np.ndarray:
    """
    Builds a user preference vector in the same space as item vectors.
    preferences keys: genres, moods, domains, keywords, fav_rating (0-10)
    """
    text_query = " ".join(
        preferences.get("genres", []) +
        preferences.get("moods", []) +
        preferences.get("domains", []) +
        preferences.get("keywords", [])
    )
    text_vec = tfidf.transform([text_query]).toarray()[0] if text_query.strip() else np.zeros(tfidf.max_features or 300)

    domain_vec = np.zeros(n_domain)
    user_rating_norm = preferences.get("fav_rating", 8.0) / 10.0
    numeric_vec = np.array([user_rating_norm, 0.85, 0.6])  # rating, popularity, mid-year preference

    user_vec = np.hstack([text_vec * 1.0, domain_vec * 0.6, numeric_vec * 0.4])
    return user_vec
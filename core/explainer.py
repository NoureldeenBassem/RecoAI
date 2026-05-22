"""
Generates human-readable explanation for each recommendation.
"""

import pandas as pd


def explain(row: pd.Series, preferences: dict) -> str:
    reasons = []

    user_genres  = [g.lower() for g in preferences.get("genres",  [])]
    user_moods   = [m.lower() for m in preferences.get("moods",   [])]
    user_domains = [d.lower() for d in preferences.get("domains", [])]

    if row["genre"].lower() in user_genres or row["subgenre"].lower() in user_genres:
        reasons.append(f"matches your interest in {row['genre']}")

    if row["mood"].lower() in user_moods:
        reasons.append(f"fits your {row['mood'].lower()} mood preference")

    if row["domain"].lower() in user_domains:
        reasons.append(f"is in your preferred domain ({row['domain']})")

    if row["rating"] >= 9.0:
        reasons.append(f"is critically acclaimed (rated {row['rating']}/10)")
    elif row["rating"] >= 8.5:
        reasons.append(f"is highly rated ({row['rating']}/10)")

    if row["popularity"] >= 95:
        reasons.append("is extremely popular")

    kws = preferences.get("keywords", [])
    matched_tags = [k for k in kws if k.lower() in row["tags"]]
    if matched_tags:
        reasons.append(f"aligns with your keywords: {', '.join(matched_tags)}")

    if not reasons:
        reasons.append("is a top-quality item in its category")

    return " | ".join(reasons[:3])
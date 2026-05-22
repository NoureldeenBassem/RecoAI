"""
Session-based history and favorites — persists within a run, exportable to CSV.
"""

import json
import os
from datetime import datetime
import pandas as pd


class SessionManager:
    def __init__(self):
        self.history: list[dict] = []      # past recommendation sessions
        self.favorites: list[dict] = []    # user-saved favorites
        self.feedback: dict[str, int] = {} # item_id → thumbs (1 up, -1 down)

    # ── History ────────────────────────────────────────────────────────────────
    def log_session(self, preferences: dict, recommendations: pd.DataFrame):
        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "preferences": preferences,
            "top_picks": recommendations["title"].head(5).tolist(),
        })

    def get_history_summary(self) -> list[dict]:
        return self.history

    # ── Favorites ──────────────────────────────────────────────────────────────
    def add_favorite(self, item: pd.Series):
        if not any(f["id"] == item["id"] for f in self.favorites):
            self.favorites.append({
                "id": item["id"],
                "title": item["title"],
                "domain": item["domain"],
                "genre": item["genre"],
                "rating": item["rating"],
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    def remove_favorite(self, item_id: str):
        self.favorites = [f for f in self.favorites if f["id"] != item_id]

    def get_favorites_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.favorites) if self.favorites else pd.DataFrame()

    # ── Feedback ───────────────────────────────────────────────────────────────
    def add_feedback(self, item_id: str, value: int):
        """value: 1 (like) or -1 (dislike)"""
        self.feedback[item_id] = value

    def get_liked_ids(self) -> list[str]:
        return [k for k, v in self.feedback.items() if v == 1]

    def get_disliked_ids(self) -> list[str]:
        return [k for k, v in self.feedback.items() if v == -1]

    # ── Export ─────────────────────────────────────────────────────────────────
    def export_recommendations(self, recs: pd.DataFrame, path: str = "output/recommendations.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = ["rank", "title", "domain", "genre", "mood", "rating", "confidence_pct", "year"]
        recs[[c for c in cols if c in recs.columns]].to_csv(path, index=False)
        return path
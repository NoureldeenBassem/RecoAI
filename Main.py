#!/usr/bin/env python3
"""
RecoAI — Hybrid AI Recommendation System
DecodeLabs Internship · Project 3
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import load_dataset
from core.features  import build_feature_matrix, build_user_vector
from core.engine    import (compute_cosine_scores, compute_preference_score,
                            compute_hybrid_score, diversity_rerank,
                            cold_start_recommendations)
from core.explainer import explain
from utils.session  import SessionManager
from utils.display  import (
    clear, print_logo, header, divider, info, success, warn, error,
    input_prompt, loading, numbered_menu, multi_select_menu,
    print_recommendations, print_favorites, print_history,
)

# ─── Valid option sets ─────────────────────────────────────────────────────────
GENRES  = ["Sci-Fi", "Action", "Drama", "Comedy", "Thriller", "Mystery",
           "Romance", "Animation", "Fantasy", "Hip-Hop", "Rock", "Pop",
           "Classical", "Ambient", "R&B", "RPG", "Platformer", "Simulation",
           "Puzzle", "Sandbox", "Self-Help", "Psychology", "History"]
MOODS   = ["Intense", "Emotional", "Lighthearted", "Epic", "Calm",
           "Motivational", "Energetic"]
DOMAINS = ["Movie", "Book", "Music", "Game"]


# ─── Preference collection wizard ─────────────────────────────────────────────
def collect_preferences(session: SessionManager) -> dict:
    header("🛠   PREFERENCE WIZARD")

    # Domain
    domains = multi_select_menu("Which domains interest you?", DOMAINS)

    # Genre
    genres = multi_select_menu("Select your favorite genres (multiple OK)", GENRES)

    # Mood
    moods = multi_select_menu("What mood are you in?", MOODS)

    # Keywords
    print()
    kw_raw = input_prompt("Enter keywords (e.g. space, mystery, piano) or press Enter to skip", "")
    keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]

    # Minimum rating
    min_rating_raw = input_prompt("Minimum item rating (1–10)", "7.5")
    try:
        min_rating = float(min_rating_raw)
    except ValueError:
        min_rating = 7.5

    # Top-N
    top_n_raw = input_prompt("How many recommendations would you like?", "10")
    try:
        top_n = max(1, min(20, int(top_n_raw)))
    except ValueError:
        top_n = 10

    # Diversity slider
    diversity_raw = input_prompt(
        "Diversity vs Relevance? (1=pure relevance, 5=max diversity)", "3"
    )
    try:
        div_level = max(1, min(5, int(diversity_raw)))
    except ValueError:
        div_level = 3
    lambda_ = 1.0 - (div_level - 1) * 0.15  # maps 1→1.0, 5→0.4

    preferences = {
        "domains":    domains,
        "genres":     genres,
        "moods":      moods,
        "keywords":   keywords,
        "min_rating": min_rating,
        "fav_rating": min_rating,
        "top_n":      top_n,
        "lambda_":    lambda_,
    }
    return preferences


# ─── Core recommendation pipeline ─────────────────────────────────────────────
def run_recommendation(df, feature_matrix, tfidf, n_domain,
                       preferences: dict, session: SessionManager):
    loading("Computing recommendations")

    # Filter by min_rating
    mask = df["rating"] >= preferences.get("min_rating", 0)
    filtered_df = df[mask].reset_index(drop=True)
    filtered_feat = feature_matrix[mask]

    if len(filtered_df) == 0:
        warn("No items match that minimum rating — relaxing filter.")
        filtered_df = df.reset_index(drop=True)
        filtered_feat = feature_matrix

    # Check for cold-start (no real preferences)
    has_prefs = any([preferences.get("genres"), preferences.get("moods"),
                     preferences.get("domains"), preferences.get("keywords")])
    top_n = preferences.get("top_n", 10)

    if not has_prefs:
        recs = cold_start_recommendations(filtered_df, top_n=top_n)
        info("Using trending-based recommendations (cold start).")
    else:
        user_vec    = build_user_vector(preferences, tfidf, n_domain)
        cos_scores  = compute_cosine_scores(user_vec, filtered_feat)
        pref_scores = compute_preference_score(filtered_df, preferences)
        hybrid      = compute_hybrid_score(cos_scores, pref_scores, filtered_df)

        # Filter out disliked items from past feedback
        disliked = session.get_disliked_ids()
        for i, row in filtered_df.iterrows():
            if row["id"] in disliked:
                hybrid[i] *= 0.1  # suppress strongly

        recs = diversity_rerank(filtered_df, hybrid, filtered_feat,
                                top_n=top_n, lambda_=preferences.get("lambda_", 0.7))

    # Generate explanations
    explanations = [explain(row, preferences) for _, row in recs.iterrows()]
    session.log_session(preferences, recs)
    return recs, explanations


# ─── Post-recommendation menu ──────────────────────────────────────────────────
def post_rec_menu(recs, session: SessionManager, preferences: dict,
                  df, feature_matrix, tfidf, n_domain):

    while True:
        choice = numbered_menu("What would you like to do?", [
            "⭐  Save a favorite",
            "👍 / 👎  Give feedback on an item",
            "📊  Open analytics dashboard",
            "📁  Export recommendations to CSV",
            "🔄  New recommendation search",
            "🏠  Return to main menu",
        ])

        if choice == 1:
            raw = input_prompt("Enter the rank number(s) to save (e.g. 1,3)", "1")
            for r in raw.split(","):
                r = r.strip()
                if r.isdigit():
                    idx = int(r) - 1
                    if 0 <= idx < len(recs):
                        session.add_favorite(recs.iloc[idx])
                        success(f"'{recs.iloc[idx]['title']}' saved to favorites!")

        elif choice == 2:
            raw = input_prompt("Enter rank number", "1")
            if raw.isdigit() and 1 <= int(raw) <= len(recs):
                row = recs.iloc[int(raw) - 1]
                fb  = numbered_menu(f"Feedback for '{row['title']}':", ["👍 Like", "👎 Dislike"])
                session.add_feedback(row["id"], 1 if fb == 1 else -1)
                success("Feedback recorded! It will influence future recommendations.")

        elif choice == 3:
            try:
                from utils.visualizer import plot_dashboard
                info("Generating analytics dashboard...")
                path = plot_dashboard(recs, preferences)
                success(f"Dashboard saved → {path}")
            except Exception as exc:
                error(f"Could not generate dashboard: {exc}")

        elif choice == 4:
            path = session.export_recommendations(recs)
            success(f"Exported → {path}")

        elif choice == 5:
            prefs = collect_preferences(session)
            recs, explanations = run_recommendation(
                df, feature_matrix, tfidf, n_domain, prefs, session
            )
            print_recommendations(recs, explanations)

        elif choice == 6:
            break


# ─── Main menu ─────────────────────────────────────────────────────────────────
def main():
    clear()
    print_logo()
    info("Loading dataset and building feature matrix...")

    df = load_dataset()
    feature_matrix, tfidf, df = build_feature_matrix(df)

    # Determine domain count (for user vector construction)
    n_domain = len(df["domain"].unique())

    session = SessionManager()

    time.sleep(0.3)
    success(f"Dataset loaded: {len(df)} items across {n_domain} domains")
    divider()

    while True:
        choice = numbered_menu("🏠  MAIN MENU", [
            "🎯  Get Personalized Recommendations",
            "⭐  View Saved Favorites",
            "📋  View Session History",
            "ℹ   About This System",
            "🚪  Exit",
        ])

        if choice == 1:
            preferences = collect_preferences(session)
            recs, explanations = run_recommendation(
                df, feature_matrix, tfidf, n_domain, preferences, session
            )
            print_recommendations(recs, explanations)
            post_rec_menu(recs, session, preferences,
                          df, feature_matrix, tfidf, n_domain)

        elif choice == 2:
            print_favorites(session.favorites)

        elif choice == 3:
            print_history(session.get_history_summary())

        elif choice == 4:
            header("ℹ   ABOUT RECO AI")
            info("Algorithm   : Hybrid (Content-Based + Cosine Similarity + Preference Scoring)")
            info("Re-ranking  : Maximal Marginal Relevance (diversity optimization)")
            info("Cold-start  : Popularity × Rating fallback for new users")
            info("Feedback    : Session-level like/dislike suppression")
            info("Dataset     : 37 curated items — Movies, Books, Music, Games")
            info("Developer   : DecodeLabs AI Intern · Project 3")
            print()

        elif choice == 5:
            success("Thank you for using RecoAI. Goodbye! 👋")
            sys.exit(0)


if __name__ == "__main__":
    main()
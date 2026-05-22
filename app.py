"""
RecoAI — Streamlit Web Interface
Run with: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from data.dataset import load_dataset
from core.features import build_feature_matrix, build_user_vector
from core.engine import (
    compute_cosine_scores, compute_preference_score,
    compute_hybrid_score, diversity_rerank, cold_start_recommendations,
)
from core.explainer import explain

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecoAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: #0F1117;
    color: #E8EAF6;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #13151f;
    border-right: 1px solid #1e2130;
}
/* ---------- hide streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }

/* ---------- section headers ---------- */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #7C83FD;
    letter-spacing: .07em;
    text-transform: uppercase;
    margin: 1.4rem 0 .5rem;
}

/* ---------- rec card ---------- */
.rec-card {
    background: #1A1D27;
    border: 1px solid #252840;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .85rem;
    transition: border .2s;
}
.rec-card:hover { border-color: #7C83FD55; }

.card-rank {
    font-size: 1.5rem;
    font-weight: 800;
    color: #7C83FD;
    margin-right: .55rem;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #E8EAF6;
}
.card-meta {
    font-size: .78rem;
    color: #8890b0;
    margin-top: .18rem;
}
.card-badge {
    display: inline-block;
    padding: .18rem .55rem;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 600;
    margin-right: .35rem;
    margin-top: .35rem;
}
.badge-movie  { background:#4C72B022; color:#4C72B0; border:1px solid #4C72B044; }
.badge-book   { background:#DD845222; color:#DD8452; border:1px solid #DD845244; }
.badge-music  { background:#55A86822; color:#55A868; border:1px solid #55A86844; }
.badge-game   { background:#C44E5222; color:#C44E52; border:1px solid #C44E5244; }
.badge-genre  { background:#7C83FD18; color:#9da5ff; border:1px solid #7C83FD33; }
.badge-mood   { background:#f4c54218; color:#f4c542; border:1px solid #f4c54233; }

.confidence-label {
    font-size: .78rem;
    color: #8890b0;
    margin-top: .5rem;
    margin-bottom: .12rem;
}
.conf-bar-wrap {
    background: #252840;
    border-radius: 999px;
    height: 7px;
    width: 100%;
}
.conf-bar-fill {
    height: 7px;
    border-radius: 999px;
}
.why-text {
    font-size: .82rem;
    color: #9da5c0;
    margin-top: .55rem;
    line-height: 1.5;
    border-left: 2px solid #7C83FD55;
    padding-left: .65rem;
}

/* ---------- stat cards ---------- */
.stat-row { display:flex; gap:.8rem; margin-bottom:1rem; }
.stat-card {
    flex:1;
    background:#1A1D27;
    border:1px solid #252840;
    border-radius:12px;
    padding:.8rem 1rem;
    text-align:center;
}
.stat-value { font-size:1.6rem; font-weight:800; color:#7C83FD; }
.stat-label { font-size:.75rem; color:#8890b0; margin-top:.1rem; }

/* ---------- fav / history cards ---------- */
.fav-card {
    background:#1A1D27;
    border:1px solid #252840;
    border-radius:10px;
    padding:.7rem 1rem;
    margin-bottom:.5rem;
    font-size:.88rem;
}

/* ---------- sidebar labels ---------- */
.sidebar-section {
    font-size:.72rem;
    font-weight:700;
    color:#7C83FD;
    letter-spacing:.1em;
    text-transform:uppercase;
    margin:1.2rem 0 .3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DOMAIN_ICON  = {"Movie": "🎬", "Book": "📚", "Music": "🎵", "Game": "🎮"}
MOOD_ICON    = {"Intense": "⚡", "Emotional": "💙", "Lighthearted": "😄",
                "Epic": "🏆", "Calm": "🌊", "Motivational": "🔥", "Energetic": "🎯"}
DOMAIN_COLOR = {"Movie": "#4C72B0", "Book": "#DD8452", "Music": "#55A868", "Game": "#C44E52"}
BG, CARD, TEXT = "#0F1117", "#1A1D27", "#E8EAF6"

GENRES  = ["Sci-Fi", "Action", "Drama", "Comedy", "Thriller", "Mystery",
           "Romance", "Animation", "Fantasy", "Hip-Hop", "Rock", "Pop",
           "Classical", "Ambient", "R&B", "RPG", "Platformer", "Simulation",
           "Puzzle", "Sandbox", "Self-Help", "Psychology", "History"]
MOODS   = ["Intense", "Emotional", "Lighthearted", "Epic", "Calm", "Motivational", "Energetic"]
DOMAINS = ["Movie", "Book", "Music", "Game"]


# ── Cached data ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    df = load_dataset()
    feature_matrix, tfidf, df = build_feature_matrix(df)
    return df, feature_matrix, tfidf


# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    for key, val in {
        "favorites": [],
        "history": [],
        "feedback": {},
        "recs": None,
        "explanations": [],
        "preferences": {},
        "page": "home",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()
df, feature_matrix, tfidf = load_data()
n_domain = len(df["domain"].unique())


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(prefs: dict):
    mask = df["rating"] >= prefs.get("min_rating", 0)
    fdf  = df[mask].reset_index(drop=True)
    ffeat = feature_matrix[mask]

    if len(fdf) == 0:
        fdf, ffeat = df.reset_index(drop=True), feature_matrix

    has_prefs = any([prefs.get("genres"), prefs.get("moods"),
                     prefs.get("domains"), prefs.get("keywords")])
    top_n = prefs.get("top_n", 10)

    if not has_prefs:
        recs = cold_start_recommendations(fdf, top_n=top_n)
    else:
        uv      = build_user_vector(prefs, tfidf, n_domain)
        cos     = compute_cosine_scores(uv, ffeat)
        pref_s  = compute_preference_score(fdf, prefs)
        hybrid  = compute_hybrid_score(cos, pref_s, fdf)

        disliked = [k for k, v in st.session_state.feedback.items() if v == -1]
        for i, row in fdf.iterrows():
            if row["id"] in disliked:
                hybrid[i] *= 0.1

        recs = diversity_rerank(fdf, hybrid, ffeat,
                                top_n=top_n, lambda_=prefs.get("lambda_", 0.7))

    expls = [explain(row, prefs) for _, row in recs.iterrows()]
    st.session_state.history.append({
        "preferences": prefs,
        "top_picks": recs["title"].head(5).tolist(),
    })
    return recs, expls


# ── Confidence bar HTML ────────────────────────────────────────────────────────
def conf_bar(pct: float) -> str:
    color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 50 else "#ef4444"
    return f"""
    <div class="confidence-label">Confidence — {pct:.1f}%</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>"""


# ── Domain badge HTML ──────────────────────────────────────────────────────────
def domain_badge(domain):
    cls = f"badge-{domain.lower()}"
    icon = DOMAIN_ICON.get(domain, "")
    return f'<span class="card-badge {cls}">{icon} {domain}</span>'

def genre_badge(g):
    return f'<span class="card-badge badge-genre">{g}</span>'

def mood_badge(m):
    icon = MOOD_ICON.get(m, "")
    return f'<span class="card-badge badge-mood">{icon} {m}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎯 RecoAI")
    st.markdown("<div style='color:#8890b0;font-size:.82rem;'>Hybrid AI Recommendation Engine</div>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", ["🏠 Recommendations", "⭐ Favorites", "📋 History", "ℹ️ About"],
                    label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>⚙️ Preferences</div>", unsafe_allow_html=True)

    sel_domains = st.multiselect("Domains", DOMAINS, default=DOMAINS, key="sel_domains")
    sel_genres  = st.multiselect("Genres", GENRES, key="sel_genres")
    sel_moods   = st.multiselect("Moods", MOODS, key="sel_moods")
    keywords_raw = st.text_input("Keywords (comma-separated)", placeholder="e.g. space, piano, mystery")
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    st.markdown("<div class='sidebar-section'>🎚️ Filters</div>", unsafe_allow_html=True)
    min_rating = st.slider("Min Rating", 1.0, 10.0, 7.5, 0.1)
    top_n      = st.slider("Number of Results", 3, 20, 10)
    diversity  = st.slider("Diversity  ◀ Relevance", 1, 5, 3,
                           help="1 = most relevant, 5 = most diverse")
    lambda_ = 1.0 - (diversity - 1) * 0.15

    st.markdown("")
    run_btn = st.button("🚀  Get Recommendations", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Built by <strong>Noureldin Bassem</strong></p>
            <p style="font-size: 0.8em; color: #888;">AI Intern | DecodeLabs</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if run_btn:
        prefs = {
            "domains": sel_domains, "genres": sel_genres, "moods": sel_moods,
            "keywords": keywords, "min_rating": min_rating, "fav_rating": min_rating,
            "top_n": top_n, "lambda_": lambda_,
        }
        with st.spinner("Computing recommendations…"):
            recs, expls = run_pipeline(prefs)
        st.session_state.recs = recs
        st.session_state.explanations = expls
        st.session_state.preferences = prefs
        st.session_state.page = "🏠 Recommendations"


# ── Page: Recommendations ──────────────────────────────────────────────────────
if st.session_state.page == "🏠 Recommendations":

    if st.session_state.recs is None:
        # Landing state
        st.markdown("## Welcome to RecoAI")
        st.markdown("<div style='color:#8890b0;max-width:560px'>Set your preferences in the sidebar and hit <b>Get Recommendations</b> to receive personalized picks across Movies, Books, Music, and Games.</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, (domain, icon, color) in zip(
            [c1, c2, c3, c4],
            [("Movies", "🎬", "#4C72B0"), ("Books", "📚", "#DD8452"),
             ("Music", "🎵", "#55A868"), ("Games", "🎮", "#C44E52")]
        ):
            count = len(df[df["domain"] == domain[:-1]])  # strip plural
            col.markdown(f"""
            <div class="stat-card">
                <div style="font-size:2rem">{icon}</div>
                <div class="stat-value" style="color:{color}">{count}</div>
                <div class="stat-label">{domain}</div>
            </div>""", unsafe_allow_html=True)

    else:
        recs  = st.session_state.recs
        expls = st.session_state.explanations
        prefs = st.session_state.preferences

        # ── Summary stats ──────────────────────────────────────────────────────
        avg_conf = recs["confidence_pct"].mean()
        avg_rat  = recs["rating"].mean()
        domains_shown = recs["domain"].nunique()

        st.markdown("### 🎯 Your Recommendations")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card"><div class="stat-value">{len(recs)}</div><div class="stat-label">Results</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-value">{avg_conf:.0f}%</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-value">★ {avg_rat:.1f}</div><div class="stat-label">Avg Rating</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-value">{domains_shown}</div><div class="stat-label">Domains</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cards + Charts split ───────────────────────────────────────────────
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)
            for i, (_, row) in enumerate(recs.iterrows()):
                rank = int(row["rank"])
                rank_color = "#FFD700" if rank == 1 else "#7C83FD" if rank <= 3 else "#E8EAF6"

                badges = domain_badge(row["domain"]) + genre_badge(row["genre"]) + mood_badge(row["mood"])

                fb_key_like    = f"like_{row['id']}"
                fb_key_dislike = f"dislike_{row['id']}"
                fav_key        = f"fav_{row['id']}"

                st.markdown(f"""
                <div class="rec-card">
                  <div>
                    <span class="card-rank" style="color:{rank_color}">#{rank}</span>
                    <span class="card-title">{DOMAIN_ICON.get(row['domain'],'•')} {row['title']}</span>
                    <span style="float:right;color:#f4c542;font-size:.9rem">★ {row['rating']}</span>
                  </div>
                  <div class="card-meta">{row['subgenre']} · {int(row['year'])}</div>
                  <div style="margin-top:.4rem">{badges}</div>
                  {conf_bar(row['confidence_pct'])}
                  <div class="why-text">💡 {expls[i]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Action buttons per card
                b1, b2, b3, _ = st.columns([1, 1, 1, 3])
                item_id = row["id"]

                liked    = st.session_state.feedback.get(item_id) == 1
                disliked = st.session_state.feedback.get(item_id) == -1
                is_fav   = any(f["id"] == item_id for f in st.session_state.favorites)

                if b1.button("👍" if not liked else "✅ Liked", key=fb_key_like):
                    st.session_state.feedback[item_id] = 1
                    st.rerun()
                if b2.button("👎" if not disliked else "🚫 Disliked", key=fb_key_dislike):
                    st.session_state.feedback[item_id] = -1
                    st.rerun()
                if b3.button("⭐ Save" if not is_fav else "✅ Saved", key=fav_key):
                    if not is_fav:
                        st.session_state.favorites.append({
                            "id": row["id"], "title": row["title"],
                            "domain": row["domain"], "genre": row["genre"],
                            "rating": float(row["rating"]),
                        })
                        st.rerun()

        with right:
            st.markdown("<div class='section-title'>Analytics</div>", unsafe_allow_html=True)

            # ── Domain pie ─────────────────────────────────────────────────────
            fig1, ax1 = plt.subplots(figsize=(4, 3), facecolor=CARD)
            domain_counts = recs["domain"].value_counts()
            wedges, _, autotexts = ax1.pie(
                domain_counts.values,
                labels=domain_counts.index,
                colors=[DOMAIN_COLOR.get(d, "#888") for d in domain_counts.index],
                autopct="%1.0f%%", startangle=140,
                textprops={"color": TEXT, "fontsize": 8},
                wedgeprops={"linewidth": 0.5, "edgecolor": CARD},
            )
            for at in autotexts:
                at.set_color(CARD); at.set_fontsize(8)
            ax1.set_title("Domain Mix", color=TEXT, fontsize=10, pad=6)
            fig1.patch.set_facecolor(CARD)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            # ── Confidence bars ────────────────────────────────────────────────
            fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor=CARD)
            ax2.set_facecolor(CARD)
            colors = [DOMAIN_COLOR.get(d, "#888") for d in recs["domain"]]
            short_titles = [t[:18] + "…" if len(t) > 18 else t for t in recs["title"]]
            bars = ax2.barh(short_titles[::-1], recs["confidence_pct"].values[::-1],
                            color=colors[::-1], edgecolor="none", height=0.6)
            for bar, pct in zip(bars, recs["confidence_pct"].values[::-1]):
                ax2.text(bar.get_width() + .5, bar.get_y() + bar.get_height() / 2,
                         f"{pct:.0f}%", va="center", color=TEXT, fontsize=7)
            ax2.set_xlim(0, 108)
            ax2.set_xlabel("Confidence %", color=TEXT, fontsize=8)
            ax2.set_title("Confidence by Item", color=TEXT, fontsize=10, pad=6)
            ax2.tick_params(colors=TEXT, labelsize=7)
            ax2.spines[:].set_visible(False)
            patches = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLOR.items() if d in recs["domain"].values]
            ax2.legend(handles=patches, fontsize=7, facecolor=BG, edgecolor="none", labelcolor=TEXT, loc="lower right")
            fig2.patch.set_facecolor(CARD)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            # ── Rating scatter ─────────────────────────────────────────────────
            fig3, ax3 = plt.subplots(figsize=(4, 3), facecolor=CARD)
            ax3.set_facecolor(CARD)
            domain_list = list(DOMAIN_COLOR.keys())
            scatter_colors = [DOMAIN_COLOR.get(d, "#888") for d in recs["domain"]]
            ax3.scatter(recs["rating"], recs["confidence_pct"],
                        c=scatter_colors, s=70, alpha=0.9, edgecolors="none")
            for _, row in recs.iterrows():
                ax3.annotate(row["title"][:10], (row["rating"], row["confidence_pct"]),
                             fontsize=6, color=TEXT, alpha=0.7,
                             xytext=(3, 3), textcoords="offset points")
            ax3.set_xlabel("Rating", color=TEXT, fontsize=8)
            ax3.set_ylabel("Confidence %", color=TEXT, fontsize=8)
            ax3.set_title("Rating vs Confidence", color=TEXT, fontsize=10, pad=6)
            ax3.tick_params(colors=TEXT, labelsize=7)
            ax3.spines[:].set_color("#333")
            fig3.patch.set_facecolor(CARD)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        # ── CSV export ─────────────────────────────────────────────────────────
        st.markdown("---")
        cols = ["rank", "title", "domain", "genre", "mood", "rating", "confidence_pct", "year"]
        csv_df = recs[[c for c in cols if c in recs.columns]]
        st.download_button(
            "📥 Export to CSV",
            data=csv_df.to_csv(index=False),
            file_name="recoai_recommendations.csv",
            mime="text/csv",
        )


# ── Page: Favorites ────────────────────────────────────────────────────────────
elif st.session_state.page == "⭐ Favorites":
    st.markdown("## ⭐ Saved Favorites")
    favs = st.session_state.favorites
    if not favs:
        st.info("No favorites saved yet. Run a recommendation and click ⭐ Save on any result.")
    else:
        st.markdown(f"**{len(favs)} item(s) saved**")
        for fav in favs:
            icon = DOMAIN_ICON.get(fav.get("domain", ""), "•")
            color = DOMAIN_COLOR.get(fav.get("domain", ""), "#7C83FD")
            st.markdown(f"""
            <div class="fav-card">
              <span style="font-size:1.1rem">{icon}</span>
              <b style="margin-left:.4rem">{fav['title']}</b>
              <span style="float:right;color:#f4c542">★ {fav['rating']}</span><br>
              <span style="color:{color};font-size:.78rem">{fav.get('domain','')} · {fav.get('genre','')}</span>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear All Favorites"):
            st.session_state.favorites = []
            st.rerun()


# ── Page: History ──────────────────────────────────────────────────────────────
elif st.session_state.page == "📋 History":
    st.markdown("## 📋 Session History")
    history = st.session_state.history
    if not history:
        st.info("No history yet. Run a recommendation search first.")
    else:
        for i, h in enumerate(reversed(history), 1):
            p = h["preferences"]
            with st.expander(f"Search #{len(history) - i + 1}  —  Top: {', '.join(h['top_picks'][:3])}"):
                st.markdown(f"**Domains:** {', '.join(p.get('domains', ['-'])) or '-'}")
                st.markdown(f"**Genres:** {', '.join(p.get('genres', ['-'])) or '-'}")
                st.markdown(f"**Moods:** {', '.join(p.get('moods', ['-'])) or '-'}")
                st.markdown(f"**Keywords:** {', '.join(p.get('keywords', [])) or '—'}")
                st.markdown(f"**Min Rating:** {p.get('min_rating', '—')}  |  **Results:** {p.get('top_n', '—')}")
                st.markdown("**Top Picks:** " + " · ".join(h["top_picks"]))


# ── Page: About ────────────────────────────────────────────────────────────────
elif st.session_state.page == "ℹ️ About":
    st.markdown("## ℹ️ About RecoAI")
    st.markdown("""
**RecoAI** is a hybrid AI recommendation engine built for the DecodeLabs Internship — Project 3.

---

### Algorithm

| Stage | Technique |
|---|---|
| Feature Engineering | TF-IDF (bigrams) + Domain one-hot + Normalized numerics |
| Similarity | Cosine similarity in joint feature space |
| Scoring | Hybrid: Cosine × 0.45 + Preference × 0.30 + Rating × 0.15 + Popularity × 0.10 |
| Re-ranking | Maximal Marginal Relevance (diversity optimization) |
| Cold-start | Popularity × Rating fallback with domain balancing |
| Feedback | Session-scoped dislike suppression (10× score penalty) |

---

### Dataset
37 curated items · 4 domains · 10+ attributes per item  
Movies · Books · Music · Games

---

### Stack
`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `streamlit`

---

*DecodeLabs AI Internship · Project 3*
""")

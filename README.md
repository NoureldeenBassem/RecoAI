# RecoAI — Hybrid AI Recommendation Engine

> **DecodeLabs Internship · Project 3** | AI Recommendation Logic  
> A production-style, modular recommendation system built in pure Python.

---

## Overview

RecoAI is a multi-domain AI recommendation system that delivers personalized suggestions for **Movies, Books, Music, and Games** using a hybrid recommendation architecture. It combines content-based filtering, cosine similarity, explicit preference scoring, and diversity-aware re-ranking into a single cohesive pipeline — with professional terminal UX and an analytics dashboard.

---

## Project Structure

```
RecoAI/
├── Main.py                  # Entry point — main menu & orchestration
├── requirements.txt
├── data/
│   └── dataset.py           # 37-item curated multi-domain catalog
├── core/
│   ├── features.py          # TF-IDF + numeric feature engineering
│   ├── engine.py            # Scoring, ranking, cold-start logic
│   └── explainer.py        # Human-readable recommendation explanations
├── utils/
│   ├── display.py           # Rich terminal UI (colors, menus, cards)
│   ├── session.py           # History, favorites, feedback, CSV export
│   └── visualizer.py        # Matplotlib analytics dashboard (PNG)
├── output/                  # Auto-generated exports and dashboards
└── docs/
    └── report.md            # Technical internship report
```

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Dataset management, DataFrame operations |
| `numpy` | Vector math, matrix operations |
| `scikit-learn` | TF-IDF vectorization, cosine similarity, scaling |
| `matplotlib` | Analytics dashboard generation |
| `colorama` | Cross-platform colored terminal output |

---

## Recommendation Methodology

RecoAI uses a **four-stage hybrid pipeline**:

### 1. Feature Engineering (`core/features.py`)
- **TF-IDF Vectorization** on a composite text corpus (title + genre + subgenre + mood + tags) with bigram support (`ngram_range=(1,2)`, `max_features=300`, `sublinear_tf=True`)
- **Domain one-hot encoding** to enforce soft domain boundaries
- **Normalized numerics**: rating, popularity, release year
- Weighted combination: `text × 1.0 | domain × 0.6 | numeric × 0.4`

### 2. Scoring (`core/engine.py`)
- **Cosine Similarity**: measures alignment between user preference vector and each item vector
- **Preference Score**: explicit matching on genre, mood, and domain (0–1 per item)
- **Hybrid Score**: weighted blend of all signals
  ```
  score = 0.45·cosine + 0.30·preference + 0.15·rating + 0.10·popularity
  ```
- **Dislike Suppression**: items marked disliked by the user are scored at 10% weight

### 3. Diversity Re-ranking — MMR (`core/engine.py`)
Maximal Marginal Relevance balances relevance against intra-list similarity:
```
MMR(i) = λ · relevance(i) − (1−λ) · max_sim(i, selected)
```
The `λ` parameter is user-controlled (diversity slider 1–5 in the wizard).

### 4. Explanation Generation (`core/explainer.py`)
Each recommendation comes with a human-readable reason combining genre match, mood fit, domain preference, rating tier, popularity, and keyword alignment.

### Cold-Start Handling
For new users with no stated preferences, the system falls back to a popularity × rating composite score, ensuring domain diversity across the top results.

---

## Dataset

**37 hand-curated items** across 4 domains, each with 10+ attributes:

| Domain | Count | Examples |
|---|---|---|
| Movie | 12 | Inception, The Dark Knight, Parasite, Dune |
| Book | 9 | Dune (Novel), Atomic Habits, Project Hail Mary |
| Music | 8 | Bohemian Rhapsody, Clair de Lune, Lose Yourself |
| Game | 8 | The Witcher 3, Portal 2, Hades, Stardew Valley |

**Attributes per item**: `id`, `title`, `domain`, `genre`, `subgenre`, `mood`, `year`, `rating`, `popularity`, `tags`

---

## Features

| Feature | Status |
|---|---|
| Hybrid recommendation engine | ✅ |
| Cosine similarity scoring | ✅ |
| Content-based filtering (TF-IDF) | ✅ |
| Preference & mood-based matching | ✅ |
| Recommendation confidence % | ✅ |
| Recommendation explanations | ✅ |
| Diversity optimization (MMR) | ✅ |
| Cold-start handling | ✅ |
| Feedback learning (like/dislike) | ✅ |
| Save favorites | ✅ |
| Recommendation history | ✅ |
| Analytics dashboard (PNG) | ✅ |
| Export to CSV | ✅ |
| Interactive CLI menu | ✅ |
| Rich colored terminal UI | ✅ |

---

## How to Run

**1. Clone / copy the project folder**

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the system**
```bash
python Main.py
```

The preference wizard will guide you through domain, genre, mood, keywords, rating threshold, result count, and diversity level. No configuration files needed.

---

## Sample Output

```
  #1    🎬 Inception                        Movie     ★ 8.8
       Genre: Sci-Fi / Thriller   Mood: ⚡ Intense   Year: 2010
       Confidence: ████████████████░░░░ 70.9%
       Why: matches your interest in Sci-Fi | fits your Intense mood preference | highly rated (8.8/10)

  #2    🎬 The Matrix                        Movie     ★ 8.7
       Confidence: ███████████████░░░░░ 68.4%
       Why: matches your interest in Sci-Fi | aligns with your keywords: simulation
```

The analytics dashboard (`output/dashboard.png`) includes:
- Confidence bar chart per recommendation
- Domain distribution pie chart
- Rating vs. Confidence scatter plot
- Genre spread bar chart
- Release year histogram

---

## Architecture Notes

- **Fully modular**: each concern (data, features, scoring, display, session) is isolated in its own module
- **Stateless core**: the scoring pipeline is pure functions — no global state
- **Session-scoped feedback**: like/dislike signals are applied within a session without requiring a database
- **Scalable dataset**: adding new items requires only appending to the `ITEMS` list in `data/dataset.py`

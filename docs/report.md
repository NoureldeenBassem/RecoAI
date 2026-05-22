# RecoAI — Technical Report
**DecodeLabs AI Internship · Project 3: AI Recommendation Logic**

---

## 1. Project Overview

RecoAI is a hybrid AI recommendation system built in Python that delivers personalized item suggestions across four content domains: Movies, Books, Music, and Games. The system is designed to professional software engineering standards — modular, scalable, and fully interactive — while implementing multiple recommendation techniques that are standard in production systems at companies like Netflix, Spotify, and Goodreads.

The core challenge in recommendation systems is the **relevance-diversity tradeoff**: returning results that are highly relevant to the user's stated preferences while avoiding redundant, homogeneous outputs. RecoAI addresses this through a four-stage pipeline described below.

---

## 2. System Architecture

The project follows a clean layered architecture:

```
Input Layer     → Preference Wizard (display.py)
Data Layer      → Dataset catalog + normalization (dataset.py)
Feature Layer   → TF-IDF + one-hot + numeric encoding (features.py)
Scoring Layer   → Cosine similarity + preference matching (engine.py)
Ranking Layer   → Hybrid score + MMR re-ranking (engine.py)
Explanation Layer → Natural language justifications (explainer.py)
Persistence Layer → Session history, favorites, feedback (session.py)
Output Layer    → Terminal UI + Matplotlib dashboard (display.py, visualizer.py)
```

Each layer communicates through well-defined function interfaces with no cross-layer coupling.

---

## 3. Recommendation Pipeline

### 3.1 Feature Engineering

Item features are encoded into a unified vector space combining three signal types:

**Text Features (TF-IDF)**  
A composite text corpus is built per item: `title + genre + subgenre + mood + tags + domain`. This is vectorized using scikit-learn's `TfidfVectorizer` with:
- Bigram support (`ngram_range=(1,2)`) to capture multi-word patterns like "open world" or "dark fantasy"
- `sublinear_tf=True` to dampen the effect of very frequent terms
- `max_features=300` to control dimensionality

**Domain One-Hot Encoding**  
Domain membership (Movie, Book, Music, Game) is encoded as a binary vector and weighted at 0.6× to allow the model to respect domain preferences without making them absolute constraints.

**Normalized Numerics**  
Item rating, popularity (0–100 scale), and release year are min-max normalized and concatenated at 0.4× weight to add quality and recency signal.

The final item vector is: `[TF-IDF × 1.0 | domain_onehot × 0.6 | numerics × 0.4]`

### 3.2 User Vector Construction

The user's preferences (genres, moods, domains, keywords) are concatenated into a single query string and passed through the same fitted TF-IDF vectorizer, producing a user vector in the same feature space as all item vectors. This enables direct cosine comparison.

### 3.3 Scoring

Three scores are computed per item:

| Score | Method | Weight |
|---|---|---|
| Cosine Similarity | `sklearn.metrics.pairwise.cosine_similarity` | 45% |
| Preference Match | Exact match on genre / mood / domain | 30% |
| Item Rating | Normalized 0–1 | 15% |
| Popularity | Normalized 0–1 | 10% |

The hybrid score formula:
```
hybrid(i) = 0.45·cosine(i) + 0.30·pref(i) + 0.15·rating(i) + 0.10·pop(i)
```

### 3.4 Diversity Re-Ranking (MMR)

Raw hybrid scores would tend to cluster recommendations around a single genre or domain. Maximal Marginal Relevance (MMR) prevents this by iteratively selecting the next best item that maximizes relevance while minimizing redundancy with already-selected items:

```
MMR(i) = λ · hybrid(i) − (1 − λ) · max_{j ∈ Selected} cos_sim(i, j)
```

The λ parameter maps directly to the user's diversity slider (1 = pure relevance, 5 = maximum diversity), giving the user transparent control over output diversity.

### 3.5 Cold-Start Handling

When no preferences are provided, the system bypasses the preference pipeline and falls back to a cold-start score:
```
cold_score(i) = 0.6 · rating_norm(i) + 0.4 · popularity_norm(i)
```
Domain-grouped `head(3)` ensures representation across all four content domains.

---

## 4. Feedback Learning

The system supports session-level feedback. Disliked items have their hybrid score multiplied by 0.1 (90% suppression) in subsequent recommendation runs within the same session. This simulates lightweight implicit feedback without requiring a persistent model update cycle.

---

## 5. Analytics Dashboard

The Matplotlib dashboard (`output/dashboard.png`) renders five panels:

1. **Confidence Bar Chart** — horizontal bars per recommendation, color-coded by domain
2. **Domain Mix Pie** — proportional split across Movie / Book / Music / Game
3. **Rating vs. Confidence Scatter** — item quality vs. model confidence
4. **Genre Spread Bar** — distribution of genres in the result set
5. **Release Year Histogram** — temporal spread of recommendations

---

## 6. Dataset

The dataset contains 37 manually curated items chosen to represent diverse quality tiers, moods, and genres within each domain. All items have real-world ratings (IMDb, Goodreads equivalents), normalized for model input. The dataset is structured as a Python list of dictionaries and loaded via `pandas.DataFrame`, making it trivially extensible.

---

## 7. Key Design Decisions

**Why TF-IDF over embeddings?**  
TF-IDF provides interpretable, fast, and dependency-free text similarity for a dataset of this scale. Embedding models (sentence-transformers) would improve semantic matching but add significant dependency weight and inference time — overkill for 37 items.

**Why session-scoped state?**  
Persistent storage (SQLite, JSON files) would complicate the setup for an internship submission. Session-scoped state in `SessionManager` provides all the behavioral benefits (history, favorites, feedback) without deployment complexity.

**Why MMR over simple top-k?**  
A naive top-k by hybrid score would often return 5 Sci-Fi movies when the user selected multiple domains. MMR ensures the output set is diverse enough to be genuinely useful.

---

## 8. Results

Sample run with preferences `Sci-Fi, Intense, Movie, keywords=space`:

| Rank | Title | Domain | Confidence |
|---|---|---|---|
| 1 | Inception | Movie | 70.9% |
| 2 | The Matrix | Movie | 70.7% |
| 3 | Interstellar | Movie | 61.0% |
| 4 | 1984 | Book | 57.5% |
| 5 | Cyberpunk 2077 | Game | 61.3% |

MMR correctly diversifies results by introducing a Book and a Game despite the Movie-heavy input preference.

---

## 9. Conclusion

RecoAI demonstrates a complete, production-aware recommendation pipeline implemented from scratch using standard Python data science libraries. The system covers content-based filtering, hybrid scoring, diversity optimization, cold-start handling, feedback suppression, and professional user experience — representing industry-standard techniques in a clean, extensible architecture.

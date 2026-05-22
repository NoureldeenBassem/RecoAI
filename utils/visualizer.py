"""
Visualization — recommendation analytics dashboard saved as PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

PALETTE = {
    "Movie": "#4C72B0",
    "Book":  "#DD8452",
    "Music": "#55A868",
    "Game":  "#C44E52",
}
BG = "#0F1117"
CARD = "#1A1D27"
TEXT = "#E8EAF6"


def plot_dashboard(recs: pd.DataFrame, preferences: dict, save_path: str = "output/dashboard.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("  AI Recommendation Dashboard", fontsize=18, fontweight="bold",
                 color=TEXT, y=0.97)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Confidence bar chart ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(CARD)
    colors = [PALETTE.get(d, "#888") for d in recs["domain"]]
    bars = ax1.barh(recs["title"][::-1], recs["confidence_pct"][::-1],
                    color=colors[::-1], edgecolor="none", height=0.6)
    for bar, pct in zip(bars, recs["confidence_pct"][::-1]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", color=TEXT, fontsize=8)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Confidence Score (%)", color=TEXT, fontsize=9)
    ax1.set_title("Top Recommendations by Confidence", color=TEXT, fontsize=11, pad=8)
    ax1.tick_params(colors=TEXT, labelsize=8)
    ax1.spines[:].set_visible(False)
    ax1.xaxis.label.set_color(TEXT)
    patches = [mpatches.Patch(color=c, label=d) for d, c in PALETTE.items()]
    ax1.legend(handles=patches, loc="lower right", fontsize=7,
               facecolor=BG, edgecolor="none", labelcolor=TEXT)

    # ── 2. Domain distribution pie ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD)
    domain_counts = recs["domain"].value_counts()
    wedges, texts, autotexts = ax2.pie(
        domain_counts.values,
        labels=domain_counts.index,
        colors=[PALETTE.get(d, "#888") for d in domain_counts.index],
        autopct="%1.0f%%", startangle=140,
        textprops={"color": TEXT, "fontsize": 8},
        wedgeprops={"linewidth": 0.5, "edgecolor": BG},
    )
    for a in autotexts:
        a.set_color(BG)
        a.set_fontsize(8)
    ax2.set_title("Domain Mix", color=TEXT, fontsize=11, pad=8)

    # ── 3. Rating vs confidence scatter ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(CARD)
    sc = ax3.scatter(
        recs["rating"], recs["confidence_pct"],
        c=[list(PALETTE.keys()).index(d) if d in PALETTE else 0 for d in recs["domain"]],
        cmap="tab10", s=80, alpha=0.85, edgecolors="none",
    )
    for _, row in recs.iterrows():
        ax3.annotate(row["title"][:12], (row["rating"], row["confidence_pct"]),
                     fontsize=6, color=TEXT, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax3.set_xlabel("Item Rating", color=TEXT, fontsize=9)
    ax3.set_ylabel("Confidence (%)", color=TEXT, fontsize=9)
    ax3.set_title("Rating vs Confidence", color=TEXT, fontsize=11, pad=8)
    ax3.tick_params(colors=TEXT, labelsize=8)
    ax3.spines[:].set_color("#333")

    # ── 4. Genre distribution bar ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CARD)
    genre_counts = recs["genre"].value_counts().head(6)
    ax4.bar(genre_counts.index, genre_counts.values,
            color="#4C72B0", edgecolor="none", width=0.6)
    ax4.set_title("Genre Spread", color=TEXT, fontsize=11, pad=8)
    ax4.tick_params(colors=TEXT, labelsize=7, axis="x", rotation=20)
    ax4.tick_params(colors=TEXT, labelsize=8, axis="y")
    ax4.spines[:].set_visible(False)

    # ── 5. Year distribution ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(CARD)
    ax5.hist(recs["year"], bins=8, color="#55A868", edgecolor=BG, linewidth=0.5)
    ax5.set_title("Release Year Spread", color=TEXT, fontsize=11, pad=8)
    ax5.set_xlabel("Year", color=TEXT, fontsize=9)
    ax5.tick_params(colors=TEXT, labelsize=8)
    ax5.spines[:].set_visible(False)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor("#222")

    fig.patch.set_facecolor(BG)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return save_path
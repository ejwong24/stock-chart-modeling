"""Generate visuals for the chapter 9 DINOv2 deep-dive section.

Extracts BNTX's actual 384-dim DINOv2 embedding, runs the same per-fold
PCA-64 the pipeline does, and visualizes:
1. The raw 384-dim vector as a heatmap
2. Cumulative variance explained vs. # PCs (showing how much PCA64 captures)
3. The 64-dim PCA projection of BNTX
4. A pipeline decision-tree showing what gets kept vs dropped
5. The final LR coefficients on the 128-dim concatenated feature
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
OUT = PROJECT / "web" / "static" / "story"
OUT.mkdir(parents=True, exist_ok=True)

P = {
    "bg": "#0e1118", "panel": "#161b25", "panel_2": "#1f2632",
    "border": "#2b3344", "text": "#e8edf5", "text_dim": "#9aa6b8",
    "accent": "#6ea8ff", "accent_2": "#4ee0a8", "warn": "#ffb454",
    "bad": "#ff7a85", "good": "#4ee0a8",
}

plt.rcParams.update({
    "figure.facecolor": P["panel"], "axes.facecolor": P["panel"],
    "axes.edgecolor": P["border"], "axes.labelcolor": P["text"],
    "axes.titlecolor": P["text"],
    "xtick.color": P["text_dim"], "ytick.color": P["text_dim"],
    "text.color": P["text"], "grid.color": P["border"], "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size": 11, "savefig.facecolor": P["panel"],
    "savefig.dpi": 130,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.parse_math": False,
})

# Load real embeddings + locate BNTX 2021-07-02
embs_all = np.load(PROJECT / "reports" / "full" / "dinov2_embeddings.npy")
anchors = pd.read_parquet(PROJECT / "reports" / "full" / "anchor_kept.parquet")
mask = (anchors["ticker"] == "BNTX") & (anchors["anchor_date"] == pd.Timestamp("2021-07-02"))
BNTX_IDX = int(anchors[mask].index[0])
bntx_raw = embs_all[BNTX_IDX]

# Simulate the per-fold PCA the pipeline does. Use anchors with year < 2021
# as the "training fold" for the 2021-07-02 example.
anchors["year"] = pd.to_datetime(anchors["anchor_date"]).dt.year
train_mask = anchors["year"] < 2021
train_embs = embs_all[train_mask.to_numpy()]
print(f"train fold n={len(train_embs)}, total n={len(embs_all)}")

# Fit StandardScaler + PCA64 on the training fold (what models.py does)
scaler = StandardScaler().fit(train_embs)
train_std = scaler.transform(train_embs)
pca = PCA(n_components=64, random_state=42).fit(train_std)
print(f"PCA64 fitted; total variance explained = {pca.explained_variance_ratio_.sum():.4f}")

# Project BNTX through the same scaler + PCA
bntx_std = scaler.transform(bntx_raw.reshape(1, -1))
bntx_pca = pca.transform(bntx_std)[0]
print(f"BNTX projected: 384 → 64 dims, range [{bntx_pca.min():.3f}, {bntx_pca.max():.3f}]")


# ─── Viz 1: the raw 384-dim BNTX embedding ─────────────────────────────

def viz_1_raw_vector():
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    # 24 × 16 = 384
    grid = bntx_raw.reshape(16, 24)
    im = ax.imshow(grid, cmap="RdBu_r", aspect="auto", vmin=-6, vmax=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("The actual 384-dim DINOv2 output for BNTX on 2021-07-02\n"
                 "(reshaped as 16×24 for visualization — each cell is one of the 384 numbers)",
                 loc="left", fontsize=12, fontweight="bold", pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("activation value", color=P["text_dim"])

    # Annotation: stats
    ax.text(0.01, -0.18,
            f"mean = {bntx_raw.mean():+.3f}     std = {bntx_raw.std():.3f}     "
            f"min = {bntx_raw.min():+.2f}     max = {bntx_raw.max():+.2f}",
            transform=ax.transAxes, color=P["text_dim"], fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "pw_d1_dinov2_raw.png")
    plt.close()
    print("wrote pw_d1_dinov2_raw.png")


# ─── Viz 2: cumulative variance explained ──────────────────────────────

def viz_2_variance_explained():
    full_pca = PCA().fit(train_std)
    cumvar = np.cumsum(full_pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(1, len(cumvar) + 1)
    ax.plot(x, cumvar * 100, color=P["accent"], lw=2.2)
    ax.fill_between(x, 0, cumvar * 100, color=P["accent"], alpha=0.15)

    pca64_var = cumvar[63] * 100
    ax.axvline(64, color=P["warn"], lw=2, linestyle="--",
                label=f"PCA-64 cutoff (used by pipeline)\nretains {pca64_var:.1f}% of total variance")
    ax.axhline(pca64_var, color=P["warn"], lw=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("How much variance does PCA-64 capture in the 384-dim DINOv2 embeddings?",
                 loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(facecolor=P["panel_2"], edgecolor=P["border"], loc="lower right",
               fontsize=10.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 384)
    ax.set_ylim(0, 102)

    ax.annotate(
        f"PCA64 keeps {pca64_var:.1f}% of variance\n"
        f"throws away the other {100-pca64_var:.1f}% of variance — but\n"
        f"with no guarantee that variance == predictive signal",
        xy=(64, pca64_var), xytext=(170, 55),
        fontsize=10, color=P["text"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor=P["panel_2"],
                   edgecolor=P["warn"], alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=P["warn"], lw=1.5))

    plt.tight_layout()
    plt.savefig(OUT / "pw_d2_pca_variance.png")
    plt.close()
    print(f"wrote pw_d2_pca_variance.png ({pca64_var:.1f}% retained)")
    return pca64_var


# ─── Viz 3: BNTX projected to 64 PCA dims ──────────────────────────────

def viz_3_bntx_projected():
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    x = np.arange(len(bntx_pca))
    colors = [P["good"] if v >= 0 else P["bad"] for v in bntx_pca]
    ax.bar(x, bntx_pca, color=colors, edgecolor=P["border"], alpha=0.85)
    ax.axhline(0, color=P["text_dim"], lw=1)
    ax.set_xlabel("Principal component (in decreasing order of variance explained)")
    ax.set_ylabel("BNTX projection value")
    ax.set_title("Stage 4b — BNTX's 384 DINOv2 numbers compressed to 64 PCA features",
                 loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, 63.5)

    plt.tight_layout()
    plt.savefig(OUT / "pw_d3_bntx_pca64.png")
    plt.close()
    print("wrote pw_d3_bntx_pca64.png")


# ─── Viz 4: pipeline decision tree ─────────────────────────────────────

def viz_4_decision_tree():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, label, color, font_size=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor=P["border"],
                                      alpha=0.92, lw=1.3))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color=P["text"], fontsize=font_size,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=P["text_dim"]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

    # Input
    box(4.5, 7.0, 3, 0.6, "252-day price + volume window", P["panel_2"], bold=True)

    # Two parallel branches
    # Image branch
    box(0.5, 5.7, 3, 0.6, "Render 224×224 chart image", P["accent"])
    box(0.5, 4.5, 3, 0.6, "Frozen DINOv2 ViT-S/14\n→ 384-dim embedding", P["accent"])
    box(0.5, 3.3, 3, 0.6, "StandardScaler + PCA-64\n→ 64 image features", P["accent"])

    # Volume branch (middle)
    box(4.5, 5.7, 3, 0.6, "zscore(log1p(volume))\n→ 252-dim vector", P["accent_2"])
    box(4.5, 4.5, 3, 0.6, "StandardScaler + PCA-64\n→ 64 volume features", P["accent_2"])
    box(4.5, 3.3, 3, 0.6, "(passes through)", P["panel_2"])

    # Engineered branch (right)
    box(8.5, 5.7, 3, 0.6, "Compute 40 engineered features\n(slope, vol, dd, MA, …)", P["good"])
    box(8.5, 4.5, 3, 0.6, "(used as-is)", P["panel_2"])
    box(8.5, 3.3, 3, 0.6, "(used as-is)", P["panel_2"])

    # Arrows from input down
    arrow(6, 7.0, 2, 6.3)
    arrow(6, 7.0, 6, 6.3)
    arrow(6, 7.0, 10, 6.3)

    # Arrows through each branch
    for x in [2, 6, 10]:
        arrow(x, 5.7, x, 5.1)
        arrow(x, 4.5, x, 3.9)

    # Two model heads
    box(2, 2.0, 4, 0.6, "Concat (64+64=128 features)\n→ LR  /  LGBM-image", P["accent"], bold=True)
    box(7.5, 2.0, 4, 0.6, "40 engineered features\n→ LGBM-engineered", P["good"], bold=True)

    arrow(2, 3.3, 2, 2.6)
    arrow(6, 3.3, 4, 2.6)
    arrow(10, 3.3, 9.5, 2.6)

    # Outputs
    box(2, 0.5, 4, 0.6, "probability score (loses to engineered)", P["bad"])
    box(7.5, 0.5, 4, 0.6, "probability score (wins post-cost)", P["good"], bold=True)

    arrow(4, 2.0, 4, 1.1)
    arrow(9.5, 2.0, 9.5, 1.1)

    # Branch labels
    ax.text(2, 6.6, "IMAGE TRACK", ha="center", color=P["accent"],
             fontsize=11, fontweight="bold")
    ax.text(6, 6.6, "VOLUME TRACK", ha="center", color=P["accent_2"],
             fontsize=11, fontweight="bold")
    ax.text(10, 6.6, "ENGINEERED TRACK", ha="center", color=P["good"],
             fontsize=11, fontweight="bold")

    ax.set_title("The three parallel tracks — what the pipeline does with each",
                  loc="left", fontsize=13.5, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(OUT / "pw_d4_decision_tree.png")
    plt.close()
    print("wrote pw_d4_decision_tree.png")


# ─── Viz 5: number-reduction funnel ────────────────────────────────────

def viz_5_information_funnel():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    stages = [
        ("224×224×3 chart image",  150_528, P["text_dim"]),
        ("DINOv2 embedding",       384,     P["accent"]),
        ("PCA-64 image features",  64,      P["warn"]),
        ("After concat with volume PCA-64", 128, P["warn"]),
        ("Final LR probability",   1,       P["bad"]),
    ]
    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    y = np.arange(len(stages))
    ax.barh(y, np.log10(counts), color=colors, edgecolor=P["border"], alpha=0.9)
    for i, (lbl, n) in enumerate(zip(labels, counts)):
        ax.text(np.log10(n) + 0.05, i, f"  = {n:,}", va="center",
                 color=P["text"], fontweight="bold", fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("log10(numbers)")
    ax.set_title("Information funnel — how many numbers represent BNTX at each stage",
                  loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 6.5)
    ax.grid(True, axis="x", alpha=0.3)

    # Reduction percentages
    for i in range(1, len(stages)):
        reduction = (1 - stages[i][1] / stages[i-1][1]) * 100
        ax.text(np.log10(counts[i]) + 1.4, i - 0.5, f"−{reduction:.1f}% information",
                 color=P["text_dim"], fontsize=9, fontstyle="italic")

    plt.tight_layout()
    plt.savefig(OUT / "pw_d5_info_funnel.png")
    plt.close()
    print("wrote pw_d5_info_funnel.png")


def main():
    viz_1_raw_vector()
    viz_2_variance_explained()
    viz_3_bntx_projected()
    viz_4_decision_tree()
    viz_5_information_funnel()
    print(f"\nDINOv2 deep-dive visuals in {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)

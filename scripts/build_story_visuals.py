"""Generate PNG visuals for the /story walkthrough.

Reads from existing reports/ artifacts (the no-cost, realistic-cost, and
SMA250 runs) and produces a dozen-or-so chart files in
web/static/story/ that the story templates embed.

Theme: dark, matches the UI's CSS color palette
  bg:        #0e1118
  panel:     #161b25
  panel-2:   #1f2632
  border:    #2b3344
  text:      #e8edf5
  text-dim:  #9aa6b8
  accent:    #6ea8ff
  accent-2:  #4ee0a8
  warn:      #ffb454
  bad:       #ff7a85
  good:      #4ee0a8
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "web" / "static" / "story"
OUT.mkdir(parents=True, exist_ok=True)

# Theme
PALETTE = {
    "bg": "#0e1118",
    "panel": "#161b25",
    "panel_2": "#1f2632",
    "border": "#2b3344",
    "text": "#e8edf5",
    "text_dim": "#9aa6b8",
    "accent": "#6ea8ff",
    "accent_2": "#4ee0a8",
    "warn": "#ffb454",
    "bad": "#ff7a85",
    "good": "#4ee0a8",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["panel"],
    "axes.facecolor": PALETTE["panel"],
    "axes.edgecolor": PALETTE["border"],
    "axes.labelcolor": PALETTE["text"],
    "axes.titlecolor": PALETTE["text"],
    "xtick.color": PALETTE["text_dim"],
    "ytick.color": PALETTE["text_dim"],
    "text.color": PALETTE["text"],
    "grid.color": PALETTE["border"],
    "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "savefig.facecolor": PALETTE["panel"],
    "savefig.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TRACK_COLORS = {
    "lgbm_image": PALETTE["accent"],
    "lgbm_engineered": PALETTE["good"],
    "lr_baseline": "#ff8eda",
    "rank_252d_return": PALETTE["warn"],
    "rank_60d_return": "#ffd070",
    "rank_ma250_extension": "#ffe49a",
    "rank_52w_high_distance": "#ff7a85",
    "rank_inv_60d_vol": "#c89aff",
}


def fmt_dollar(x, _=None):
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    if x >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:.0f}"


# --- Chart 1: the claim gap ---------------------------------------------

def chart_claim_gap():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    labels = ["Original\nclaim", "Our honest\nreproduction\n(same method)",
              "Best of 3 ML\ntracks, no costs", "Best track\nrealistic costs",
              "Random\nmedian"]
    values = [3_234_598, 298_262, 496_558, 256_050, 294_804]
    colors = [PALETTE["bad"], "#ff8eda", PALETTE["accent"],
              PALETTE["good"], PALETTE["text_dim"]]

    bars = ax.bar(labels, values, color=colors, edgecolor=PALETTE["border"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.04, fmt_dollar(v),
                ha="center", va="bottom", color=PALETTE["text"], fontsize=11,
                fontweight="bold")

    ax.set_ylabel("Ending equity from $100k (7.3 years)")
    ax.set_title("The claim collapses under honest reproduction",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.set_yscale("log")
    ax.set_ylim(50_000, 8_000_000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_dollar))
    ax.tick_params(axis="x", labelsize=9.5)

    ax.axhline(100_000, color=PALETTE["text_dim"], linestyle=":", alpha=0.5,
               label="Starting capital ($100k)")
    ax.legend(loc="upper right", facecolor=PALETTE["panel_2"],
              edgecolor=PALETTE["border"])

    plt.tight_layout()
    plt.savefig(OUT / "01_claim_gap.png")
    plt.close()
    print("wrote 01_claim_gap.png")


# --- Chart 2: equity curves overlay (no-cost) ---------------------------

def chart_equity_curves(run_tag: str, fname: str, title_suffix: str):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    runs_dir = PROJECT / "reports" / run_tag
    tracks = sorted([p.stem.replace("equity_", "")
                      for p in runs_dir.glob("equity_*.parquet")])

    # Sort by final equity descending
    by_final = []
    for t in tracks:
        df = pd.read_parquet(runs_dir / f"equity_{t}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        by_final.append((t, df, df["equity"].iloc[-1]))
    by_final.sort(key=lambda r: -r[2])

    for t, df, final in by_final:
        color = TRACK_COLORS.get(t, PALETTE["text_dim"])
        lw = 2.0 if not t.startswith("rank_") else 1.4
        alpha = 1.0 if not t.startswith("rank_") else 0.7
        ax.plot(df["date"], df["equity"], color=color, lw=lw, alpha=alpha,
                label=f"{t}  ({fmt_dollar(final)})")

    ax.set_yscale("log")
    ax.set_ylabel("Account equity")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_dollar))
    ax.axhline(100_000, color=PALETTE["text_dim"], linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", facecolor=PALETTE["panel_2"],
              edgecolor=PALETTE["border"], fontsize=9,
              labelcolor="#dddddd")
    ax.set_title(f"All 8 strategies over time — {title_suffix}",
                 loc="left", fontsize=14, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / fname)
    plt.close()
    print(f"wrote {fname}")


# --- Chart 3: no-cost vs realistic-cost side-by-side --------------------

def chart_cost_impact():
    old_dir = PROJECT / "reports" / "full"
    new_dir = PROJECT / "reports" / "full_realistic"
    old = json.loads((old_dir / "headline.json").read_text())
    new = json.loads((new_dir / "headline.json").read_text())
    old_s = {s["track"]: s for s in old["summaries"]}
    new_s = {s["track"]: s for s in new["summaries"]}

    tracks = sorted(old_s.keys(), key=lambda t: -old_s[t]["cagr"])
    x = np.arange(len(tracks))
    w = 0.38

    fig, (ax_cagr, ax_dd) = plt.subplots(1, 2, figsize=(13, 5.5))

    old_cagr = [old_s[t]["cagr"] * 100 for t in tracks]
    new_cagr = [new_s[t]["cagr"] * 100 for t in tracks]
    ax_cagr.bar(x - w / 2, old_cagr, w, color=PALETTE["accent"],
                 label="No costs", edgecolor=PALETTE["border"])
    ax_cagr.bar(x + w / 2, new_cagr, w, color=PALETTE["warn"],
                 label="Realistic costs", edgecolor=PALETTE["border"])
    ax_cagr.set_xticks(x)
    ax_cagr.set_xticklabels(tracks, rotation=35, ha="right", fontsize=9)
    ax_cagr.set_ylabel("CAGR (%)")
    ax_cagr.axhline(0, color=PALETTE["text_dim"], linestyle="-", alpha=0.6)
    ax_cagr.set_title("CAGR per strategy: cost-aware reality",
                       loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_cagr.legend(facecolor=PALETTE["panel_2"], edgecolor=PALETTE["border"])
    ax_cagr.grid(True, axis="y", alpha=0.3)

    old_dd = [old_s[t]["max_dd"] * 100 for t in tracks]
    new_dd = [new_s[t]["max_dd"] * 100 for t in tracks]
    ax_dd.bar(x - w / 2, old_dd, w, color=PALETTE["bad"],
              label="No costs (no stop)", edgecolor=PALETTE["border"])
    ax_dd.bar(x + w / 2, new_dd, w, color=PALETTE["accent_2"],
              label="Realistic + trailing stop", edgecolor=PALETTE["border"])
    ax_dd.set_xticks(x)
    ax_dd.set_xticklabels(tracks, rotation=35, ha="right", fontsize=9)
    ax_dd.set_ylabel("Max drawdown (%)")
    ax_dd.set_title("Max drawdown: 18% trailing stop helps",
                     loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_dd.legend(facecolor=PALETTE["panel_2"], edgecolor=PALETTE["border"])
    ax_dd.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "03_cost_impact.png")
    plt.close()
    print("wrote 03_cost_impact.png")


# --- Chart 4: deflated Sharpe ------------------------------------------

def chart_deflated_sharpe():
    n_trials = 108
    observed_sr = 0.510
    threshold = 0.747

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sharpes = np.linspace(-0.5, 2.0, 500)
    null_pdf = np.exp(-((sharpes - 0) ** 2) / 0.4)
    deflated_pdf = np.exp(-((sharpes - threshold) ** 2) / 0.18)

    ax.fill_between(sharpes, 0, null_pdf, color=PALETTE["text_dim"], alpha=0.4,
                     label="What you'd see by luck (zero true skill, 1 trial)")
    ax.fill_between(sharpes, 0, deflated_pdf * 0.6, color=PALETTE["bad"],
                     alpha=0.4,
                     label=f"What you'd see by luck (zero true skill, {n_trials} trials, take best)")

    ax.axvline(observed_sr, color=PALETTE["accent"], lw=2.5,
                label=f"Observed Sharpe = {observed_sr:.2f}")
    ax.axvline(threshold, color=PALETTE["warn"], lw=2.5, linestyle="--",
                label=f"Deflated significance threshold = {threshold:.2f}")

    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Likelihood (sketch)")
    ax.set_title("Why the result isn't statistically real",
                 loc="left", fontsize=14, fontweight="bold", pad=10)
    ax.set_yticks([])
    ax.legend(facecolor=PALETTE["panel_2"], edgecolor=PALETTE["border"],
              loc="upper left", fontsize=9.5)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.5, 2.0)
    ax.grid(True, axis="x", alpha=0.3)

    ax.annotate(
        "Observed Sharpe is BELOW the threshold\nthat the best-of-108-trials random luck reaches\n→ result NOT distinguishable from chance",
        xy=(observed_sr, 0.7), xytext=(1.05, 0.6),
        fontsize=10, color=PALETTE["text"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["panel_2"],
                   edgecolor=PALETTE["border"], alpha=0.95))

    plt.tight_layout()
    plt.savefig(OUT / "04_deflated_sharpe.png")
    plt.close()
    print("wrote 04_deflated_sharpe.png")


# --- Chart 5: random distribution vs model -----------------------------

def chart_random_distribution(run_tag: str, fname: str, title_suffix: str):
    runs_dir = PROJECT / "reports" / run_tag
    rnd = pd.read_parquet(runs_dir / "random_seeds_summary.parquet")
    head = json.loads((runs_dir / "headline.json").read_text())
    summaries = head["summaries"]
    best = max(summaries, key=lambda s: s["end_equity"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(rnd["end_equity"], bins=30, color=PALETTE["text_dim"],
             edgecolor=PALETTE["border"], alpha=0.85,
             label="200 random portfolios, same rules")

    p50 = rnd["end_equity"].median()
    p95 = rnd["end_equity"].quantile(0.95)
    ax.axvline(p50, color=PALETTE["warn"], lw=2,
                label=f"Random median: {fmt_dollar(p50)}")
    ax.axvline(p95, color=PALETTE["bad"], lw=2, linestyle="--",
                label=f"Random 95th pct: {fmt_dollar(p95)}")
    ax.axvline(best["end_equity"], color=PALETTE["accent_2"], lw=3,
                label=f"Best strategy ({best['track']}): {fmt_dollar(best['end_equity'])}")

    ax.set_xlabel("Ending equity from $100k")
    ax.set_ylabel("Number of random seeds")
    ax.set_title(f"Random distribution vs best model — {title_suffix}",
                 loc="left", fontsize=14, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel_2"], edgecolor=PALETTE["border"],
              loc="upper right", fontsize=9.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_dollar))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / fname)
    plt.close()
    print(f"wrote {fname}")


# --- Chart 6: SMA250 test boxplot --------------------------------------

def chart_sma250_test():
    d = PROJECT / "reports" / "sma250_test"
    s20 = pd.read_csv(d / "H20_per_seed_summary.csv")
    s40 = pd.read_csv(d / "H40_per_seed_summary.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))

    bp_eq = a1.boxplot(
        [s20["ending_equity"], s40["ending_equity"]],
        labels=["20-day hold", "40-day hold"],
        patch_artist=True, widths=0.55,
        medianprops={"color": PALETTE["accent_2"], "lw": 2.5},
        boxprops={"facecolor": PALETTE["panel_2"], "edgecolor": PALETTE["border"]},
        whiskerprops={"color": PALETTE["text_dim"]},
        capprops={"color": PALETTE["text_dim"]},
        flierprops={"marker": "o", "markersize": 4,
                     "markerfacecolor": PALETTE["warn"],
                     "markeredgecolor": PALETTE["warn"]})
    for med_h, color in zip([s20["ending_equity"].median(),
                              s40["ending_equity"].median()],
                              [PALETTE["accent"], PALETTE["good"]]):
        pass
    a1.yaxis.set_major_formatter(plt.FuncFormatter(fmt_dollar))
    a1.set_ylabel("Ending equity from $100k")
    a1.set_title("Random portfolio dispersion (200 seeds)",
                  loc="left", fontsize=13, fontweight="bold", pad=8)
    a1.set_yscale("log")
    a1.axhline(100_000, color=PALETTE["text_dim"], linestyle=":", alpha=0.5)
    a1.grid(True, axis="y", alpha=0.3)
    a1.text(1, s20["ending_equity"].median() * 1.15,
             f"median\n{fmt_dollar(s20['ending_equity'].median())}",
             ha="center", color=PALETTE["accent_2"], fontweight="bold")
    a1.text(2, s40["ending_equity"].median() * 1.15,
             f"median\n{fmt_dollar(s40['ending_equity'].median())}",
             ha="center", color=PALETTE["accent_2"], fontweight="bold")

    bp_cagr = a2.boxplot(
        [s20["cagr"] * 100, s40["cagr"] * 100],
        labels=["20-day hold", "40-day hold"],
        patch_artist=True, widths=0.55,
        medianprops={"color": PALETTE["accent_2"], "lw": 2.5},
        boxprops={"facecolor": PALETTE["panel_2"], "edgecolor": PALETTE["border"]},
        whiskerprops={"color": PALETTE["text_dim"]},
        capprops={"color": PALETTE["text_dim"]},
        flierprops={"marker": "o", "markersize": 4,
                     "markerfacecolor": PALETTE["warn"],
                     "markeredgecolor": PALETTE["warn"]})
    a2.set_ylabel("CAGR (%)")
    a2.set_title("Random portfolio CAGR dispersion",
                  loc="left", fontsize=13, fontweight="bold", pad=8)
    a2.axhline(0, color=PALETTE["text_dim"], linestyle="-", alpha=0.5)
    a2.grid(True, axis="y", alpha=0.3)
    a2.text(1, s20["cagr"].median() * 100 + 1.5,
             f"+{s20['cagr'].median() * 100:.2f}%",
             ha="center", color=PALETTE["accent_2"], fontweight="bold")
    a2.text(2, s40["cagr"].median() * 100 + 1.5,
             f"+{s40['cagr'].median() * 100:.2f}%",
             ha="center", color=PALETTE["accent_2"], fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "06_sma250_dispersion.png")
    plt.close()
    print("wrote 06_sma250_dispersion.png")


# --- Chart 7: Effective N visualization --------------------------------

def chart_effective_n():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # 25 concurrent positions × overlapping → ~45 independent trades
    naive_n = 1104
    eff_n = 45

    ax.barh(["Effective N\n(López de Prado avg uniqueness)",
              "Naive count\n(closed trades)"],
             [eff_n, naive_n],
             color=[PALETTE["good"], PALETTE["bad"]],
             edgecolor=PALETTE["border"])

    for i, v in enumerate([eff_n, naive_n]):
        ax.text(v * 1.02, i, f"{v:,}", va="center",
                 color=PALETTE["text"], fontweight="bold", fontsize=12)

    widen = (naive_n / eff_n) ** 0.5
    ax.set_xlabel("Number of trades")
    ax.set_title(f"Trade count is misleading: 1,104 trades, but only ~45 independent observations\n"
                 f"→ SE(Sharpe) is {widen:.1f}× wider than the naive count suggests",
                  loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, naive_n * 1.18)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "07_effective_n.png")
    plt.close()
    print("wrote 07_effective_n.png")


# --- Diagram: walk-forward leakage ------------------------------------

def diagram_walk_forward():
    fig, ax = plt.subplots(figsize=(11, 4.4))

    years = list(range(2017, 2026))
    test_year = 2022
    horizon_days = 40

    # Original (broken) walk-forward
    for y in years:
        x = y - 2017
        is_test = (y == test_year)
        color = PALETTE["warn"] if is_test else PALETTE["panel_2"]
        edge = PALETTE["accent"] if is_test else PALETTE["border"]
        ax.add_patch(Rectangle((x, 1.4), 1, 0.6, facecolor=color,
                                edgecolor=edge, lw=1.5))
        ax.text(x + 0.5, 1.7, str(y), ha="center", va="center",
                color=PALETTE["text"], fontweight="bold")

    # Show the leak: late-Dec 2021 training row with label resolving in Feb 2022
    ax.add_patch(Rectangle((4.75, 1.4), 0.25, 0.6, facecolor=PALETTE["bad"],
                            alpha=0.6, edgecolor="none"))
    ax.annotate(
        "Late-Dec 2021 training row\nwith 40-day forward label that\nresolves IN the 2022 test fold\n→ leakage",
        xy=(4.95, 1.7), xytext=(2.5, 0.3),
        fontsize=9, color=PALETTE["bad"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["bad"], lw=1.5))

    ax.text(-0.05, 1.7, "Walk-forward", ha="right", va="center",
            color=PALETTE["text_dim"], fontweight="bold")

    # Corrected walk-forward with embargo
    for y in years:
        x = y - 2017
        is_test = (y == test_year)
        color = PALETTE["warn"] if is_test else PALETTE["panel_2"]
        edge = PALETTE["accent"] if is_test else PALETTE["border"]
        ax.add_patch(Rectangle((x, 2.6), 1, 0.6, facecolor=color,
                                edgecolor=edge, lw=1.5))
        ax.text(x + 0.5, 2.9, str(y), ha="center", va="center",
                color=PALETTE["text"], fontweight="bold")

    # Show purge zone (dropped rows before test fold)
    ax.add_patch(Rectangle((4.75, 2.6), 0.25, 0.6, facecolor=PALETTE["panel"],
                            edgecolor=PALETTE["text_dim"], lw=1, alpha=0.4,
                            hatch="////"))
    ax.annotate(
        "Embargo zone:\nthe last ~8 weeks of\ntraining data are dropped\nso no label leaks",
        xy=(4.85, 2.9), xytext=(2.5, 3.9),
        fontsize=9, color=PALETTE["good"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["good"], lw=1.5))

    ax.text(-0.05, 2.9, "Purged walk-forward\n(López de Prado)",
            ha="right", va="center",
            color=PALETTE["text_dim"], fontweight="bold")

    ax.set_xlim(-2.5, 9.5)
    ax.set_ylim(-0.4, 4.6)
    ax.axis("off")

    legend_x = 8.5
    ax.add_patch(Rectangle((legend_x, 4.0), 0.4, 0.3,
                            facecolor=PALETTE["panel_2"],
                            edgecolor=PALETTE["border"]))
    ax.text(legend_x + 0.55, 4.15, "training year",
             color=PALETTE["text"], fontsize=9, va="center")
    ax.add_patch(Rectangle((legend_x, 3.55), 0.4, 0.3,
                            facecolor=PALETTE["warn"],
                            edgecolor=PALETTE["accent"]))
    ax.text(legend_x + 0.55, 3.7, "test year",
             color=PALETTE["text"], fontsize=9, va="center")
    ax.add_patch(Rectangle((legend_x, 3.1), 0.4, 0.3,
                            facecolor=PALETTE["bad"], alpha=0.6))
    ax.text(legend_x + 0.55, 3.25, "leak zone (bad)",
             color=PALETTE["text"], fontsize=9, va="center")
    ax.add_patch(Rectangle((legend_x, 2.65), 0.4, 0.3,
                            facecolor=PALETTE["panel"],
                            edgecolor=PALETTE["text_dim"], lw=1,
                            hatch="////"))
    ax.text(legend_x + 0.55, 2.8, "embargo (purged)",
             color=PALETTE["text"], fontsize=9, va="center")

    ax.set_title("Walk-forward with leakage vs. purged walk-forward with embargo",
                  loc="left", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(OUT / "08_walk_forward_diagram.png")
    plt.close()
    print("wrote 08_walk_forward_diagram.png")


# --- Diagram: multiple comparison fan ----------------------------------

def diagram_multiple_comparison():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    rng = np.random.default_rng(0)

    # Draw 72 "no-skill" trial outcomes
    n_trials = 72
    null_cagrs = rng.normal(0.11, 0.06, n_trials)  # centered on random median
    null_cagrs.sort()

    x = np.arange(n_trials)
    bars = ax.bar(x, null_cagrs * 100, color=PALETTE["text_dim"], alpha=0.5,
                   edgecolor="none", width=0.85)
    # Highlight the BEST
    best_idx = np.argmax(null_cagrs)
    bars[best_idx].set_color(PALETTE["bad"])
    bars[best_idx].set_alpha(1.0)
    ax.annotate(
        "Original paper reported\nthis one as 'the model'",
        xy=(best_idx, null_cagrs[best_idx] * 100),
        xytext=(best_idx - 22, 24),
        fontsize=10, color=PALETTE["bad"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["bad"], lw=1.5))

    ax.axhline(null_cagrs.mean() * 100, color=PALETTE["text_dim"],
                linestyle="--", lw=1.5,
                label=f"Mean of all 72 = {null_cagrs.mean() * 100:.1f}% (random-equivalent)")
    ax.set_xlabel("Trial index (sorted by CAGR)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Why 'best of 72 configs' isn't impressive — even with zero true skill",
                 loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel_2"], edgecolor=PALETTE["border"],
               loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax.text(36, -8, "If you flip a coin enough times you'll get 7 heads in a row.\n"
            "Reporting only the lucky streak is statistically meaningless.",
             ha="center", color=PALETTE["text"], fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["panel_2"],
                        edgecolor=PALETTE["border"]))

    ax.set_ylim(-12, 30)
    plt.tight_layout()
    plt.savefig(OUT / "09_multiple_comparison.png")
    plt.close()
    print("wrote 09_multiple_comparison.png")


# --- Chart 10: confidence intervals ------------------------------------

def chart_confidence_intervals():
    """Show how wide the CIs are for our best track."""
    # Hand-set: from honest_report_card.md auto-fill on reports/full
    # Best of full = lgbm_image, CAGR 17.27%, CI [-11.0%, +59.3%]
    fig, ax = plt.subplots(figsize=(10, 4.0))

    tracks = ["lgbm_image\n(no cost)", "lgbm_engineered\n(realistic cost)",
              "lr_baseline\n(matches original method)"]
    points = [17.27, 9.79, 5.12]
    ci_lo = [-11.0, -8.0, -14.0]
    ci_hi = [59.3, 27.0, 22.0]

    y = np.arange(len(tracks))
    err_lo = [p - lo for p, lo in zip(points, ci_lo)]
    err_hi = [hi - p for hi, p in zip(ci_hi, points)]
    ax.errorbar(points, y, xerr=[err_lo, err_hi], fmt="o", markersize=10,
                 color=PALETTE["accent_2"], capsize=8, capthick=2, lw=2,
                 elinewidth=2, ecolor=PALETTE["warn"])

    for i, (p, lo, hi) in enumerate(zip(points, ci_lo, ci_hi)):
        ax.text(p, i + 0.18, f"{p:+.2f}%", ha="center",
                 color=PALETTE["accent_2"], fontweight="bold")
        ax.text(lo, i - 0.2, f"{lo:+.1f}%", ha="center", color=PALETTE["warn"],
                 fontsize=9)
        ax.text(hi, i - 0.2, f"{hi:+.1f}%", ha="center", color=PALETTE["warn"],
                 fontsize=9)

    ax.axvline(0, color=PALETTE["bad"], linestyle="--", lw=1.5, alpha=0.7,
                label="Zero (no edge)")
    ax.set_yticks(y)
    ax.set_yticklabels(tracks, fontsize=10)
    ax.set_xlabel("CAGR with 95% block-bootstrap confidence interval")
    ax.set_title("Confidence intervals cross zero — the apparent edge is well within sampling noise",
                 loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel_2"], edgecolor=PALETTE["border"],
               loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(-25, 65)

    plt.tight_layout()
    plt.savefig(OUT / "10_confidence_intervals.png")
    plt.close()
    print("wrote 10_confidence_intervals.png")


# --- Chart 11: survivorship bias diagram ---------------------------------

def diagram_survivorship():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    rng = np.random.default_rng(7)
    n_stocks = 40
    years = np.arange(2016, 2027)

    # Each "stock" is a horizontal line. Some die mid-history.
    deaths = rng.integers(2018, 2030, n_stocks)
    deaths[rng.random(n_stocks) > 0.20] = 2027  # 20% die

    for i in range(n_stocks):
        death = deaths[i]
        if death >= 2027:
            ax.plot([2016, 2026], [i, i], color=PALETTE["accent"], lw=2,
                     alpha=0.85)
        else:
            ax.plot([2016, death], [i, i], color=PALETTE["bad"], lw=2,
                     alpha=0.65)
            ax.plot(death, i, "x", color=PALETTE["bad"], markersize=8, mew=2)

    ax.axvline(2026, color=PALETTE["text_dim"], linestyle="--", alpha=0.5)
    ax.text(2026.05, n_stocks + 1, "Snapshot date\n(2026-04-20)",
             color=PALETTE["text_dim"], fontsize=9)

    ax.text(2016.1, n_stocks + 1,
             "Your dataset contains:", fontsize=10, color=PALETTE["text"])

    ax.text(2027.5, n_stocks * 0.7, "ALIVE\n(in your dataset)",
             color=PALETTE["accent"], fontsize=11, fontweight="bold")
    ax.text(2027.5, n_stocks * 0.3, "DEAD\n(MISSING from\nyour dataset)",
             color=PALETTE["bad"], fontsize=11, fontweight="bold")

    ax.set_xlim(2015.5, 2030)
    ax.set_ylim(-2, n_stocks + 3)
    ax.set_xlabel("Year")
    ax.set_yticks([])
    ax.set_title("Survivorship bias: companies that went bankrupt are missing from the data",
                 loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT / "11_survivorship_diagram.png")
    plt.close()
    print("wrote 11_survivorship_diagram.png")


# --- Chart 12: final summary infographic --------------------------------

def chart_bottom_line():
    fig, ax = plt.subplots(figsize=(11, 5.2))

    scenarios = [
        ("Original claim", 61.1, PALETTE["bad"]),
        ("Same method, honest reproduction\n(no realistic costs)", 11.47,
         "#ff8eda"),
        ("Best of 3 ML tracks, no costs", 17.27, PALETTE["accent"]),
        ("Same method, realistic costs", 5.12, PALETTE["warn"]),
        ("Best simple non-AI strategy\n(low-vol momentum, realistic costs)",
         7.26, PALETTE["good"]),
        ("S&P 500 long-run real CAGR\n(for comparison)", 7.0,
         PALETTE["text_dim"]),
    ]
    labels = [s[0] for s in scenarios]
    cagrs = [s[1] for s in scenarios]
    colors = [s[2] for s in scenarios]

    y = np.arange(len(scenarios))
    bars = ax.barh(y, cagrs, color=colors, edgecolor=PALETTE["border"])
    for i, (bar, c) in enumerate(zip(bars, cagrs)):
        ax.text(c + 0.6, i, f"{c:+.2f}%", va="center",
                 color=PALETTE["text"], fontweight="bold", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("CAGR (% per year)")
    ax.set_title("The bottom line — what the strategy actually delivers",
                 loc="left", fontsize=13.5, fontweight="bold", pad=10)
    ax.set_xlim(0, 70)
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(7.0, color=PALETTE["text_dim"], linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT / "12_bottom_line.png")
    plt.close()
    print("wrote 12_bottom_line.png")


def main():
    chart_claim_gap()
    chart_equity_curves("full", "02_equity_no_cost.png",
                         "no realistic costs")
    chart_equity_curves("full_realistic", "02b_equity_realistic.png",
                         "realistic costs + 18% trailing stop")
    chart_cost_impact()
    chart_deflated_sharpe()
    chart_random_distribution("full_realistic",
                                "05_random_vs_model.png",
                                "realistic costs")
    chart_sma250_test()
    chart_effective_n()
    diagram_walk_forward()
    diagram_multiple_comparison()
    chart_confidence_intervals()
    diagram_survivorship()
    chart_bottom_line()
    print("\nAll visuals written to", OUT)


if __name__ == "__main__":
    sys.exit(main() or 0)

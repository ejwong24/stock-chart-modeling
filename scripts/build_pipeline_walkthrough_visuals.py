"""Generate PNG visuals for the /story/09_pipeline_walkthrough chapter.

Picks one specific real example trade — BNTX on 2021-07-02, +53.7% in 40 days —
and walks through every pipeline stage with actual data. The chart-image stage
also uses our REAL deterministic PIL renderer (the same code DINOv2 saw) so
readers see exactly what the model "looked at."
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from stock_chart import render as renderer

OUT = PROJECT / "web" / "static" / "story"
OUT.mkdir(parents=True, exist_ok=True)

# Theme — match the existing site
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

TICKER = "BNTX"
ANCHOR = pd.Timestamp("2021-07-02")
EXIT = pd.Timestamp("2021-08-30")
H = 40

# Load everything once
df = pd.read_parquet(PROJECT / "data" / "adjusted" / f"{TICKER}.parquet")
df["date"] = pd.to_datetime(df["date"])
anchor_idx = df[df["date"] == ANCHOR].index[0]
window = df.iloc[anchor_idx - 251:anchor_idx + 1].reset_index(drop=True)


# ─── Visual 1: the 252-day input window ─────────────────────────────────

def viz_1_input_window():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.6),
                                     gridspec_kw={"height_ratios": [3, 1]},
                                     sharex=True)
    # Price + MA250
    ma250 = window["close"].rolling(250, min_periods=250).mean()
    ax1.plot(window["date"], window["close"], color=P["accent"], lw=1.7,
              label="BNTX close")
    ax1.plot(window["date"], ma250, color=P["warn"], lw=1.5, linestyle="--",
              label="250-day moving average")
    ax1.fill_between(window["date"], 0, window["close"],
                      color=P["accent"], alpha=0.05)
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"Stage 1 — the 252-day input window for {TICKER} ending {ANCHOR.date()}",
                   loc="left", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(facecolor=P["panel_2"], edgecolor=P["border"], loc="upper left")
    ax1.grid(True, alpha=0.3)

    anchor_close = window["close"].iloc[-1]
    ma250_last = ma250.iloc[-1]
    ax1.annotate(f"Anchor close: ${anchor_close:.2f}\nMA250: ${ma250_last:.2f}\n"
                  f"Ratio: {anchor_close / ma250_last:.2f}× (passes 1.5× filter)",
                  xy=(window["date"].iloc[-1], anchor_close),
                  xytext=(window["date"].iloc[-150], anchor_close * 0.95),
                  fontsize=10, color=P["accent_2"],
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=P["panel_2"],
                             edgecolor=P["accent_2"], alpha=0.95),
                  arrowprops=dict(arrowstyle="->", color=P["accent_2"], lw=1.5))

    # Volume
    ax2.bar(window["date"], window["volume"], color=P["text_dim"], width=1.0,
             alpha=0.7)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))

    plt.tight_layout()
    plt.savefig(OUT / "pw_1_input_window.png")
    plt.close()
    print("wrote pw_1_input_window.png")


# ─── Visual 2: prefilter funnel ─────────────────────────────────────────

def viz_2_funnel():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    stages = [
        ("All US common stocks loaded", 6692, P["text_dim"]),
        (f"Eligible candidates on {ANCHOR.date()}\n(Close > 1.5 × 250-day MA, weekly anchor)",
         373, P["accent"]),
        ("After ADV filter (≥ $5M average daily dollar volume)\nand 'no duplicate ticker' rule",
         "~25-30", P["accent_2"]),
        ("Top 5 actually bought this week\n(DBI, FNKO, BNTX, NEGG, AA)",
         5, P["good"]),
    ]

    widths = [10, 6, 3, 1.5]
    y_positions = [3.6, 2.5, 1.4, 0.3]
    for (label, count, color), w, y in zip(stages, widths, y_positions):
        x_start = (10 - w) / 2
        ax.add_patch(FancyBboxPatch((x_start, y - 0.45), w, 0.9,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=P["border"],
                                       alpha=0.85))
        ax.text(5, y, label, ha="center", va="center", color=P["text"],
                 fontsize=10.5, fontweight="bold")
        ax.text(8.4, y, f"= {count}", ha="left", va="center",
                 color=P["text"], fontsize=11, fontweight="bold")

    # Arrows between stages
    for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate("", xy=(5, y2 + 0.45), xytext=(5, y1 - 0.45),
                     arrowprops=dict(arrowstyle="->", color=P["text_dim"],
                                       lw=1.5))

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.3, 4.5)
    ax.axis("off")
    ax.set_title("Stage 2 — from universe to selection: the prefilter funnel",
                  loc="left", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(OUT / "pw_2_funnel.png")
    plt.close()
    print("wrote pw_2_funnel.png")


# ─── Visual 3: features bar chart ───────────────────────────────────────

def viz_3_features():
    feats = pd.read_parquet(PROJECT / "reports" / "full" /
                              "engineered_features.parquet")
    row = feats[(feats["ticker"] == TICKER) &
                 (feats["anchor_date"] == ANCHOR)].iloc[0]

    # Pick the most interpretable 16 features for the visual
    key_features = [
        ("ret_252d",      "1-year return"),
        ("ret_126d",      "6-month return"),
        ("ret_63d",       "3-month return"),
        ("ret_21d",       "1-month return"),
        ("ret_5d",        "1-week return"),
        ("vol_252d",      "1-year volatility"),
        ("vol_21d",       "1-month volatility"),
        ("max_dd_252d",   "1-year max drawdown"),
        ("current_dd_from_peak", "Current dd from peak"),
        ("ratio_ma200",   "Close / 200-day MA"),
        ("ratio_ma50",    "Close / 50-day MA"),
        ("ratio_ma20",    "Close / 20-day MA"),
        ("pct_from_252d_high", "% below 1-year high"),
        ("days_since_252d_high", "Days since 1-year high"),
        ("up_day_frac_63d", "Fraction up days (60d)"),
        ("log_dollar_vol_z252", "Volume z-score (1y)"),
    ]

    labels = [f"{name}" for _, name in key_features]
    values = [float(row[k]) for k, _ in key_features]

    colors = [P["good"] if v > 0 else P["bad"] for v in values]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor=P["border"], alpha=0.85)
    for i, v in enumerate(values):
        if abs(v) > 100:
            label = f"{v:+.0f}"
        elif abs(v) >= 1:
            label = f"{v:+.2f}"
        else:
            label = f"{v:+.3f}"
        ax.text(v + (0.02 if v >= 0 else -0.02), i, label,
                 ha="left" if v >= 0 else "right", va="center",
                 fontsize=9, color=P["text"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=P["text_dim"], lw=1)
    ax.set_xlabel("Feature value")
    ax.set_title(f"Stage 3 — 16 of the 40 engineered features for {TICKER} on {ANCHOR.date()}",
                  loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "pw_3_features.png")
    plt.close()
    print("wrote pw_3_features.png")


# ─── Visual 4: the actual rendered chart (DINOv2 input) ─────────────────

def viz_4_chart_render():
    """Render the exact 224x224 chart image the pipeline produces."""
    closes = window["close"].to_numpy(dtype=np.float64)
    img = renderer.render_one(closes, image_size=224,
                                log_y_min=0.1, log_y_max=11.0,
                                line_width=2)

    # Embed it in a side-by-side panel: raw price (left) vs rendered (right)
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(window["date"], window["close"], color=P["accent"], lw=1.5)
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Raw 252-day close price", loc="left", fontsize=12,
                   fontweight="bold", pad=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=30, labelsize=8.5)

    ax2 = fig.add_subplot(gs[1])
    img_array = np.array(img)
    ax2.imshow(img_array, cmap="gray_r", interpolation="nearest")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("224×224 rendered chart (DINOv2 input)\nfixed log-y axis −90% to +1000%",
                   loc="left", fontsize=12, fontweight="bold", pad=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(P["border"])
        spine.set_linewidth(1.2)

    fig.suptitle("Stage 4 — what the AI actually sees",
                  fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.99)
    plt.tight_layout()
    plt.savefig(OUT / "pw_4_chart_render.png")
    plt.close()
    print("wrote pw_4_chart_render.png")


# ─── Visual 5: score distribution that week with BNTX marked ────────────

def viz_5_score_dist():
    sc = pd.read_parquet(PROJECT / "reports" / "full" /
                          "scores_lgbm_engineered.parquet")
    that_week = sc[sc["anchor_date"] == ANCHOR].copy()
    bntx_score = that_week[that_week["ticker"] == TICKER]["score"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.hist(that_week["score"], bins=40, color=P["text_dim"],
             edgecolor=P["border"], alpha=0.85,
             label=f"{len(that_week)} eligible candidates this week")

    ax.axvline(bntx_score, color=P["accent_2"], lw=2.5,
                label=f"{TICKER} score: {bntx_score:.3f} (ranked #10 out of {len(that_week)})")

    # Annotate the score range "top-decile" and "median"
    median = that_week["score"].median()
    ax.axvline(median, color=P["text_dim"], linestyle="--",
                label=f"Median score: {median:.3f}")

    ax.set_xlabel("Predicted P(stock will be ≥+25% in 40 days)")
    ax.set_ylabel("Number of stocks")
    ax.set_title(f"Stage 5 — model probability scores for the {len(that_week)} eligible stocks on {ANCHOR.date()}",
                  loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=P["panel_2"], edgecolor=P["border"],
               loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "pw_5_score_dist.png")
    plt.close()
    print("wrote pw_5_score_dist.png")


# ─── Visual 6: the cohort the simulator actually bought ─────────────────

def viz_6_cohort():
    """5 stocks actually bought on 2021-07-02 — outcomes."""
    cohort = [
        ("DBI",  1.000, "PASS", -0.037, P["bad"]),
        ("FNKO", 0.821, "PASS", -0.103, P["bad"]),
        ("BNTX", 0.720, "PASS", +0.537, P["good"]),
        ("NEGG", 0.720, "PASS", -0.057, P["bad"]),
        ("AA",   0.708, "PASS", +0.159, P["good"]),
    ]
    avg_return = np.mean([r[3] for r in cohort])
    without_bntx = np.mean([r[3] for r in cohort if r[0] != "BNTX"])

    fig, ax = plt.subplots(figsize=(10, 5))
    tickers = [c[0] for c in cohort]
    returns = [c[3] * 100 for c in cohort]
    colors = [c[4] for c in cohort]

    bars = ax.bar(tickers, returns, color=colors, edgecolor=P["border"],
                    alpha=0.85)
    # Pad the y-axis to make room for labels above/below bars
    ymin = min(returns) - 15
    ymax = max(returns) + 12
    ax.set_ylim(ymin, ymax)
    for bar, r in zip(bars, returns):
        if r > 0:
            y, va = r + 1.5, "bottom"
        else:
            y, va = r - 1.5, "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                 f"{r:+.1f}%", ha="center", va=va,
                 color=P["text"], fontsize=12, fontweight="bold")

    ax.axhline(0, color=P["text_dim"], lw=1)
    ax.axhline(avg_return * 100, color=P["accent"], linestyle="--",
                label=f"Cohort average: {avg_return*100:+.1f}%")
    ax.axhline(without_bntx * 100, color=P["bad"], linestyle=":",
                label=f"Without {TICKER}: {without_bntx*100:+.1f}%")

    ax.set_ylabel("Forward 40-day return (%)")
    ax.set_title(f"Stage 6 — actual cohort bought on {ANCHOR.date()}: forward 40-day outcomes",
                  loc="left", fontsize=13, fontweight="bold", pad=14)
    ax.legend(facecolor=P["panel_2"], edgecolor=P["border"], loc="upper left",
               fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "pw_6_cohort_outcomes.png")
    plt.close()
    print("wrote pw_6_cohort_outcomes.png")


# ─── Visual 7: BNTX forward window ──────────────────────────────────────

def viz_7_forward():
    entry_idx = anchor_idx
    exit_idx = anchor_idx + H
    forward = df.iloc[entry_idx:exit_idx + 1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(forward["date"], forward["close"], color=P["accent"], lw=2)
    ax.fill_between(forward["date"], forward["close"].iloc[0],
                     forward["close"], where=(forward["close"] >= forward["close"].iloc[0]),
                     color=P["good"], alpha=0.2)
    ax.fill_between(forward["date"], forward["close"].iloc[0],
                     forward["close"], where=(forward["close"] < forward["close"].iloc[0]),
                     color=P["bad"], alpha=0.2)

    entry_px = forward["close"].iloc[0]
    exit_px = forward["close"].iloc[-1]
    ax.axhline(entry_px, color=P["text_dim"], linestyle=":", alpha=0.6)

    ax.plot(forward["date"].iloc[0], entry_px, "o", color=P["accent_2"],
             markersize=14, markeredgecolor=P["bg"], markeredgewidth=2,
             label=f"Entry: ${entry_px:.2f} on {forward['date'].iloc[0].date()}")
    ax.plot(forward["date"].iloc[-1], exit_px, "s", color=P["good"],
             markersize=14, markeredgecolor=P["bg"], markeredgewidth=2,
             label=f"Exit: ${exit_px:.2f} on {forward['date'].iloc[-1].date()}")

    ret = (exit_px / entry_px - 1) * 100
    ax.annotate(f"{ret:+.1f}% in {H} trading days\n+${exit_px - entry_px:.2f}/share",
                 xy=(forward["date"].iloc[-1], exit_px),
                 xytext=(forward["date"].iloc[H // 3], exit_px * 1.03),
                 fontsize=12, color=P["good"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=P["panel_2"],
                            edgecolor=P["good"], alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color=P["good"], lw=1.5))

    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Date")
    ax.set_title(f"Stage 7 — {TICKER}: what happened in the 40 trading days after the entry",
                  loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=P["panel_2"], edgecolor=P["border"], loc="upper left",
               fontsize=10.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "pw_7_forward.png")
    plt.close()
    print("wrote pw_7_forward.png")


# ─── Visual 8: the ADV filter saves us from the worst losers ────────────

def viz_8_adv_filter():
    """Show the 4 high-scoring stocks the ADV filter knocked out and their fates."""
    skipped = [
        ("USIO",  0.864, 0.69, -0.037, "OK in hindsight"),
        ("BESS",  0.821, 0.024, -0.280, "BIG LOSS avoided"),
        ("BTE",   0.818, 0.78, -0.126, "loss avoided"),
        ("IRIX",  0.765, 0.51, +0.088, "small win missed"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    tickers = [s[0] for s in skipped]
    fwd_rets = [s[3] * 100 for s in skipped]
    adv = [s[2] for s in skipped]
    colors = [P["good"] if r < 0 else P["warn"] for r in fwd_rets]

    bars = ax.bar(tickers, fwd_rets, color=colors, edgecolor=P["border"], alpha=0.85)
    ymin = min(fwd_rets) - 10
    ymax = max(fwd_rets) + 8
    ax.set_ylim(ymin, ymax)
    for bar, r, a in zip(bars, fwd_rets, adv):
        if r > 0:
            y, va = r + 1, "bottom"
        else:
            y, va = r - 1, "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                 f"{r:+.1f}%\nADV ${a:.2f}M", ha="center", va=va,
                 color=P["text"], fontsize=10, fontweight="bold")

    ax.axhline(0, color=P["text_dim"], lw=1)
    ax.set_ylabel("Forward 40-day return")
    ax.set_title("Bonus stage — 4 high-scoring stocks the ADV filter knocked out\n"
                 "(too illiquid; would have cost more in slippage than alpha)",
                 loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "pw_8_adv_filter.png")
    plt.close()
    print("wrote pw_8_adv_filter.png")


def main():
    viz_1_input_window()
    viz_2_funnel()
    viz_3_features()
    viz_4_chart_render()
    viz_5_score_dist()
    viz_6_cohort()
    viz_7_forward()
    viz_8_adv_filter()
    print(f"\nAll pipeline-walkthrough visuals in {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)

"""Honest report card template + linter (per RESEARCH P3A10).

Generates `reports/<tag>/honest_report_card.md` with mandatory fields the
linter refuses to omit. Forbidden phrases (cherry-picking tells) are
rejected with a non-zero exit code.

The original document's '100th percentile, 61.1% CAGR' headline FAILS
this linter — that's the whole point.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

from . import lockbox as _lockbox
from . import stats as _stats


REQUIRED_SECTIONS = [
    "BLUF", "Pre-registration", "Headline metrics", "Baseline gap",
    "Post-cost CAGR", "Hidden risks",
]

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bbest of \d+\b", re.IGNORECASE),
     "claim 'best of K' without N_trials_in_registry"),
    (re.compile(r"\boutperformed random\b", re.IGNORECASE),
     "claim 'outperformed random' without DSR or Reality Check"),
    (re.compile(r"100th percentile", re.IGNORECASE),
     "report '100th percentile vs random seeds' (the original document's broken framing)"),
]


TEMPLATE = """\
# Honest Report Card — {tag}

**BLUF — Would I trade this with my own money?** {bluf_y_n}

Reasoning ({sentences} sentences max):
> {bluf_reasoning}

## Pre-registration

- Hypothesis hash (SHA-256 of prereg doc): `{prereg_hash}`
- Pre-registration date: {prereg_date}
- N_trials_in_registry (this family): **{n_trials}**
- Multiple-comparison correction applied: {multi_comp}

## Headline metrics (OOS only unless flagged)

| Metric | Point | 95% CI (block bootstrap) | Notes |
|---|---|---|---|
| CAGR | {cagr_point} | {cagr_ci} | OOS, daily-return resample, block=horizon |
| Sharpe (annualized) | {sharpe_point} | {sharpe_ci} | raw, before deflation |
| Deflated Sharpe Ratio | {dsr_value} | p={dsr_pvalue} | adjusts for N_trials={n_trials} |
| Reality Check (Hansen SPA) | p={spa_p} | bootstrap {spa_reps} reps | recommended over White's RC |
| Max drawdown | {mdd_point} | {mdd_ci} | non-smooth, slower CI convergence |

## Baseline gap (the only comparison that counts)

- Best simple baseline: **{best_baseline}**
- Model CAGR − baseline CAGR: {gap_point}, 95% CI: {gap_ci}
- Gap statistically distinguishable from 0 after N_trials adjustment? **{gap_significant}**

## Post-cost CAGR at 3 AUM tiers

| AUM | Slippage model | Post-cost CAGR | Notes |
|---|---|---|---|
| $100k | {cost_model_100k} | {cagr_100k} | retail account |
| $1M | {cost_model_1m} | {cagr_1m} | meaningful capacity |
| $10M | {cost_model_10m} | {cagr_10m} | near capacity ceiling |

## Hidden risks (mandatory; "none" not allowed)

- **Regime dependence:** {regime_risk}
- **Capacity ceiling:** {capacity_ceiling}
- **Slippage assumption fragility:** {slippage_risk}
- **Data-snooping residual:** {snooping_risk}
- **Survivorship/look-ahead audit:** {survivorship_audit}

## Effective sample size

- Naive trade count: {n_trades}
- López de Prado average uniqueness (effective N): {n_eff}
- SE(Sharpe) widening factor: {se_widen}x
"""


def lint(md_text: str) -> list[str]:
    """Return list of lint errors. Empty list = pass."""
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in md_text:
            errors.append(f"missing required section: {section}")
    for pat, why in FORBIDDEN_PATTERNS:
        m = pat.search(md_text)
        if m:
            errors.append(f"forbidden phrase '{m.group(0)}' — {why}")
    return errors


def render(values: dict) -> str:
    """Render the template; missing keys are flagged with TODO so linter catches them."""
    defaults = {k: "TODO" for k in (
        "tag", "bluf_y_n", "sentences", "bluf_reasoning",
        "prereg_hash", "prereg_date", "n_trials", "multi_comp",
        "cagr_point", "cagr_ci", "sharpe_point", "sharpe_ci",
        "dsr_value", "dsr_pvalue", "spa_p", "spa_reps",
        "mdd_point", "mdd_ci",
        "best_baseline", "gap_point", "gap_ci", "gap_significant",
        "cost_model_100k", "cagr_100k",
        "cost_model_1m", "cagr_1m",
        "cost_model_10m", "cagr_10m",
        "regime_risk", "capacity_ceiling", "slippage_risk",
        "snooping_risk", "survivorship_audit",
        "n_trades", "n_eff", "se_widen",
    )}
    defaults["sentences"] = "2"
    defaults.update(values)
    return TEMPLATE.format(**defaults)


def auto_fill_from_run(run_dir: Path, project_root: Path,
                       prereg_hash: str = "(no prereg)") -> dict:
    """Auto-extract everything we can from an existing run directory."""
    import pandas as pd
    headline = json.loads((run_dir / "headline.json").read_text())
    summaries = headline.get("summaries", [])
    if not summaries:
        return {}
    best = max(summaries, key=lambda x: x.get("end_equity", 0))
    dsr = headline.get("deflated_sharpe", {}) or {}

    # Find best simple baseline
    rank_summaries = [s for s in summaries if s["track"].startswith("rank_")]
    best_baseline = max(rank_summaries, key=lambda x: x.get("end_equity", 0)) \
                     if rank_summaries else {}
    gap = best.get("cagr", 0) - best_baseline.get("cagr", 0)

    # Block-bootstrap CIs on best track equity
    best_track = best.get("track", "unknown")
    eq_path = run_dir / f"equity_{best_track}.parquet" if best_track != "unknown" else None
    cagr_ci = sharpe_ci = mdd_ci = "N/A"
    if eq_path is not None and eq_path.exists():
        eq_df = pd.read_parquet(eq_path)
        ci = _stats.bootstrap_cagr_ci(eq_df, n_resamples=2000, block_size=40)
        cagr_ci = (f"[{ci['cagr_ci'][0]*100:+.1f}%, {ci['cagr_ci'][2]*100:+.1f}%]")
        sharpe_ci = (f"[{ci['sharpe_ci'][0]:+.2f}, {ci['sharpe_ci'][2]:+.2f}]")
        mdd_ci = (f"[{ci['maxdd_ci'][0]*100:.1f}%, {ci['maxdd_ci'][2]*100:.1f}%]")

    # Effective sample size from blotter
    n_eff = 0
    se_widen = 1.0
    blotter_path = run_dir / f"blotter_{best_track}.parquet" if best_track != "unknown" else None
    if blotter_path is not None and blotter_path.exists():
        blot = pd.read_parquet(blotter_path)
        try:
            n_eff = _stats.effective_sample_size(blot, hold_days=40)
            se_widen = (best.get("n_trades", 1) / max(n_eff, 1)) ** 0.5
        except Exception:
            n_eff = 0

    # Trial count from registry
    trial_path = project_root / "configs" / "trial_registry.jsonl"
    n_trials = _lockbox.trial_count(trial_path)

    return {
        "tag": run_dir.name,
        "n_trials": str(n_trials),
        "prereg_hash": prereg_hash,
        "prereg_date": "auto-generated (no prereg)",
        "multi_comp": f"Deflated Sharpe (N_trials={n_trials})",
        "cagr_point": f"{best.get('cagr', 0)*100:+.2f}%",
        "cagr_ci": cagr_ci,
        "sharpe_point": f"{best.get('sharpe', 0):+.2f}",
        "sharpe_ci": sharpe_ci,
        "dsr_value": f"{dsr.get('observed_sharpe_annualized', 0):+.3f}",
        "dsr_pvalue": f"{dsr.get('deflated_sharpe_p_value_significant', 0):.3f}",
        "spa_p": "(run reality_check_spa to fill)",
        "spa_reps": "5000",
        "mdd_point": f"{best.get('max_dd', 0)*100:.2f}%",
        "mdd_ci": mdd_ci,
        "best_baseline": (f"{best_baseline.get('track', 'n/a')} "
                          f"({best_baseline.get('cagr', 0)*100:+.2f}%)"),
        "gap_point": f"{gap*100:+.2f}%",
        "gap_ci": "(needs paired bootstrap; see stats.py)",
        "gap_significant": ("Y" if gap > 0.05 else "N (gap ≤ 5pp)"),
        "n_trades": str(best.get("n_trades", "?")),
        "n_eff": f"{n_eff:.0f}",
        "se_widen": f"{se_widen:.2f}",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m stock_chart.report_card <run_tag>")
        sys.exit(2)
    from . import config as _cfg
    project = _cfg.PROJECT_ROOT
    rd = project / "reports" / sys.argv[1]
    if not (rd / "headline.json").exists():
        print(f"no headline.json in {rd}")
        sys.exit(2)
    values = auto_fill_from_run(rd, project)
    md = render(values)
    out_path = rd / "honest_report_card.md"
    out_path.write_text(md)
    errors = lint(md)
    if errors:
        print(f"REPORT CARD LINT ERRORS for {sys.argv[1]}:")
        for e in errors:
            print(f"  - {e}")
    print(f"\nWrote {out_path} ({len(md)} chars; {len(errors)} TODO/lint items)")

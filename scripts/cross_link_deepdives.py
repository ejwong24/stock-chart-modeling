"""Inject cross-links into the pipeline deep-dive markdown files.

For each known concept (e.g. "deflated Sharpe", "Almgren-Chriss", "PCA-64"),
this script finds the FIRST plain-text occurrence in every deep-dive file
(skipping the file that IS the target) and wraps it in a Markdown link to
the corresponding /story/09/{slug} page.

The pass is idempotent: occurrences that are already inside an existing
Markdown link (`[...](...)`) or inside a code block (``` ... ``` or `...`)
are left alone, so re-running the script produces no further changes.

Run from the project root:
    python scripts/cross_link_deepdives.py [--dry-run]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEEPDIVE_DIR = PROJECT / "RESEARCH" / "pipeline_deepdives"

# (regex pattern, replacement_url, target_slug_to_skip_self_link)
# Patterns: \b prefix/suffix, case-insensitive. First match wins.
LINK_RULES: list[tuple[str, str, str]] = [
    # Round 1
    (r"\bDINOv2(?:\s+ViT-S/14)?\b", "/story/09/01_dinov2_architecture", "01_dinov2_architecture"),
    (r"\bPCA-?64\b", "/story/09/02_pca_math", "02_pca_math"),
    (r"\b(?:40 )?engineered features\b", "/story/09/03_engineered_features", "03_engineered_features"),
    (r"\bLogisticRegression(?:\s+with\s+class_weight)?\b", "/story/09/04_logistic_regression", "04_logistic_regression"),
    (r"\bvolume PCA-?64\b", "/story/09/05_volume_processing", "05_volume_processing"),
    (r"\bsimulator(?:'s)? (?:inner|per-day) loop\b", "/story/09/06_simulator_loop", "06_simulator_loop"),
    (r"\bMA250 prefilter\b", "/story/09/07_ma250_prefilter", "07_ma250_prefilter"),
    (r"\bwalk-forward (?:embargo|cross-validation|fold|CV)\b", "/story/09/08_walkforward_embargo", "08_walkforward_embargo"),
    (r"\bAlmgren[-–]Chriss\b", "/story/09/09_almgren_chriss", "09_almgren_chriss"),
    (r"\bDeflated Sharpe(?:\s+Ratio)?\b", "/story/09/10_deflated_sharpe", "10_deflated_sharpe"),
    (r"\btrailing stop\b", "/story/09/11_trailing_stop_interactions", "11_trailing_stop_interactions"),
    (r"\b(?:reproducibility|random) seeds?\b", "/story/09/12_reproducibility_seeds", "12_reproducibility_seeds"),
    # Round 2
    (r"\bdata acquisition\b", "/story/09/13_data_acquisition", "13_data_acquisition"),
    (r"\buniverse construction\b", "/story/09/14_universe_construction", "14_universe_construction"),
    (r"\bTailscale serve\b", "/story/09/15_deployment_tailscale", "15_deployment_tailscale"),
    (r"\btrial registry\b", "/story/09/16_trial_registry", "16_trial_registry"),
    (r"\bdisaster recovery\b", "/story/09/17_disaster_recovery", "17_disaster_recovery"),
    (r"\bweekly anchors?\b", "/story/09/18_weekly_anchors", "18_weekly_anchors"),
    (r"\blabel grid\b", "/story/09/19_label_grid", "19_label_grid"),
    (r"\bsimple baselines?\b", "/story/09/20_simple_baselines", "20_simple_baselines"),
    (r"\bvalidation modes?\b", "/story/09/21_validation_modes", "21_validation_modes"),
    (r"\b(?:stationary )?block bootstrap\b", "/story/09/22_block_bootstrap_params", "22_block_bootstrap_params"),
    (r"\bmultiple[- ]comparison\b", "/story/09/23_multiple_comparison_landscape", "23_multiple_comparison_landscape"),
    (r"\beffective(?:\s+sample\s+size|\s+N)\b", "/story/09/24_effective_sample_size", "24_effective_sample_size"),
    (r"\bfrictions beyond impact\b", "/story/09/25_frictions_beyond_impact", "25_frictions_beyond_impact"),
    (r"\bcapacity ceiling\b", "/story/09/26_capacity_ceiling", "26_capacity_ceiling"),
    # Round 3
    (r"\bSharpe ratio\b", "/story/09/27_sharpe_ratio", "27_sharpe_ratio"),
    (r"\bCalmar(?:\s+ratio)?\b", "/story/09/28_cagr_drawdown_calmar", "28_cagr_drawdown_calmar"),
    (r"\bfixed log-y axis\b", "/story/09/29_fixed_log_y_axis", "29_fixed_log_y_axis"),
    (r"\bisotonic(?:\s+calibration|\s+regression)?\b", "/story/09/30_isotonic_calibration", "30_isotonic_calibration"),
    (r"\bLightGBM\b", "/story/09/31_lightgbm_internals", "31_lightgbm_internals"),
    (r"\bposition[- ]size(?:\s+formula)?\b", "/story/09/32_position_size_formula", "32_position_size_formula"),
    (r"\bADV20\b", "/story/09/33_adv20_metric", "33_adv20_metric"),
    (r"\bblotter\b", "/story/09/34_blotter_equity_summary", "34_blotter_equity_summary"),
    (r"\blog returns?\b", "/story/09/35_log_vs_simple_returns", "35_log_vs_simple_returns"),
    (r"\ballocation budget\b", "/story/09/36_allocation_budget", "36_allocation_budget"),
    (r"\btrading calendar\b", "/story/09/37_trading_calendar", "37_trading_calendar"),
    (r"\bSPY beta\b", "/story/09/38_spy_beta", "38_spy_beta"),
]


def _split_protected_regions(text: str) -> list[tuple[str, bool]]:
    """Split text into (chunk, is_protected) tuples.

    Protected = inside a code block (``` ... ``` or `...`) or inside an
    existing Markdown link `[...](...)`. We never rewrite inside these.
    """
    out: list[tuple[str, bool]] = []
    i = 0
    n = len(text)
    while i < n:
        # Fenced code block
        if text.startswith("```", i):
            end = text.find("```", i + 3)
            if end == -1:
                out.append((text[i:], True))
                break
            out.append((text[i:end + 3], True))
            i = end + 3
            continue
        # Inline code span
        if text[i] == "`":
            end = text.find("`", i + 1)
            if end == -1:
                out.append((text[i:], True))
                break
            out.append((text[i:end + 1], True))
            i = end + 1
            continue
        # Existing Markdown link
        if text[i] == "[":
            link_end = text.find("]", i + 1)
            if link_end != -1 and link_end + 1 < n and text[link_end + 1] == "(":
                paren_end = text.find(")", link_end + 2)
                if paren_end != -1:
                    out.append((text[i:paren_end + 1], True))
                    i = paren_end + 1
                    continue
        # Plain text run — extend until the next protected delimiter.
        # Start at i+1 to guarantee forward progress: we only reach here when
        # text[i] is a '[' that did NOT open a Markdown link (e.g. a prose CI
        # like "[-0.18, +1.21]" or "[CLS]"). Consuming that '[' as a literal
        # avoids an infinite loop (j would otherwise stay == i forever).
        j = i + 1
        while j < n and text[j] not in "`[":
            if text.startswith("```", j):
                break
            j += 1
        out.append((text[i:j], False))
        i = j
    return out


def inject_links(text: str, self_slug: str) -> tuple[str, int]:
    """Inject one link per rule per file (first occurrence, plain text only).
    Returns (new_text, n_replacements)."""
    chunks = _split_protected_regions(text)
    n_total = 0
    for pattern, url, target_slug in LINK_RULES:
        # Don't link to ourselves
        if target_slug == self_slug:
            continue
        # Idempotency guard: if a link to this target already exists anywhere
        # in the file, skip this rule entirely. Without this, re-running the
        # script links a *new* (later) occurrence each pass — because the
        # earlier one is now inside a protected [...](...) span — so link
        # count grows unbounded across runs. With it, re-runs are a no-op.
        if f"]({url})" in text:
            continue
        rgx = re.compile(pattern, re.IGNORECASE)
        # Find the FIRST plain-text chunk that contains a match.
        for idx, (chunk, protected) in enumerate(chunks):
            if protected:
                continue
            m = rgx.search(chunk)
            if not m:
                continue
            # Build replacement: [matched_text](url)
            matched = chunk[m.start():m.end()]
            replacement = f"[{matched}]({url})"
            new_chunk = chunk[:m.start()] + replacement + chunk[m.end():]
            chunks[idx] = (new_chunk, False)
            n_total += 1
            break  # only the first match of this pattern in this file
    return "".join(c for c, _ in chunks), n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                      help="Don't write; just report counts.")
    args = ap.parse_args()

    md_files = sorted(DEEPDIVE_DIR.glob("*.md"))
    print(f"Scanning {len(md_files)} deep-dive files in {DEEPDIVE_DIR}")
    total_inserted = 0
    for p in md_files:
        self_slug = p.stem
        text = p.read_text()
        new_text, n = inject_links(text, self_slug)
        if n > 0:
            print(f"  {self_slug}: +{n} links")
            if not args.dry_run:
                p.write_text(new_text)
            total_inserted += n
    print(f"\nTotal links inserted: {total_inserted}")
    if args.dry_run:
        print("(dry run — no files modified)")


if __name__ == "__main__":
    sys.exit(main() or 0)

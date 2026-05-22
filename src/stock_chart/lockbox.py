"""Lockbox + trial-registry + pre-registration infrastructure.

Implements problem 3 from RESEARCH/03_lockbox_protocol.md.

Three artifacts:
  - reports/lockbox.audit.jsonl   — append-only event log
  - reports/lockbox.registry.json — one-shot 2025 evaluation per (H,T,model)
  - configs/trial_registry.jsonl  — append-only log of every config tried,
                                     drives N_trials in deflated Sharpe

Decision rule: the FIRST 2025 evaluation for any (H, T, model) tuple is the
headline — good or bad, no iteration.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def sha256_of_dict(d: dict) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class LockboxError(Exception):
    pass


def audit(audit_path: Path, event: str, payload: dict | None = None) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _now_iso(), "event": event, "git_sha": _git_sha(),
           "user": os.getenv("USER", "unknown"), **(payload or {})}
    with open(audit_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def claim_lockbox(registry_path: Path, *, horizon: int, threshold: float,
                   model: str, allow_overwrite: bool = False) -> dict:
    """Claim a one-shot 2025 evaluation for this (H, T, model) tuple.

    Refuses to overwrite an existing claim unless `allow_overwrite=True`.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    key = f"{horizon}|{threshold}|{model}"
    reg = {}
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text())
        except Exception:
            reg = {}
    if key in reg and not allow_overwrite:
        raise LockboxError(
            f"Lockbox already claimed for {key} at {reg[key].get('ts')}; "
            f"refuse second claim. Pass allow_overwrite=True to override "
            f"(and document why in lockbox-broken.json).")
    reg[key] = {"ts": _now_iso(), "claimed": True, "git_sha": _git_sha()}
    registry_path.write_text(json.dumps(reg, indent=2))
    return reg[key]


def log_trial(trial_registry_path: Path, *, config: dict, headline_sharpe: float,
              n_obs: int, prereg_id: str | None = None) -> dict:
    """Append a trial entry. Every distinct (model, label, H, T, universe)
    counts as a separate trial for deflated Sharpe / Reality Check.
    """
    trial_registry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _now_iso(),
        "trial_id": f"t_{int(time.time()*1000)}",
        "sha_config": sha256_of_dict(config)[:16],
        "sha_code": _git_sha()[:16],
        "model": config.get("model"),
        "label": config.get("label"),
        "horizon_d": config.get("horizon_d"),
        "threshold_q": config.get("threshold_q"),
        "universe": config.get("universe"),
        "headline_sharpe": float(headline_sharpe),
        "n_obs": int(n_obs),
        "prereg_id": prereg_id,
    }
    with open(trial_registry_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def trial_count(trial_registry_path: Path) -> int:
    """Effective N_trials = unique (model, label, horizon, threshold, universe) tuples."""
    if not trial_registry_path.exists():
        return 1
    seen = set()
    for line in trial_registry_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            seen.add((d.get("model"), d.get("label"), d.get("horizon_d"),
                      d.get("threshold_q"), d.get("universe")))
        except Exception:
            continue
    return max(len(seen), 1)


PREREGISTRATION_TEMPLATE = """\
# Preregistration <id>

- **Date (UTC):** <YYYY-MM-DDTHH:MM:SSZ>
- **Author:** <name>
- **Hypothesis:** <one-paragraph statement of what you predict and why>
- **Model spec:** <package version, hyperparameters, seed>
- **Features:** <list, with feature_set_hash>
- **Label:** <name + transformation>
- **Dataset:** <source + date range + universe definition>
- **Splits:** train <years>, val <years>, **OOS <years> (untouched)**
- **Metric:** <primary metric>; secondary: <list>
- **Pass threshold:** <metric >= X AND t-stat >= Y on OOS>
- **Fail action:** <archive, do not iterate on same OOS window>
- **Alternatives considered (and rejected):** <list — prevents post-hoc "we meant to try this">
- **Config hash:** <sha256 of canonical config JSON>
"""

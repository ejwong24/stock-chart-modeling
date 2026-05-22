"""Tests for lockbox.py — audit, claim, trial registry."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import lockbox as lb


def test_sha256_of_dict_deterministic():
    """Same dict → same hash regardless of key order."""
    a = {"horizon": 40, "threshold": 0.25, "model": "lgbm"}
    b = {"model": "lgbm", "threshold": 0.25, "horizon": 40}
    assert lb.sha256_of_dict(a) == lb.sha256_of_dict(b)


def test_sha256_of_dict_differs():
    a = {"horizon": 40, "threshold": 0.25}
    b = {"horizon": 40, "threshold": 0.30}
    assert lb.sha256_of_dict(a) != lb.sha256_of_dict(b)


def test_audit_appends(tmp_path):
    p = tmp_path / "audit.jsonl"
    lb.audit(p, "evt1", {"x": 1})
    lb.audit(p, "evt2", {"y": "hello"})
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["event"] == "evt1" and e1["x"] == 1
    assert e2["event"] == "evt2" and e2["y"] == "hello"
    # Every entry has ts and git_sha
    assert "ts" in e1 and "git_sha" in e1


def test_audit_creates_parent_dirs(tmp_path):
    """audit() should mkdir -p the parent."""
    deep = tmp_path / "a" / "b" / "c.jsonl"
    lb.audit(deep, "ok")
    assert deep.exists()


def test_claim_lockbox_first_succeeds(tmp_path):
    p = tmp_path / "reg.json"
    out = lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lgbm")
    assert out["claimed"] is True
    data = json.loads(p.read_text())
    assert "40|0.25|lgbm" in data


def test_claim_lockbox_second_call_raises(tmp_path):
    """Second claim for same (H, T, model) tuple should refuse."""
    p = tmp_path / "reg.json"
    lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lgbm")
    with pytest.raises(lb.LockboxError):
        lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lgbm")


def test_claim_lockbox_different_tuple_ok(tmp_path):
    """Different (H, T, model) tuples can each claim once."""
    p = tmp_path / "reg.json"
    lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lgbm")
    lb.claim_lockbox(p, horizon=20, threshold=0.25, model="lgbm")
    lb.claim_lockbox(p, horizon=40, threshold=0.30, model="lgbm")
    lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lr")
    data = json.loads(p.read_text())
    assert len(data) == 4


def test_claim_lockbox_allow_overwrite(tmp_path):
    """Explicit override flag should bypass refusal."""
    p = tmp_path / "reg.json"
    lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lgbm")
    lb.claim_lockbox(p, horizon=40, threshold=0.25, model="lgbm",
                       allow_overwrite=True)
    # No exception means OK


def test_log_trial_then_count(tmp_path):
    p = tmp_path / "trials.jsonl"
    lb.log_trial(p, config={"model": "a", "label": "x", "horizon_d": 5,
                              "threshold_q": 0.1, "universe": "us"},
                   headline_sharpe=1.0, n_obs=100)
    lb.log_trial(p, config={"model": "b", "label": "x", "horizon_d": 5,
                              "threshold_q": 0.1, "universe": "us"},
                   headline_sharpe=0.5, n_obs=100)
    # Duplicate of first → still counts as one trial tuple
    lb.log_trial(p, config={"model": "a", "label": "x", "horizon_d": 5,
                              "threshold_q": 0.1, "universe": "us"},
                   headline_sharpe=1.2, n_obs=100)
    assert lb.trial_count(p) == 2


def test_log_trial_with_missing_fields(tmp_path):
    """A trial entry missing some keys should still be loggable."""
    p = tmp_path / "trials.jsonl"
    lb.log_trial(p, config={"model": "lgbm"}, headline_sharpe=0.0, n_obs=0)
    assert p.exists()
    line = p.read_text().splitlines()[0]
    d = json.loads(line)
    assert d["model"] == "lgbm"
    assert d["headline_sharpe"] == 0.0


def test_trial_count_skips_blank_lines_and_bad_json(tmp_path):
    p = tmp_path / "trials.jsonl"
    p.write_text(
        '{"model":"a","label":"x","horizon_d":5,"threshold_q":0.1,"universe":"us"}\n'
        '\n'
        'not valid json at all\n'
        '{"model":"b","label":"x","horizon_d":5,"threshold_q":0.1,"universe":"us"}\n'
    )
    # Should still count the two valid distinct entries
    assert lb.trial_count(p) == 2


def test_trial_count_missing_file(tmp_path):
    """Missing file returns 1 (conservative default)."""
    assert lb.trial_count(tmp_path / "missing.jsonl") == 1

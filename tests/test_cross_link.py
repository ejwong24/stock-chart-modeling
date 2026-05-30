"""Tests for scripts/cross_link_deepdives.py.

Locks down two real bugs found during the round-3 linking work:
  1. _split_protected_regions infinite-looped on a non-link '[' (e.g. a prose
     CI like "[-0.18, +1.21]" or "[CLS]") because the plain-text scanner
     started at j=i and never advanced.
  2. inject_links was non-idempotent: re-running linked a *new* later mention
     each pass, growing link count unbounded.
"""
import importlib.util
import sys
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "cross_link_deepdives", PROJECT / "scripts" / "cross_link_deepdives.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load()


# ─── _split_protected_regions: termination + losslessness ───────────────

@pytest.mark.parametrize("text", [
    "The 95% CI is [-0.18, +1.21] here.",          # bug #1: prose bracket
    "the [CLS] token and close[t+40] math",
    "ends with a bare bracket [",
    "[",                                             # degenerate
    "[[[[nested non-links]]]]",
    "see [Sharpe](/story/09/27_sharpe_ratio) ok",   # real link
    "code `equity[t-1]` span",
    "```\nfenced close[t]\n``` after",
    "",                                              # empty
])
def test_split_terminates_and_is_lossless(text):
    parts = M._split_protected_regions(text)
    # Rejoining must reproduce the input exactly (no chars lost/added)
    assert "".join(c for c, _ in parts) == text


def test_real_link_is_protected():
    parts = M._split_protected_regions("x [a](b) y")
    protected = [c for c, p in parts if p]
    assert "[a](b)" in protected


def test_inline_and_fenced_code_protected():
    parts = M._split_protected_regions("a `b[c]` d ```\ne[f]\n``` g")
    protected = [c for c, p in parts if p]
    assert any("`b[c]`" == c for c in protected)
    assert any(c.startswith("```") for c in protected)


# ─── inject_links: correctness + idempotency ────────────────────────────

def test_injects_first_plain_occurrence():
    txt = "A mention of Sharpe ratio in prose."
    out, n = M.inject_links(txt, self_slug="10_deflated_sharpe")
    assert n == 1
    assert "[Sharpe ratio](/story/09/27_sharpe_ratio)" in out


def test_does_not_link_self():
    txt = "This page is about the Sharpe ratio itself."
    out, n = M.inject_links(txt, self_slug="27_sharpe_ratio")
    assert "](/story/09/27_sharpe_ratio)" not in out


def test_leaves_prose_brackets_intact():
    txt = "The Sharpe ratio CI is [-0.1, +0.2] wide."
    out, _ = M.inject_links(txt, self_slug="10_deflated_sharpe")
    assert "[-0.1, +0.2]" in out  # not mangled into a link


def test_idempotent_second_pass_is_noop():
    txt = "A mention of Sharpe ratio and another Sharpe ratio later."
    once, n1 = M.inject_links(txt, self_slug="10_deflated_sharpe")
    twice, n2 = M.inject_links(once, self_slug="10_deflated_sharpe")
    assert n1 >= 1
    assert n2 == 0, "second pass must insert nothing (idempotency guard)"
    assert once == twice


def test_does_not_link_inside_code():
    txt = "Use `Sharpe ratio` as a column name."
    out, n = M.inject_links(txt, self_slug="10_deflated_sharpe")
    assert n == 0
    assert "`Sharpe ratio`" in out


# ─── corpus-level: the real files must be stable + link-clean ───────────

def test_corpus_links_all_resolve():
    """Every /story/09/<slug> link in the real deep-dive corpus resolves."""
    import re
    d = PROJECT / "RESEARCH" / "pipeline_deepdives"
    if not d.exists():
        pytest.skip("deep-dive corpus not present")
    rgx = re.compile(r"/story/09/([0-9]{2}_[a-z0-9_]+)")
    for md in d.glob("*.md"):
        for slug in rgx.findall(md.read_text()):
            assert (d / f"{slug}.md").exists(), \
                f"{md.name} links to missing /story/09/{slug}"


def test_corpus_is_idempotent_now():
    """Running inject_links on each real file inserts 0 (already linked)."""
    d = PROJECT / "RESEARCH" / "pipeline_deepdives"
    if not d.exists():
        pytest.skip("deep-dive corpus not present")
    for md in d.glob("*.md"):
        _, n = M.inject_links(md.read_text(), self_slug=md.stem)
        assert n == 0, f"{md.name} would gain {n} more links — not stable"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

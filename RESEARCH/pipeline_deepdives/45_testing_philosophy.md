# Testing philosophy — why finite-value tests let a dead feature live

The pipeline ships with 334 tests. They are green. And yet for the project's entire history `beta_spy_63d` was hard-wired to `0.0` for every single stock — a dead feature masquerading as a live one ([why beta was zero](/story/09/42_beta_zero_bug)). The tests didn't fail because they never asked the only question that mattered: *is this number right?* They asked whether it was finite. And `0.0` is exquisitely finite.

This is the story of what separates a test that catches bugs from a test that merely exists.

## Finiteness and shape are necessary but weak

The original beta test looked roughly like this:

```python
def test_beta_is_finite():
    df = compute_features(prices)
    assert "beta_spy_63d" in df.columns
    assert df["beta_spy_63d"].notna().all()
    assert np.isfinite(df["beta_spy_63d"]).all()
```

Every assertion here passes for a column of zeros. Shape checks confirm the plumbing — the column exists, has the right length, contains no `NaN` or `inf`. That is real value: it catches crashes, broadcasting mistakes, and unhandled division-by-zero. But it is a test of *liveness of the pipe*, not *correctness of the water*. A constant flows through the pipe perfectly.

The fix added a **golden-value test** — an input where the correct output is known by construction and the buggy code provably cannot produce it:

```python
def test_beta_unity_when_market_equals_stock():
    # If the "market" series IS the stock's own series,
    # beta must be 1.0 by definition (cov(x,x)/var(x) == 1).
    feats = compute_features(stock, market=stock)
    assert feats["beta_spy_63d"].iloc[-1] == pytest.approx(1.0, abs=1e-6)
```

The buggy code returns `0.0` here. `0.0 != 1.0`. The test fails on the bug and passes on the fix. **The principle: at least one test per feature should pin a value the bug would change.**

## Discriminating assertions: would this test fail if the code were broken?

A subtler trap is the test that passes both before *and* after a fix — it tests nothing about the fix at all. The [look-ahead exit bug](/story/09/40_lookahead_exit_bug) is the cautionary case. The naive regression test would assert on `exit_date` — but the exit date is the *same* whether or not the look-ahead leak is present, so that assertion is inert. The audit instead chose an assertion that genuinely discriminates: through the price gap, the position must remain **open** (`n_positions == 1`).

Before writing any regression test, the question to ask is mutation-style: *if I reintroduced the bug, would this specific assertion go red?* If not, rewrite the assertion until it does.

## Invariants and properties: catch classes, not cases

Golden values pin individual points. **Invariant tests** assert relationships that must hold for *all* runs, catching whole categories of bugs without enumerating inputs. The simulator carries conservation laws ([simulator loop](/story/09/06_simulator_loop), [blotter/equity/summary](/story/09/34_blotter_equity_summary)):

```python
assert cash + position_mtm == pytest.approx(equity)   # nothing created/destroyed
assert end_cash == pytest.approx(end_equity)           # closure with no open positions
```

Any accounting bug — a double-counted fill, a fee charged twice, an MTM drift — violates these regardless of the price path that triggered it. Paired with **determinism tests** ([reproducibility seeds](/story/09/12_reproducibility_seeds)): a bit-identical blotter from the same seed rules out hidden nondeterminism — unordered dict iteration, leaked global RNG state, set-based tie-breaking — that no single golden value would surface.

## Adversarial verification: refute by default

A 334-test suite can still drown in false findings. The audit ([the methodology](/story/09/51_audit_methodology)) used **refute-by-default** verification: every candidate bug was handed to an independent skeptic prompted to *disprove* it — and, critically, to check whether an existing test already covered it. Only findings that survived a genuine attempt to refute them were promoted to fixes. Verification is not "can I imagine this being wrong?" — it is "can I *fail* to make it wrong?"

## Regression tests as bug tombstones

Every confirmed-and-fixed bug earned a named test in `tests/test_audit_fixes.py`, so it can never silently return. The categories:

- **Look-ahead / leakage** — exits and signals that peeked at future bars.
- **Value correctness** — beta unity, feature golden values that reject dead constants.
- **Row alignment** — index/date join integrity across merged frames ([row alignment](/story/09/41_row_alignment)).
- **Accounting invariants** — cash/MTM/equity conservation and closure ([phantom positions](/story/09/49_phantom_positions)).
- **Determinism** — seed-stable blotters and feature outputs.

Each test is a tombstone: it names the bug, encodes the exact condition that resurrects it, and stands guard forever.

## The coverage gaps that remain

Honesty matters more than a green badge. Known holes:

- **`run_pipeline.main` has no end-to-end test.** The [row-alignment landmine](/story/09/41_row_alignment) lived precisely in this untested seam between stages — an integration test *does* exist but it re-implements the stage sequence inline rather than calling the real `main`.
- **The live `forward_pick` path is only unit-tested** ([forward paper trading](/story/09/46_forward_pick_harness)); the wired-together forward flow has no integration assertion.
- **[DINOv2](/story/09/01_dinov2_architecture) and yfinance HTTP paths are smoke-tested, not asserted** — we confirm they don't throw, not that their outputs are correct.

These and other items are tracked in the [feature gaps roadmap](/story/09/47_roadmap).

## The meta-lesson

Tests encode what you *thought* to check. A bug, by definition, is something you *didn't* think to check — so a test suite, however large, is a record of your past imagination, not a proof of correctness. The 334 tests didn't fail to catch beta-zero; they were never pointed at it.

The mitigations are the structural ones, because they don't depend on foresight: **invariant and property tests** that hold across all inputs, **golden values** that reject dead constants, **mutation-style discipline** ("would this test fail if I broke the code?"), and **periodic adversarial audits** that go looking for what the suite assumes. Count of tests is vanity. The question for each one is the only one that matters: *what bug does this test forbid?* See [the hardening story](/story/09/39_hardening_story) for what this discipline caught.

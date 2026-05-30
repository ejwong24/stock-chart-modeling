# How the audit worked — multi-agent bug hunting with adversarial verification

This page documents the *methodology* behind the audit that produced the eleven most recent fixes in [the hardening story](/story/09/39_hardening_story). The bugs themselves are written up elsewhere — [look-ahead exit](/story/09/40_lookahead_exit_bug), [beta zero](/story/09/42_beta_zero_bug), [label corruption](/story/09/43_label_corruption), [phantom positions](/story/09/49_phantom_positions), and the still-open items in the [feature gaps roadmap](/story/09/47_roadmap). What follows is *how* they were found, and why the process was built the way it was.

## The shape: fan-out, then refute, then synthesize

The audit ran in three stages.

**Fan-out (find).** A set of finder agents was launched in parallel. Each one owned a single scope: either one module (the simulator loop, the feature builder, the labeler, the position sizer, the web app) or one *cross-cutting dimension* (look-ahead leakage, row alignment between features and labels, numerical stability, determinism/seeding). Every finder read its assigned code in full — not grep-and-skim, but the whole file or the whole concern — and returned a list of *candidate* bugs with a described trigger.

**Verify (refute).** Each candidate was then handed to a separate skeptic agent. The skeptic's job was the opposite of the finder's: it was prompted to **refute** the claim and to default to "this is not a bug" unless it could confirm a concrete trigger — specific inputs that produce a specific wrong output. It was also told to check whether an existing test already covered the case. A candidate only survived if the skeptic failed to refute it *and* confirmed no test already pinned the behavior.

**Synthesize (decide).** Surviving findings were collected, de-duplicated, and ranked. Only candidates that were both **confirmed** and **untested** became fixes. Everything else was logged and dropped.

## Why refute-by-default verification matters

Finder agents over-report. Reading code with a "find the bug" mandate produces a long tail of plausible-but-wrong findings: edge cases that can't actually be reached, "off-by-one" suspicions that are correct on a second read, behaviors that look wrong but are intentional. If you act on the raw finder output you spend your time writing fixes for non-bugs and, worse, you risk *introducing* regressions into correct code.

A single skeptical pass with the burden of proof flipped filters that tail cheaply. The asymmetry is the whole point: a *real* bug survives a demand for a concrete trigger, because the trigger exists. A *false alarm* doesn't. This is exactly the same logic as the rule that you must produce a failing regression test before you're allowed to write a fix — if you can't make it fail, you haven't found a bug, you've found a feeling. See [testing philosophy](/story/09/45_testing_philosophy).

## The cross-cutting finders earned their keep

The single most valuable design choice was assigning some agents to *dimensions* rather than files. The two highest-impact findings — the [row-alignment](/story/09/41_row_alignment) landmine and the look-ahead leakage trace — both spanned multiple files. The defect lived in the *seam* between modules: features computed in one place, labels computed in another, joined in a third. A finder reading any one of those files in isolation would have seen locally-correct code and moved on. The bug only becomes visible when one agent holds the whole data-flow in its head at once and asks "are these two frames actually aligned on the index I think they are?"

The lesson generalizes: file-scoped finders catch local bugs (a wrong comparison, an unguarded divide). Whole-class bugs — leakage, alignment, units, determinism — hide between files and need a finder whose mandate is the *property*, not the *location*. Budget for both.

## The reliability lesson we learned mid-audit

The first attempt at the fan-out failed completely, and the failure was instructive. We initially required every agent to emit findings as **machine-readable structured output** via a tool-call handshake; the orchestrator would parse the structured payload and merge it. Every single agent failed the handshake. The fan-out returned **zero usable findings**. Critically, this was a *framework* failure, not a result about the code: the agents may well have found real bugs, but the contract we imposed was too brittle to deliver any of it.

The recovery was to make the agent contract as simple as possible — plain prose-returning agents (there is no handshake to fail when the deliverable is just text), with the orchestrator doing the parsing. In parallel, the orchestrator **read the two highest-stakes files directly** (`run_pipeline.py` and `web/app.py`) rather than delegating them, on the theory that the cost of a silent bug in the pipeline entrypoint or the web surface is high enough to justify a direct read.

Three durable lessons:

- **Prefer the simplest reliable agent contract.** Prose in, prose out beats a clever structured protocol that fails silently.
- **Verify infrastructure at small scale before fanning out.** One test agent against the real handshake would have caught the failure before it cost a full parallel sweep.
- **Read the few highest-cost files yourself.** Don't fully delegate the places where a missed bug is most expensive.

## Self-verification before any edit

Subagents are wrong sometimes — confidently wrong, even after the refute pass. So the final gate before writing *any* fix was a direct re-read: the orchestrator opened the actual code for each confirmed finding and checked the claim against the source line by line. This caught several "scrutiny points" that turned out to be already-correct or already-covered by a test, which were then *not* fixed. The skeptic narrows the field; the direct read is the last line of defense before you touch the codebase.

## Every fix shipped with a failing-first regression test

No fix was committed without a regression test designed to **fail against the pre-fix code** — see [testing philosophy](/story/09/45_testing_philosophy) for the discipline in full. The test was written first, confirmed red on the old behavior, then the fix turned it green. After all eleven fixes landed, the full suite — 334 tests — ran clean, confirming zero regressions elsewhere in the pipeline.

## An honest note on cost

The failed structured-output attempt was not free: it burned roughly **400k tokens and produced nothing**. The prose-based retry plus the targeted orchestrator reads succeeded on the second pass. We document the dead end deliberately, so the next audit skips the fragile structured-handshake path and goes straight to the simple contract. The cheapest token is the one you don't spend re-learning a lesson that's already written down.

# Atomic writes, duplicate dates, and data integrity

Three bugs in this codebase shared a single failure mode: they let a transient event — a crash mid-write, a re-emitted bar, an empty array — quietly corrupt or crash persisted state. None threw a loud error at the obvious moment. Each silently poisoned downstream state, exactly the class of bug that's expensive precisely because it stays invisible until the numbers are already wrong. This deep-dive walks the three fixes and the discipline they enforce.

## 1. The lockbox was self-corrupting

The lockbox exists to guarantee one-shot claims survive across runs — it is the [trial registry](/story/09/16_trial_registry)'s durable backbone, and [validation modes](/story/09/21_validation_modes) lean on it to refuse a second claim on the same slot. So it was darkly funny that `claim_lockbox` persisted the registry like this:

```python
registry_path.write_text(json.dumps(registry, indent=2))
```

`write_text` is a truncate-then-write. It opens the file, zeroes it, then streams new bytes. If the process is killed (OOM, SIGKILL, a yanked cable) between the truncate and the final flush, you are left with a half-written or empty file. The exact data the lockbox exists to protect — every prior one-shot claim — is erased. The integrity guarantee defeats itself under the one condition it was built for.

The fix is the standard atomic-write pattern: write to a temp file in the *same directory*, then rename over the target.

```python
tmp = registry_path.with_name(registry_path.name + ".tmp")
tmp.write_text(json.dumps(reg, indent=2))
os.replace(tmp, registry_path)
```

`os.replace` is an atomic rename on POSIX: the target either points at the old inode or the new one, never a torn intermediate. Same-directory matters because rename is only atomic *within* a filesystem. This is the general principle for any file whose integrity matters — append-only logs, single-source-of-truth registries, config that other processes read concurrently: never truncate in place; stage and rename.

One honest caveat: this fixes torn writes, not the TOCTOU race between two concurrent claimers (both read, both write, last-writer-wins). That's a real race but out of scope for a single-user workflow, and it's now documented rather than silently assumed away (tracked in the [roadmap](/story/09/47_roadmap)).

## 2. A duplicate date shifts the entire index

`_normalize_one` in [data acquisition](/story/09/13_data_acquisition) trusted yfinance to return one row per date. It doesn't always: occasionally it re-emits a corrected bar for a date already present, yielding two rows for one calendar day.

That sounds cosmetic until you remember that **the entire pipeline indexes price series positionally by `anchor_idx`**. `labels.idx_map` maps date → integer position; features index by `ti`; the simulator builds a date-indexed series off the same offsets. One duplicate date inserts an extra row, and every subsequent position shifts by one. The date → index map is now silently wrong for that ticker, and the error propagates into the [36-label grid](/story/09/19_label_grid), into features, and into fills — with no exception, just numbers attached to the wrong day.

The fix is one line, but a load-bearing one:

```python
out = out.drop_duplicates(subset="date", keep="last")
```

`keep="last"` is deliberate: the later, corrected bar wins over the stale one.

This is the same lesson as the [trading calendar](/story/09/37_trading_calendar) and the [row-alignment landmine](/story/09/41_row_alignment): **positional indexing makes a clean date axis a hard invariant.** When position carries meaning, the index must be unique, sorted, and gap-checked — one dup or one gap is silent misalignment, not a crash. Contrast a date-keyed join, where pandas matches on the *value* and a duplicate surfaces as a visible NaN or a many-to-one explosion. The [SPY beta date alignment](/story/09/42_beta_zero_bug) bug was the same theme from the other direction.

## 3. Empty is a valid input

`embed_arrays` in [DINOv2](/story/09/01_dinov2_architecture) crashed on an empty input because `np.concatenate([])` raises rather than returning an empty array. Minor in isolation, but a 0-length batch is a legitimate state, and the function should return a correctly-shaped result:

```python
if n == 0:
    return np.empty((0, 384), dtype=np.float32)
```

The shape matters — `(0, 384)` still stacks and slices correctly downstream. This is the empty-input discipline the codebase now applies everywhere, the same reflex behind the [numerical stability](/story/09/44_numerical_stability) guards.

## Regression tests

Each fix is pinned by a test that fails on the old behavior: feed two rows for one date with different values, assert one row survives with the `keep="last"` value; call `claim_lockbox`, assert no `.tmp` file is left and prior claims persist; call `embed_arrays` on an empty array, assert the result is exactly shape `(0, 384)`.

## The lesson

Three small diffs, one principle: persisted state must be written atomically; positionally-indexed data demands a deduped, sorted, gap-checked index; and empty is a valid input everywhere, not a special case to crash on. These are the unglamorous invariants that keep a pipeline honest — the same hardening philosophy traced across [the hardening story](/story/09/39_hardening_story), and the reason [disaster recovery](/story/09/17_disaster_recovery) has something intact to recover *to*.

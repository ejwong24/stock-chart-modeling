# data_hashes.json — detecting yfinance drift

The pipeline's headline promise — [bit-identical reproduction](/story/09/12_reproducibility_seeds) — is a two-part claim, and seeds only cover one part. Seeds pin every source of randomness: the train/test split, the [LightGBM](/story/09/31_lightgbm_internals) column subsampling, the [DINOv2](/story/09/01_dinov2_architecture) dataloader shuffle. Re-run the pipeline with the same seeds and the *transformations* are deterministic. But determinism over a fixed input says nothing about whether the input itself is fixed. `f(x)` is reproducible; `f` applied to a silently-mutated `x` is not. And our `x` — the price history in `data/adjusted/*.parquet` — comes from yfinance, which is not a fixed input at all.

## Why hashes, not just seeds

yfinance is a view onto a live, restated dataset. Splits get back-applied, dividends get corrected weeks late, exchanges issue adjustment notices, and Yahoo quietly rewrites the adjusted-close series for a ticker months after the fact. None of this bumps a version number. A re-download in August can hand you a different `AAPL.parquet` than the one you trained on in May — same filename, same shape, different bytes. Seeds are blind to this: feed restated data through the same seeded RNG and you get a perfectly deterministic *wrong* answer, with no signal that anything moved. The only defense is to fingerprint the bytes you actually used.

That is `manifest.write_data_hashes`: it globs `*.parquet` under `data/adjusted/`, SHA-256s each file, and writes `{file_count, glob, hashes}` to `data/data_hashes.json`. The write is atomic — it serializes to a `.tmp` sibling and `os.replace`s it into place — so an interrupted run can never leave a half-written manifest that would later read as spurious drift. `sort_keys=True` keeps the JSON stable across runs so the committed manifest diffs cleanly in git.

## The changed / missing / added taxonomy

`verify_data_hashes` loads the committed manifest, re-hashes what's on disk, and sorts every filename into one of four buckets:

- **`changed`** — present in both manifest and disk, but the hash differs. This is the yfinance restatement signal, the thing the whole mechanism exists to catch. **Failure.**
- **`missing`** — in the manifest, gone from disk. Either a botched restore or a vanished ticker. **Failure.**
- **`added`** — on disk, not in the manifest. A *new* ticker you've pulled since the snapshot. **Not a failure.**
- **`ok`** — byte-identical.

The asymmetry is the whole point. `added` is informational because growing the universe is expected and harmless — it doesn't touch any byte you've already trained on. But `changed` means an input you already consumed has been rewritten underneath you, and `missing` means an input you depended on is gone. So `scripts/check_data_hashes.py --verify` exits `1` only on `changed or missing`, and merely prints the `added` count. A missing manifest is a distinct, louder failure: `verify_data_hashes` returns an `error` key and the script exits `2` — you can't verify against nothing.

## The workflow

1. After a pipeline run, `python scripts/check_data_hashes.py --write` snapshots the hashes.
2. **Commit `data/data_hashes.json` to git** alongside the run's results. The manifest is small and text; the parquet files are not committed.
3. On any later restore — a fresh box, a rebuilt data tree, a [disaster recovery](/story/09/17_disaster_recovery) drill — run `--verify`. Exit `0` means the data tree reproduces the manifest exactly and the seeds can do their job. Exit `1` means a name drifted, and you know *which* one before you trust a single downstream number.

## Why this bites the survivorship-aware universe hardest

Our [universe construction](/story/09/14_universe_construction) deliberately includes delisted names so the backtest isn't poisoned by survivorship bias. But delisted tickers are precisely the ones most prone to restatement: their final splits, terminal dividends, and corporate-action cleanup land *after* delisting, and Yahoo's housekeeping on dead symbols is erratic and late. The names that matter most for honest backtesting (see [survivorship accounting](/story/09/57_survivorship_accounting)) are the names whose bytes are least stable, so the [data acquisition](/story/09/13_data_acquisition) step is exactly where this manifest earns its keep.

## Regression coverage

`tests/test_roadmap_features.py` pins all of it: a clean write-then-verify returns empty `changed/missing/added`; a constructed drift scenario (rewrite `T0`, delete `T2`, add `T9`) returns `changed == ["T0.parquet"]`, `missing == ["T2.parquet"]`, `added == ["T9.parquet"]`; and verifying against a nonexistent path returns the `error` key. The taxonomy isn't just documented, it's enforced.

## The limitation

This mechanism **detects** drift; it does not **fix** it. Hashes tell you `BLDP.parquet` changed — they cannot hand you back the original bytes. For that you need an actual snapshot of the parquet tree: a NAS or S3 copy of the exact files behind the committed manifest, so a `changed` result becomes recoverable rather than merely diagnostic. That bytes-level snapshot is the next link in the chain — see the [hardening story](/story/09/39_hardening_story) and [roadmap](/story/09/47_roadmap). Until then, treat `data_hashes.json` as the smoke detector, not the sprinkler.

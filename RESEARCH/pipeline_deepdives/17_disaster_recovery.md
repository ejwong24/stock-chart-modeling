# Disaster recovery — restoring the pipeline from scratch

## Blast radius: what actually dies with the Oracle box

If the Oracle ARM64 instance hosting `/home/ubuntu/projects/stock_chart_modeling/` is destroyed, the damage is not uniform. The pipeline has three data tiers with wildly different recovery profiles.

**Tier 1 — Code & docs.** Everything under git. Source-of-truth is `https://github.com/ejwong24/stock-chart-modeling`. Recoverable in ~30 seconds with a `git clone`.

**Tier 2 — Raw data.** `data/adjusted/*.parquet` — roughly 6,500 parquets at ~50 KB each, totaling ~325 MB. Explicitly excluded via `.gitignore`. Recoverable by re-running `scripts/02_acquire_data.py`. Wall-clock: ~25–40 minutes.

**Tier 3 — Computed artifacts.** `reports/full/*`: ~150 MB of [DINOv2](/story/09/01_dinov2_architecture) embeddings, per-fold scores, blotters, equity curves. Not in git. Recoverable by re-running `scripts/run_pipeline.py` end-to-end. Wall-clock: ~6–10 hours.

**Tier 4 — Model objects.** Per-fold joblib pickles. Downstream of Tier 3.

Total cold-start time: **7–11 hours** of unattended compute.

## Full recovery procedure

```bash
# 1. System deps
sudo apt install python3.12-venv build-essential libffi-dev libopenblas-dev

# 2. Clone
cd /home/ubuntu/projects
git clone https://github.com/ejwong24/stock-chart-modeling
cd stock-chart-modeling

# 3. Venv + deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Re-acquire raw data (30 min)
python scripts/02_acquire_data.py

# 5. Re-run full pipeline (6-10 hr)
python scripts/run_pipeline.py

# 6. Start web server
bash web/start.sh --bg
```

## Verification: 276 green dots

After step 5 completes:

```bash
pytest tests/ --tb=short -q
```

Expected: **276/276 tests pass**. If even one fails, treat the restore as incomplete. The deterministic seed chain means a fresh run against identical raw data should produce **bit-identical blotters** to the original.

If determinism tests fail but functional tests pass, the most likely culprit is Tier 2 drift.

## "What if GitHub is also down?"

GitHub is not load-bearing. The Oracle box's `/home/ubuntu/` is on hourly snapshot, so `.git` itself is recoverable from snapshot. The recovery procedure works from any complete `.git` directory.

## Tailscale: re-issuing the serve rule

The [Tailscale serve](/story/09/15_deployment_tailscale) rule is per-instance and must be re-issued after recovery:

```bash
tailscale serve --bg --https=3344 http://localhost:3344
```

The Tailnet hostname (`openclaw.tail92a69b.ts.net`) persists across rebuilds — it's tied to the Tailscale account, not the VM. The HTTPS cert auto-renews.

## The yfinance data-drift problem

If recovery happens months after the original run, the same yfinance API calls will return **slightly different** historical data. Yahoo applies corporate-action corrections and silent fixes. That means the bit-identicality guarantee only holds if `data/adjusted/*.parquet` is byte-for-byte identical.

Two mitigations, neither currently in place:

- **Commit SHA-256 hashes** of every parquet to a `data_hashes.json`. The restore would flag drift loudly. (Future improvement.)
- **Store the parquets on a NAS or S3 bucket**. The only way to actually guarantee reproducibility months out.

## Minimum-viable recovery

If you can only recover the git repo:

- Serve the **web UI** — story chapters, deep-dives, methodology pages — all static HTML/Markdown.
- View **any committed reports** (currently none).
- **Re-run the pipeline from scratch**, accepting drift.

---

> **The one artifact that actually matters for reproducibility:** `data/adjusted/`. Code is in git. Reports can be regenerated. But `data/adjusted/` is the only tier where, once lost, you cannot get exactly the same thing back. If anything deserves a parallel backup, it's that tree.

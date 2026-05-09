"""Bulk-download daily OHLCV via yfinance for all tickers in the working universe."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import data_acq as da


def main():
    c = cfg.load()
    universe_path = cfg.project_path("data", "universe", "working_universe.csv")
    out_dir = cfg.project_path("data", "adjusted")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(universe_path)
    tickers = df["ticker"].tolist()

    start = c["universe"]["start_date"]
    end = c["universe"]["end_date"]

    t0 = time.time()
    summary = da.fetch_all(tickers, start, end, out_dir,
                           batch_size=50, sleep_between=0.6,
                           skip_existing=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["start"] = start
    summary["end"] = end
    print(json.dumps(summary, indent=2))

    with open(cfg.project_path("data", "adjusted", "_acquire_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

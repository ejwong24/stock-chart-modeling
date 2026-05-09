"""Build the working universe and persist it to data/universe/working_universe.csv."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import universe as uni


def main():
    c = cfg.load()
    out_dir = cfg.project_path("data", "universe")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching active universe from NASDAQ Trader...")
    active = uni.fetch_active_universe()
    print(f"  active tickers: {len(active)}")

    delisted = uni.load_delisted_seed(cfg.project_path(c["universe"]["delisted_seed_path"]))
    print(f"  delisted seed: {len(delisted)}")

    df = uni.build(active, delisted)
    df.to_csv(out_dir / "working_universe.csv", index=False)
    print(f"Wrote {len(df)} tickers -> {out_dir / 'working_universe.csv'}")
    print(f"  active : {df['active'].sum()}")
    print(f"  delisted: {(~df['active']).sum()}")


if __name__ == "__main__":
    main()

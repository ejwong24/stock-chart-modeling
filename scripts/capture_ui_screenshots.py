"""Take screenshots of the live web UI for embedding in the /story walkthrough."""
from __future__ import annotations
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "web" / "static" / "story"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:3344"

PAGES = [
    ("home", "/", 1100, 750, None),
    ("runs_full_realistic", "/runs/full_realistic", 1200, 950,
     "#equity-chart"),
    ("report_card_full", "/runs/full/report-card", 1100, 1100, None),
    ("research_index", "/research", 1100, 800, None),
    ("flaws", "/flaws", 1100, 900, None),
    ("glossary", "/glossary", 1100, 950, None),
]


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 800},
                                    device_scale_factor=1.6)
        for name, path, w, h, wait_selector in PAGES:
            page = ctx.new_page()
            page.set_viewport_size({"width": w, "height": h})
            url = BASE + path
            print(f"screenshotting {url} -> {name}.png")
            try:
                page.goto(url, wait_until="networkidle", timeout=20_000)
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=10_000)
                    except Exception:
                        pass
                # extra dwell for Plotly to finish rendering
                time.sleep(2.0)
                page.screenshot(path=str(OUT / f"ss_{name}.png"),
                                  full_page=False)
            except Exception as e:
                print(f"  failed: {e}")
            finally:
                page.close()
        browser.close()
    print("\nScreenshots in", OUT)


if __name__ == "__main__":
    sys.exit(main() or 0)

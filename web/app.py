"""FastAPI web UI for the stock chart modeling pipeline.

Run with:
    cd /home/ubuntu/projects/stock_chart_modeling
    source .venv/bin/activate
    python web/app.py

Then visit http://127.0.0.1:3340/ (or http://<your-tailscale-host>:3340/).
"""
from __future__ import annotations
import json, os, signal, subprocess, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
DATA_ADJUSTED = PROJECT / "data" / "adjusted"
LOGS_DIR = PROJECT / "web" / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(title="Stock Chart Modeling — Corrected Rebuild")
app.mount("/static", StaticFiles(directory=str(PROJECT / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT / "web" / "templates"))


# ----- Background-process tracking ----------------------------------------

class RunRegistry:
    def __init__(self) -> None:
        self.path = LOGS_DIR / "registry.json"

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def add(self, tag: str, pid: int, cmd: list[str], log: str) -> None:
        d = self._load()
        d[tag] = {"pid": pid, "cmd": cmd, "log": log,
                  "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._save(d)

    def list(self) -> dict:
        return self._load()

    def alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except Exception:
            return False


REG = RunRegistry()


# ----- Helpers ------------------------------------------------------------

def _run_dirs() -> list[Path]:
    if not REPORTS.exists():
        return []
    return sorted([p for p in REPORTS.iterdir() if p.is_dir() and (p / "headline.json").exists()])


def _run_summary(tag: str) -> dict:
    p = REPORTS / tag / "headline.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _equity_curve(tag: str, track: str) -> pd.DataFrame:
    p = REPORTS / tag / f"equity_{track}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _list_tracks(tag: str) -> list[str]:
    out = []
    for p in (REPORTS / tag).glob("equity_*.parquet"):
        out.append(p.stem.replace("equity_", ""))
    return sorted(out)


def _data_status() -> dict:
    universe_path = PROJECT / "data" / "universe" / "working_universe.csv"
    n_universe = 0
    if universe_path.exists():
        try:
            n_universe = sum(1 for _ in open(universe_path)) - 1
        except Exception:
            n_universe = 0
    n_parquets = len(list(DATA_ADJUSTED.glob("*.parquet"))) if DATA_ADJUSTED.exists() else 0
    summary_path = DATA_ADJUSTED / "_acquire_summary.json"
    acq = {}
    if summary_path.exists():
        try:
            acq = json.loads(summary_path.read_text())
        except Exception:
            acq = {}
    return {
        "universe_size": n_universe,
        "tickers_on_disk": n_parquets,
        "acquisition_summary": acq,
    }


def _flaws_md() -> str:
    p = PROJECT / "FLAWS_AND_FIXES.md"
    return p.read_text() if p.exists() else "(FLAWS_AND_FIXES.md missing)"


def _readme_md() -> str:
    p = PROJECT / "README.md"
    return p.read_text() if p.exists() else "(README.md missing)"


# ----- Routes -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    runs = []
    for d in _run_dirs():
        s = _run_summary(d.name)
        runs.append({
            "tag": d.name,
            "label": s.get("headline_label", "?"),
            "n_tickers": s.get("n_tickers_loaded", 0),
            "n_anchors": s.get("n_anchor_rows_after_prefilter", 0),
            "wall_seconds": s.get("wall_seconds", 0),
        })

    # Determine if there are any tracked active background runs
    active = []
    for tag, info in REG.list().items():
        if REG.alive(info["pid"]):
            active.append({"tag": tag, **info})

    return templates.TemplateResponse(request, "home.html", {
        "runs": runs,
        "data_status": _data_status(),
        "active_runs": active,
    })


@app.get("/runs", response_class=HTMLResponse)
def list_runs(request: Request):
    runs = []
    for d in _run_dirs():
        s = _run_summary(d.name)
        rb = s.get("random_baseline", {}) or {}
        best_track = max(s.get("summaries", []) or [{}],
                         key=lambda x: x.get("end_equity", 0))
        runs.append({
            "tag": d.name,
            "label": s.get("headline_label", "?"),
            "n_tickers": s.get("n_tickers_loaded", 0),
            "n_anchors": s.get("n_anchor_rows_after_prefilter", 0),
            "best_track": best_track.get("track", "?"),
            "best_equity": best_track.get("end_equity", 0),
            "best_cagr": best_track.get("cagr", 0),
            "random_median_cagr": rb.get("median_cagr", 0),
            "random_p95_eq": rb.get("p95_end_equity", 0),
            "wall_seconds": s.get("wall_seconds", 0),
        })
    return templates.TemplateResponse(request, "runs.html", {"runs": runs})


@app.get("/runs/new", response_class=HTMLResponse)
def new_run_form(request: Request):
    return templates.TemplateResponse(request, "run_new.html",
                                       {"data_status": _data_status()})


@app.get("/runs/{tag}", response_class=HTMLResponse)
def run_detail(request: Request, tag: str):
    if not (REPORTS / tag).exists():
        raise HTTPException(404, f"run {tag} not found")
    s = _run_summary(tag)
    tracks = _list_tracks(tag)
    return templates.TemplateResponse(request, "run_detail.html", {
        "tag": tag, "summary": s, "tracks": tracks,
    })


@app.get("/api/runs/{tag}/equity/{track}")
def api_equity(tag: str, track: str):
    df = _equity_curve(tag, track)
    if df.empty:
        return JSONResponse({"date": [], "equity": []})
    return JSONResponse({
        "date": df["date"].tolist(),
        "equity": df["equity"].astype(float).tolist(),
    })


@app.get("/api/runs/{tag}/blotter/{track}")
def api_blotter(tag: str, track: str):
    p = REPORTS / tag / f"blotter_{track}.parquet"
    if not p.exists():
        return JSONResponse([])
    df = pd.read_parquet(p)
    for c in ["entry_date", "exit_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return JSONResponse(df.head(500).to_dict(orient="records"))


@app.post("/runs/new")
def kick_off_run(
    horizon: int = Form(...),
    threshold: float = Form(...),
    n_tickers: int = Form(0),
    random_seeds: int = Form(100),
    out_tag: str = Form(...),
):
    out_tag = out_tag.strip().replace(" ", "_")
    if not out_tag.replace("_", "").replace("-", "").isalnum():
        return PlainTextResponse("out_tag must be alphanumeric/underscore/dash only",
                                  status_code=400)

    # Avoid clobbering existing
    if (REPORTS / out_tag).exists():
        return PlainTextResponse(
            f"A run named '{out_tag}' already exists. Pick a new tag.",
            status_code=400)

    venv_python = str(PROJECT / ".venv" / "bin" / "python")
    log_path = LOGS_DIR / f"{out_tag}.log"
    cmd = [
        venv_python,
        str(PROJECT / "scripts" / "run_pipeline.py"),
        "--horizon", str(horizon),
        "--threshold", str(threshold),
        "--n-tickers", str(n_tickers),
        "--random-seeds", str(random_seeds),
        "--out-tag", out_tag,
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                             cwd=str(PROJECT), preexec_fn=os.setsid)
    REG.add(out_tag, proc.pid, cmd, str(log_path))
    return RedirectResponse(f"/monitor/{out_tag}", status_code=303)


@app.get("/monitor/{tag}", response_class=HTMLResponse)
def monitor(request: Request, tag: str):
    info = REG.list().get(tag, {})
    return templates.TemplateResponse(request, "monitor.html", {
        "tag": tag, "info": info,
    })


@app.get("/api/monitor/{tag}")
def api_monitor(tag: str, lines: int = Query(80, ge=1, le=2000)):
    info = REG.list().get(tag, {})
    if not info:
        # try to find a log even if not registered
        return JSONResponse({"alive": False, "info": {}, "log": "(no registered run)"})
    log = ""
    try:
        if Path(info["log"]).exists():
            log = subprocess.check_output(["tail", "-n", str(lines), info["log"]],
                                           text=True, errors="replace")
    except Exception as e:
        log = f"(could not read log: {e})"
    headline_ready = (REPORTS / tag / "headline.json").exists()
    return JSONResponse({
        "alive": REG.alive(info["pid"]),
        "info": info,
        "log": log,
        "headline_ready": headline_ready,
    })


@app.post("/api/monitor/{tag}/stop")
def stop_run(tag: str):
    info = REG.list().get(tag, {})
    if not info:
        raise HTTPException(404, "not registered")
    pid = info["pid"]
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": True})


@app.get("/flaws", response_class=HTMLResponse)
def flaws(request: Request):
    return templates.TemplateResponse(request, "markdown.html", {
        "title": "Flaws & Fixes (12-Subagent Critique Synthesis)",
        "subtitle": "What was wrong with the original document and what this rebuild fixes",
        "markdown": _flaws_md(),
    })


@app.get("/readme", response_class=HTMLResponse)
def readme(request: Request):
    return templates.TemplateResponse(request, "markdown.html", {
        "title": "README — How this pipeline is built",
        "subtitle": "Layout, run instructions, hardware notes",
        "markdown": _readme_md(),
    })


@app.get("/data", response_class=HTMLResponse)
def data_status(request: Request):
    return templates.TemplateResponse(request, "data.html", {
        "status": _data_status(),
    })


@app.get("/glossary", response_class=HTMLResponse)
def glossary(request: Request):
    return templates.TemplateResponse(request, "glossary.html", {})


@app.get("/research", response_class=HTMLResponse)
def research_index(request: Request):
    rdir = PROJECT / "RESEARCH"
    docs = []
    if rdir.exists():
        for p in sorted(rdir.glob("*.md")):
            if p.name == "INDEX.md":
                continue
            docs.append({"slug": p.stem, "title": p.stem.replace("_", " ").title(),
                         "size_kb": p.stat().st_size / 1024})
    index_md = (rdir / "INDEX.md").read_text() if (rdir / "INDEX.md").exists() else ""
    return templates.TemplateResponse(request, "research_index.html",
                                       {"docs": docs, "index_md": index_md})


@app.get("/story", response_class=HTMLResponse)
def story_index(request: Request):
    """Guided walkthrough index — 8 chapters explaining the project."""
    return templates.TemplateResponse(request, "story_index.html", {})


@app.get("/story/{chapter}", response_class=HTMLResponse)
def story_chapter(request: Request, chapter: str):
    """Render a single story chapter."""
    valid = {"01_claim", "02_audit", "03_reproduction", "04_costs",
              "05_falsification", "06_statistics", "07_sma250",
              "08_bottom_line"}
    if chapter not in valid:
        raise HTTPException(404, f"chapter '{chapter}' not found")
    return templates.TemplateResponse(request, f"story/{chapter}.html", {})


@app.get("/research/{slug}", response_class=HTMLResponse)
def research_detail(request: Request, slug: str):
    p = PROJECT / "RESEARCH" / f"{slug}.md"
    if not p.exists():
        raise HTTPException(404, f"{slug} not found")
    return templates.TemplateResponse(request, "markdown.html", {
        "title": slug.replace("_", " ").title(),
        "subtitle": "12-subagent deep-dive",
        "markdown": p.read_text(),
    })


@app.get("/runs/{tag}/report-card", response_class=HTMLResponse)
def report_card_view(request: Request, tag: str):
    p = REPORTS / tag / "honest_report_card.md"
    if not p.exists():
        # try to auto-generate
        try:
            import sys as _sys
            _sys.path.insert(0, str(PROJECT / "src"))
            from stock_chart import report_card as _rc
            run_dir = REPORTS / tag
            if not (run_dir / "headline.json").exists():
                raise HTTPException(404, f"run {tag} has no headline.json")
            values = _rc.auto_fill_from_run(run_dir, PROJECT)
            md = _rc.render(values)
            p.write_text(md)
        except Exception as e:
            raise HTTPException(500, f"could not auto-generate: {e}")
    return templates.TemplateResponse(request, "markdown.html", {
        "title": f"Honest Report Card — {tag}",
        "subtitle": "Auto-generated; TODO fields are mandatory to fill before sharing",
        "markdown": p.read_text(),
    })


@app.get("/config", response_class=HTMLResponse)
def show_config(request: Request):
    p = PROJECT / "config" / "default.yaml"
    body = p.read_text() if p.exists() else "(no config)"
    manifest_path = PROJECT / "manifest.json"
    manifest = manifest_path.read_text() if manifest_path.exists() else "(no manifest)"
    return templates.TemplateResponse(request, "config.html", {
        "config_yaml": body, "manifest_json": manifest,
    })


@app.get("/health")
def health():
    return {"ok": True, "project": str(PROJECT)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=3344, reload=False,
                log_level="info")

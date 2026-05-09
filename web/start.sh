#!/usr/bin/env bash
# Start the stock-chart-modeling web UI on port 3340.
# Usage:
#   bash web/start.sh           # foreground
#   bash web/start.sh --bg      # background, log to /tmp/stockweb.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
PORT="${PORT:-3344}"
HOST="${HOST:-127.0.0.1}"
if [[ "${1:-}" == "--bg" ]]; then
  nohup python -m uvicorn web.app:app --host "$HOST" --port "$PORT" \
        > /tmp/stockweb.log 2>&1 &
  echo "Started PID $! · port $PORT · log /tmp/stockweb.log"
  echo "Open: http://127.0.0.1:$PORT/"
else
  exec python -m uvicorn web.app:app --host "$HOST" --port "$PORT"
fi

# Web UI deployment — FastAPI + Tailscale + the no-DNS trick

The stock-modeling pipeline ships with a small web UI for kicking off backtests, browsing runs, and watching live pipeline progress. Deploying it required answering a deceptively annoying question: *how do you make a localhost service reachable from a phone, over HTTPS, without messing with DNS, port-forwarding, or self-signed cert prompts?* The answer is a four-layer stack — FastAPI on localhost, a background uvicorn, Tailscale serve, and the Tailnet hostname — that takes about ten seconds to start and zero infrastructure to maintain.

## The full stack

The app itself lives in `web/app.py`. A request flows: **uvicorn → ASGI → FastAPI router → Jinja2 template → HTML response.** Static files in `web/static/` (PNG story visuals, CSS, JS) are served by FastAPI's `StaticFiles` mount.

The listener binds to **127.0.0.1:3344**. That's deliberately localhost-only — the service is invisible to the LAN and invisible to the public internet. Process management is a one-liner: `bash web/start.sh --bg` backgrounds uvicorn, writes the PID to `/tmp/stockweb.pid`, and tees stdout/stderr to `/tmp/stockweb.log`.

The trick that makes any of this useful is the next layer:

```
tailscale serve --bg --https=3344 http://localhost:3344
```

That tells the Tailscale daemon to terminate HTTPS on port 3344 of the Tailnet hostname and reverse-proxy plaintext to `localhost:3344`. The Tailnet hostname `openclaw.tail92a69b.ts.net` is auto-provisioned, and the certificate is a real Let's Encrypt cert issued through Tailscale's ACME integration. The URL `https://openclaw.tail92a69b.ts.net:3344/` works on the laptop browser and on the Pixel 9 Tailscale app with no setup beyond "be signed into the Tailnet."

## Why this beats opening port 3344 to the internet

- **No DNS to configure.** `tail92a69b.ts.net` is the Tailnet's MagicDNS suffix, handed out automatically.
- **No firewall rules.** Tailscale uses NAT traversal with DERP relays as fallback. No inbound public port is opened.
- **Real HTTPS.** Phones don't show "your connection is not private" prompts.
- **Access control = Tailscale ACL.** Only devices already enrolled in the Tailnet can resolve or reach the hostname. No app-level password is needed on any page.

## The HTTP/HTTPS gotcha

Tailscale serve listens on port 3344 for **HTTPS only**. If you accidentally `curl http://openclaw.tail92a69b.ts.net:3344` (no `s`), uvicorn answers — because it's plaintext HTTP — but Tailscale's TLS terminator on the same port responds with the dreaded `Client sent an HTTP request to an HTTPS server.` Always type `https://`.

## Reloads are manual

To keep memory low, uvicorn runs **without** `--reload`. That means editing `web/app.py` or any Jinja2 template does *not* hot-pick-up:

```
pkill -f uvicorn && bash web/start.sh --bg
```

Documented in `workspace/server-registry.json` under the `stock_chart_modeling` entry.

## Static assets and screen sizes

`web/static/` holds PNGs, one CSS file, and a small amount of vanilla JS. The CSS has mobile-first `@media` queries tuned for the Pixel 9's 412×892 viewport — sticky headers shrink, tables collapse to cards, run-status badges stack.

We regression-test two sizes with Playwright in `scripts/capture_ui_screenshots.py`:

- **Desktop:** 1920×1080
- **Mobile:** 412×892, same routes, captured against the Tailnet URL

## The /api/monitor/{tag} route

When the user kicks off a long backtest from `/runs/new`, they don't want to refresh. The route `/api/monitor/{tag}` polls for live progress and returns three fields: `alive`, `info`, `log`. If the tag is unknown, it returns a safe default rather than a 500.

---

> **Service inventory.** `workspace/server-registry.json` is the canonical list of every locally-running web service on the Oracle ARM64 box. If you add a server and forget to register it, it's invisible to Mission Control's Health → Servers card. Register first, deploy second.

from __future__ import annotations

import secrets
import threading
from typing import Any

from espresso_rl.application.admin_pipeline import AdminPipelineService


def create_admin_dashboard_app(service: AdminPipelineService, admin_token: str):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="EspressoRL Admin", docs_url=None, redoc_url=None, openapi_url=None)

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        token = _token_from_authorization(authorization)
        if token is None or not secrets.compare_digest(token, admin_token):
            raise HTTPException(status_code=401, detail="admin token required")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/status", dependencies=[Depends(require_admin)])
    def status() -> dict:
        return service.status().to_dict()

    @app.post("/api/mirror/run", dependencies=[Depends(require_admin)])
    def run_mirror(payload: dict[str, Any] | None = None) -> dict:
        return service.mirror_once(
            limit=_limit(payload, default=100, maximum=500),
            dry_run=_dry_run(payload),
            requested_by="dashboard",
        ).to_dict()

    @app.post("/api/validation/run", dependencies=[Depends(require_admin)])
    def run_validation(payload: dict[str, Any] | None = None) -> dict:
        return service.validate_once(
            limit=_limit(payload, default=100, maximum=500),
            dry_run=_dry_run(payload),
            requested_by="dashboard",
        ).to_dict()

    @app.post("/api/purge/run", dependencies=[Depends(require_admin)])
    def run_purge(payload: dict[str, Any] | None = None) -> dict:
        return service.purge_queue_once(
            dry_run=_dry_run(payload),
            requested_by="dashboard",
        ).to_dict()

    @app.post("/api/priors/run", dependencies=[Depends(require_admin)])
    def run_priors(payload: dict[str, Any] | None = None) -> dict:
        return service.generate_priors_once(
            limit=_limit(payload, default=5000, maximum=50_000),
            dry_run=_dry_run(payload),
            requested_by="dashboard",
        ).to_dict()

    @app.post("/api/priors/dry-run", dependencies=[Depends(require_admin)])
    def dry_run_priors(payload: dict[str, Any] | None = None) -> dict:
        return service.generate_priors_once(
            limit=_limit(payload, default=5000, maximum=50_000),
            dry_run=True,
            requested_by="dashboard",
        ).to_dict()

    return app


def start_admin_dashboard(
    service: AdminPipelineService,
    *,
    admin_token: str,
    host: str,
    port: int,
    stop_event: threading.Event,
) -> threading.Thread:
    import uvicorn

    app = create_admin_dashboard_app(service, admin_token)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    def run() -> None:
        server.run()

    def watch_stop() -> None:
        stop_event.wait()
        server.should_exit = True

    thread = threading.Thread(target=run, name="espresso-rl-admin-dashboard", daemon=True)
    watcher = threading.Thread(target=watch_stop, name="espresso-rl-admin-dashboard-stop", daemon=True)
    thread.start()
    watcher.start()
    return thread


def _token_from_authorization(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix) :].strip()
    return token or None


def _limit(payload: object, *, default: int, maximum: int) -> int:
    if isinstance(payload, dict):
        value = payload.get("limit", default)
    elif isinstance(payload, (int, float)):
        value = payload
    else:
        value = default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, parsed))


def _dry_run(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("dry_run") is True


_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EspressoRL Admin</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f6f7f8; color: #111; }
    main { max-width: 1080px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 26px; margin: 0 0 18px; }
    h2 { font-size: 18px; margin: 0 0 10px; }
    section { margin: 0 0 18px; padding: 16px; background: #fff; border: 1px solid #d9dee3; border-radius: 6px; }
    label { display: block; font-size: 13px; margin-bottom: 6px; color: #4b5563; }
    input { width: min(420px, 100%); padding: 10px; border: 1px solid #b8c0cc; border-radius: 4px; }
    button { margin: 6px 8px 6px 0; padding: 9px 12px; border: 1px solid #7d8794; border-radius: 4px; background: #fff; cursor: pointer; }
    button:hover { background: #eef2f5; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
    .metric { padding: 12px; border: 1px solid #d9dee3; border-radius: 4px; background: #fbfcfd; }
    .metric strong { display: block; font-size: 24px; }
    pre { overflow: auto; padding: 12px; background: #111827; color: #f9fafb; border-radius: 4px; }
    @media (prefers-color-scheme: dark) {
      body { background: #111827; color: #f9fafb; }
      section, button, input { background: #1f2937; color: #f9fafb; border-color: #374151; }
      .metric { background: #172033; border-color: #374151; }
    }
  </style>
</head>
<body>
  <main>
    <h1>EspressoRL Admin</h1>
    <section>
      <h2>Session</h2>
      <label for="token">Admin token</label>
      <input id="token" type="password" autocomplete="off">
      <button onclick="login()">Unlock</button>
      <button onclick="refreshStatus()">Refresh</button>
    </section>
    <section>
      <h2>Pipeline</h2>
      <button onclick="action('/api/mirror/run')">Run mirror once</button>
      <button onclick="action('/api/mirror/run', true)">Dry-run mirror</button>
      <button onclick="action('/api/validation/run', true)">Dry-run validation</button>
      <button onclick="action('/api/validation/run')">Run validation once</button>
      <button onclick="action('/api/priors/dry-run')">Dry-run priors</button>
      <button onclick="action('/api/priors/run')">Generate priors once</button>
      <button onclick="action('/api/purge/run', true)">Dry-run purge</button>
      <button onclick="action('/api/purge/run')">Purge retained queue rows</button>
    </section>
    <section>
      <h2>Status</h2>
      <div id="metrics" class="grid"></div>
    </section>
    <section>
      <h2>Latest Rejections</h2>
      <pre id="rejections">[]</pre>
    </section>
    <section>
      <h2>Last Action</h2>
      <pre id="output">{}</pre>
    </section>
  </main>
  <script>
    let adminToken = '';
    async function login() {
      adminToken = document.getElementById('token').value;
      document.getElementById('token').value = '';
      const response = await fetch('/api/status', {headers: authHeaders()});
      if (!response.ok) {
        document.getElementById('output').textContent = 'Invalid admin token';
        adminToken = '';
        return;
      }
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      renderStatus(data);
    }
    async function refreshStatus() {
      const response = await fetch('/api/status', {headers: authHeaders()});
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      if (!response.ok) return;
      renderStatus(data);
    }
    async function action(path, dryRun = false) {
      const response = await fetch(path, {
        method: 'POST',
        headers: {...authHeaders(), 'Content-Type': 'application/json'},
        body: JSON.stringify({dry_run: dryRun})
      });
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      await refreshStatus();
    }
    function authHeaders() {
      return adminToken ? {'Authorization': `Bearer ${adminToken}`} : {};
    }
    function renderStatus(data) {
      const raw = data.raw_upload_counts || {};
      const purge = data.local_raw_upload_purge_eligible_counts || {};
      const metrics = {
        mirrored: raw.mirrored || 0,
        rejected: raw.rejected || 0,
        validated: raw.validated || 0,
        purge_validated: purge.validated || 0,
        purge_rejected: purge.rejected || 0,
        validated_shots: data.validated_shot_count || 0,
        training_rows: data.training_row_count || 0,
        community_priors: data.community_prior_count || 0,
        abuse_events: data.abuse_event_count || 0
      };
      document.getElementById('metrics').innerHTML = Object.entries(metrics)
        .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
        .join('');
      document.getElementById('rejections').textContent = JSON.stringify(data.latest_rejections || [], null, 2);
    }
  </script>
</body>
</html>
"""

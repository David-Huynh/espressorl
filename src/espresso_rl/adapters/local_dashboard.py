from __future__ import annotations

import secrets
import threading
from typing import Any

from espresso_rl.application.local_data import LocalDataService
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService


DASHBOARD_MAX_BODY_BYTES = 64_000
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
}


def create_local_dashboard_app(
    service: LocalDataService,
    upload_maintenance: UploadQueueMaintenanceService,
    local_token: str,
):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse, PlainTextResponse

    app = FastAPI(title="EspressoRL Local", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def harden_dashboard(request, call_next):
        if _body_too_large(request.headers.get("content-length")):
            return PlainTextResponse("request body too large", status_code=413, headers=SECURITY_HEADERS)
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    def require_local(authorization: str | None = Header(default=None)) -> None:
        token = _token_from_authorization(authorization)
        if token is None or not secrets.compare_digest(token, local_token):
            raise HTTPException(status_code=401, detail="local dashboard token required")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/status", dependencies=[Depends(require_local)])
    def status(limit: int = 500) -> dict:
        return service.status(limit=limit).to_dict()

    @app.post("/api/purge-useless", dependencies=[Depends(require_local)])
    def purge_useless(payload: dict[str, Any] | None = None) -> dict:
        return service.purge_useless_shots(
            bean_context_id=_optional_string(payload, "bean_context_id"),
            grinder_context_id=_optional_string(payload, "grinder_context_id"),
            limit=_limit(payload, default=100, maximum=1000),
            dry_run=_dry_run(payload),
        ).to_dict()

    @app.post("/api/contexts/{bean_context_id}/reset", dependencies=[Depends(require_local)])
    def reset_context(bean_context_id: str, payload: dict[str, Any] | None = None) -> dict:
        return service.reset_optimizer_context(
            bean_context_id,
            grinder_context_id=_optional_string(payload, "grinder_context_id"),
            dry_run=_dry_run(payload),
        ).to_dict()

    @app.post("/api/contexts/{bean_context_id}/purge-useless", dependencies=[Depends(require_local)])
    def purge_context(bean_context_id: str, payload: dict[str, Any] | None = None) -> dict:
        return service.purge_useless_shots(
            bean_context_id=bean_context_id,
            grinder_context_id=_optional_string(payload, "grinder_context_id"),
            limit=_limit(payload, default=100, maximum=1000),
            dry_run=_dry_run(payload),
        ).to_dict()

    @app.post("/api/shots/{shot_id}/exclude", dependencies=[Depends(require_local)])
    def exclude_shot(shot_id: str, payload: dict[str, Any] | None = None) -> dict:
        return service.exclude_shot(shot_id, dry_run=_dry_run(payload)).to_dict()

    @app.delete("/api/shots/{shot_id}", dependencies=[Depends(require_local)])
    def delete_shot(shot_id: str, payload: dict[str, Any] | None = None) -> dict:
        return service.delete_shot(shot_id, dry_run=_dry_run(payload)).to_dict()

    @app.post("/api/upload/rejected/purge", dependencies=[Depends(require_local)])
    def purge_rejected_uploads(payload: dict[str, Any] | None = None) -> dict:
        result = upload_maintenance.purge_rejected(
            limit=_limit(payload, default=100, maximum=500),
            local_record_id=_optional_string(payload, "local_record_id"),
        )
        return {
            "action": "purge_rejected_upload_queue",
            "result": result.__dict__,
        }

    @app.post("/api/upload/rejected/requeue", dependencies=[Depends(require_local)])
    def requeue_rejected_uploads(payload: dict[str, Any] | None = None) -> dict:
        result = upload_maintenance.requeue_valid_rejected(
            limit=_limit(payload, default=25, maximum=500),
        )
        return {
            "action": "requeue_valid_rejected_uploads",
            "result": result.__dict__,
        }

    return app


def start_local_dashboard(
    service: LocalDataService,
    upload_maintenance: UploadQueueMaintenanceService,
    *,
    local_token: str,
    host: str,
    port: int,
    stop_event: threading.Event,
) -> threading.Thread:
    import uvicorn

    app = create_local_dashboard_app(service, upload_maintenance, local_token)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    def run() -> None:
        server.run()

    def watch_stop() -> None:
        stop_event.wait()
        server.should_exit = True

    thread = threading.Thread(target=run, name="espresso-rl-local-dashboard", daemon=True)
    watcher = threading.Thread(target=watch_stop, name="espresso-rl-local-dashboard-stop", daemon=True)
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


def _body_too_large(content_length: str | None) -> bool:
    if not content_length:
        return False
    try:
        parsed = int(content_length)
    except ValueError:
        return True
    return parsed > DASHBOARD_MAX_BODY_BYTES


def _limit(payload: object, *, default: int, maximum: int) -> int:
    if isinstance(payload, dict):
        value = payload.get("limit", default)
    else:
        value = default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, parsed))


def _dry_run(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("dry_run") is True


def _optional_string(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EspressoRL Local</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f7f8fa; color: #111827; }
    main { max-width: 1180px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 16px; font-size: 24px; }
    h2 { margin: 0 0 10px; font-size: 17px; }
    section { margin-bottom: 16px; padding: 14px; border: 1px solid #d8dee5; border-radius: 6px; background: #fff; }
    input, button, select { padding: 9px 10px; border: 1px solid #aeb8c4; border-radius: 4px; background: #fff; color: inherit; }
    button { cursor: pointer; margin: 4px 6px 4px 0; }
    button:hover { background: #eef2f6; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }
    .card { padding: 10px; border: 1px solid #d8dee5; border-radius: 4px; background: #fbfcfd; }
    .muted { color: #5f6b7a; font-size: 13px; }
    .pill { display: inline-block; margin: 2px 4px 2px 0; padding: 2px 7px; border-radius: 999px; background: #e8edf3; font-size: 12px; }
    .bad { background: #ffe4e6; }
    .good { background: #dcfce7; }
    .warn { background: #fef3c7; }
    pre { overflow: auto; padding: 12px; background: #111827; color: #f9fafb; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px; border-bottom: 1px solid #e5e9f0; text-align: left; vertical-align: top; }
    th { font-size: 11px; text-transform: uppercase; color: #5f6b7a; }
    @media (prefers-color-scheme: dark) {
      body { background: #111827; color: #f9fafb; }
      section, .card, input, button, select { background: #1f2937; border-color: #374151; }
      .muted, th { color: #a7b0bd; }
      .pill { background: #374151; }
      .bad { background: #7f1d1d; }
      .good { background: #14532d; }
      .warn { background: #713f12; }
    }
  </style>
</head>
<body>
  <main>
    <h1>EspressoRL Local Data</h1>
    <section>
      <h2>Session</h2>
      <input id="token" type="password" autocomplete="off" placeholder="Local dashboard token">
      <button onclick="login()">Unlock</button>
      <button onclick="refreshStatus()">Refresh</button>
    </section>
    <section>
      <h2>Context Actions</h2>
      <select id="contextSelect"></select>
      <button onclick="contextAction('reset', true)">Dry-run Reset BO</button>
      <button onclick="contextAction('reset', false)">Reset BO</button>
      <button onclick="contextAction('purge-useless', true)">Dry-run Purge Useless</button>
      <button onclick="contextAction('purge-useless', false)">Purge Useless</button>
      <button onclick="purgeAll(true)">Dry-run All Useless</button>
      <button onclick="purgeAll(false)">Purge All Useless</button>
      <p class="muted">Reset BO keeps history but excludes this context from local optimization. Purge useless deletes only shots already excluded from BO or classified as non-espresso.</p>
    </section>
    <section>
      <h2>Upload Queue</h2>
      <button onclick="uploadAction('/api/upload/rejected/requeue')">Retry Valid Rejected Uploads</button>
      <button onclick="uploadAction('/api/upload/rejected/purge')">Clean Rejected Upload Queue</button>
      <p class="muted">Cleaning rejected uploads may delete linked local shots only when those shots are already useless for local optimization.</p>
    </section>
    <section>
      <h2>Contexts</h2>
      <div id="contexts" class="grid"></div>
    </section>
    <section>
      <h2>Recent Shots</h2>
      <div id="shots"></div>
    </section>
    <section>
      <h2>Last Action</h2>
      <pre id="output">{}</pre>
    </section>
  </main>
  <script>
    let localToken = '';
    let currentStatus = null;
    async function login() {
      localToken = document.getElementById('token').value;
      document.getElementById('token').value = '';
      await refreshStatus();
    }
    function authHeaders() {
      return localToken ? {'Authorization': `Bearer ${localToken}`} : {};
    }
    async function refreshStatus() {
      const response = await fetch('/api/status?limit=1000', {headers: authHeaders()});
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      if (!response.ok) return;
      currentStatus = data;
      renderStatus(data);
    }
    async function postJson(path, body = {}) {
      const response = await fetch(path, {
        method: 'POST',
        headers: {...authHeaders(), 'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      await refreshStatus();
    }
    async function deleteShot(id, dryRun) {
      const response = await fetch(`/api/shots/${encodeURIComponent(id)}`, {
        method: 'DELETE',
        headers: {...authHeaders(), 'Content-Type': 'application/json'},
        body: JSON.stringify({dry_run: dryRun})
      });
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      await refreshStatus();
    }
    function contextAction(action, dryRun) {
      const raw = document.getElementById('contextSelect').value;
      if (!raw) return;
      const context = JSON.parse(raw);
      postJson(`/api/contexts/${encodeURIComponent(context.bean_context_id)}/${action}`, {
        dry_run: dryRun,
        grinder_context_id: context.grinder_context_id || null
      });
    }
    function purgeAll(dryRun) {
      postJson('/api/purge-useless', {dry_run: dryRun, limit: 1000});
    }
    function uploadAction(path) {
      postJson(path, {limit: 500});
    }
    function renderStatus(data) {
      const contexts = data.contexts || [];
      document.getElementById('contextSelect').innerHTML = contexts
        .filter(c => c.bean_context_id)
        .map(c => {
          const value = JSON.stringify({bean_context_id: c.bean_context_id, grinder_context_id: c.grinder_context_id || null});
          const label = `${c.bean_context_id} / ${c.grinder_context_id || 'No grinder'}`;
          return `<option value="${escapeAttr(value)}">${escapeHtml(label)} (${c.optimizer_shot_count}/${c.shot_count} BO)</option>`;
        })
        .join('');
      document.getElementById('contexts').innerHTML = contexts.map(c => `
        <div class="card">
          <strong>${escapeHtml(c.bean_context_id || 'No bean')}</strong>
          <div class="muted">${escapeHtml(c.grinder_context_id || 'No grinder')}</div>
          <div class="muted">latest ${formatTime(c.latest_shot_at)}</div>
          <span class="pill">shots ${c.shot_count}</span>
          <span class="pill ${c.optimizer_shot_count ? 'good' : 'warn'}">BO ${c.optimizer_shot_count}</span>
          <span class="pill">rated ${c.rated_shot_count}</span>
          <span class="pill ${c.rejected_upload_count ? 'bad' : ''}">rejected uploads ${c.rejected_upload_count}</span>
        </div>`).join('');
      const rows = (data.recent_shots || []).map(s => `
        <tr>
          <td>${formatTime(s.timestamp)}<br><span class="muted">${escapeHtml(s.shot_id)}</span></td>
          <td>${escapeHtml(s.bean_context_id || '')}<br><span class="muted">${escapeHtml(s.grinder_context_id || '')}</span></td>
          <td>${escapeHtml(s.shot_type || '')}<br>${num(s.shot_time_s)}s ${num(s.beverage_out_g)}g</td>
          <td><span class="pill ${s.included_in_optimizer ? 'good' : 'warn'}">${s.included_in_optimizer ? 'BO included' : 'not BO'}</span>
              ${s.rejected_upload ? '<span class="pill bad">rejected upload</span>' : ''}
              ${s.profile_flow_masked ? '<span class="pill warn">flow masked</span>' : ''}</td>
          <td>
            <button data-shot-action="exclude" data-shot-id="${escapeAttr(s.shot_id)}">Exclude</button>
            <button data-shot-action="delete-dry" data-shot-id="${escapeAttr(s.shot_id)}">Dry-run Delete</button>
            <button data-shot-action="delete" data-shot-id="${escapeAttr(s.shot_id)}">Delete</button>
          </td>
        </tr>`).join('');
      document.getElementById('shots').innerHTML = `<table><thead><tr><th>Shot</th><th>Context</th><th>Result</th><th>Use</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`;
      document.querySelectorAll('[data-shot-action]').forEach(button => {
        button.addEventListener('click', () => {
          const id = button.getAttribute('data-shot-id') || '';
          const action = button.getAttribute('data-shot-action');
          if (!id) return;
          if (action === 'exclude') {
            postJson(`/api/shots/${encodeURIComponent(id)}/exclude`);
          } else if (action === 'delete-dry') {
            deleteShot(id, true);
          } else if (action === 'delete') {
            deleteShot(id, false);
          }
        });
      });
    }
    function formatTime(value) {
      const ts = Number(value);
      if (!Number.isFinite(ts) || ts <= 0) return 'unknown';
      return new Date(ts * 1000).toLocaleString();
    }
    function num(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(1) : '-';
    }
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function escapeAttr(value) { return escapeHtml(value); }
  </script>
</body>
</html>
"""

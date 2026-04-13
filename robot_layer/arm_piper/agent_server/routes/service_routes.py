"""Service management endpoints."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from auth import require_admin
from services import ServiceManager

router = APIRouter(prefix="/services", tags=["services"])

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>🤖 Piper Robot Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 24px; }
  h1 { margin-bottom: 8px; }
  .subtitle { color: #888; margin-bottom: 20px; font-size: 14px; }
  .dry-run-badge { background: #ff9800; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #16213e; border-radius: 8px; overflow: hidden; }
  th, td { padding: 14px 18px; text-align: left; border-bottom: 1px solid #1a1a2e; }
  th { background: #0f3460; color: #aaa; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  tr:last-child td { border-bottom: none; }
  tr:hover { background: #1a2744; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
  .dot.on  { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
  .dot.off { background: #f44336; }
  button { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; color: #fff; font-weight: 500; transition: all 0.2s; }
  .btn-start { background: #2e7d32; }
  .btn-start:hover { background: #388e3c; }
  .btn-stop  { background: #c62828; }
  .btn-stop:hover { background: #d32f2f; }
  .btn-restart { background: #1565c0; margin-left: 8px; }
  .btn-restart:hover { background: #1976d2; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .log-box { background: #0d1117; padding: 14px; border-radius: 8px; margin-bottom: 20px;
             max-height: 250px; overflow-y: auto; font-family: 'SF Mono', Monaco, monospace; font-size: 12px;
             white-space: pre-wrap; color: #8b949e; border: 1px solid #30363d; }
  .status-text { font-size: 13px; }
  .status-text.on { color: #4caf50; }
  .status-text.off { color: #f44336; }
  .uptime { color: #888; font-size: 12px; }
  .actions { display: flex; gap: 8px; }
  .refresh-info { color: #666; font-size: 12px; margin-top: 16px; }
  .state-section { margin-bottom: 24px; }
  .state-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
  .state-card { background: #16213e; border-radius: 8px; padding: 16px; }
  .state-card h3 { font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
  .state-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a2e; }
  .state-row:last-child { border-bottom: none; }
  .state-label { color: #888; font-size: 13px; }
  .state-value { font-family: 'SF Mono', Monaco, monospace; font-size: 13px; color: #4caf50; }
  .state-value.disconnected { color: #f44336; }
  .control-section { margin-bottom: 24px; }
  .control-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .control-card { background: #16213e; border-radius: 8px; padding: 16px; }
  .control-card h3 { font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
  .control-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1a1a2e; }
  .control-row:last-child { border-bottom: none; }
  .control-label { color: #888; font-size: 13px; }
  .btn-home { background: #9c27b0; }
  .btn-home:hover { background: #ab47bc; }
  .btn-action { padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; color: #fff; font-weight: 500; transition: all 0.2s; width: 100%; margin-top: 8px; }
  .btn-action:disabled { opacity: .5; cursor: not-allowed; }
  .activity-badge { font-size: 12px; padding: 4px 10px; border-radius: 4px; font-weight: 500; }
  .activity-badge.idle { background: #1b5e20; color: #4caf50; }
  .activity-badge.executing { background: #0d47a1; color: #42a5f5; animation: pulse 1.5s infinite; }
  .activity-badge.resetting { background: #4a148c; color: #ce93d8; animation: pulse 1s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
</style></head><body>
<h1>🤖 Piper Robot Dashboard<span id="dry-run-badge" class="dry-run-badge" style="display:none">DRY-RUN</span></h1>
<p class="subtitle">Piper Robot Agent Server — Service Manager
  <span style="margin-left: 16px;">
    <a href="/docs/guide/html" target="_blank" style="color: #64b5f6; text-decoration: none; margin-right: 12px;">Getting Started ↗</a>
    <a href="/docs" target="_blank" style="color: #64b5f6; text-decoration: none; margin-right: 12px;">API Docs ↗</a>
    <a href="/code/sdk/html" target="_blank" style="color: #64b5f6; text-decoration: none;">SDK Reference ↗</a>
  </span>
</p>

<!-- Row 1: Robot State -->
<div class="state-section">
  <div class="state-grid">
    <div class="state-card">
      <h3>Arm Joints (rad)</h3>
      <div class="state-row"><span class="state-label">J1</span><span class="state-value" id="j0">—</span></div>
      <div class="state-row"><span class="state-label">J2</span><span class="state-value" id="j1">—</span></div>
      <div class="state-row"><span class="state-label">J3</span><span class="state-value" id="j2">—</span></div>
      <div class="state-row"><span class="state-label">J4</span><span class="state-value" id="j3">—</span></div>
      <div class="state-row"><span class="state-label">J5</span><span class="state-value" id="j4">—</span></div>
      <div class="state-row"><span class="state-label">J6</span><span class="state-value" id="j5">—</span></div>
    </div>
    <div class="state-card">
      <h3>End-Effector Pose</h3>
      <div class="state-row"><span class="state-label">X (m)</span><span class="state-value" id="ee-x">—</span></div>
      <div class="state-row"><span class="state-label">Y (m)</span><span class="state-value" id="ee-y">—</span></div>
      <div class="state-row"><span class="state-label">Z (m)</span><span class="state-value" id="ee-z">—</span></div>
      <div class="state-row"><span class="state-label">Qx</span><span class="state-value" id="ee-qx">—</span></div>
      <div class="state-row"><span class="state-label">Qy</span><span class="state-value" id="ee-qy">—</span></div>
      <div class="state-row"><span class="state-label">Qz</span><span class="state-value" id="ee-qz">—</span></div>
      <div class="state-row"><span class="state-label">Qw</span><span class="state-value" id="ee-qw">—</span></div>
    </div>
    <div class="state-card">
      <h3>Gripper</h3>
      <div class="state-row"><span class="state-label">Position (m)</span><span class="state-value" id="gripper-pos">—</span></div>
      <div style="margin-top: 10px; background: #0d1117; border-radius: 4px; height: 8px; overflow: hidden;">
        <div id="gripper-bar" style="height: 100%; width: 0%; background: #4caf50; border-radius: 4px; transition: width 0.5s;"></div>
      </div>
      <div style="margin-top: 4px; font-size: 11px; color: #666; display: flex; justify-content: space-between;">
        <span>0 (closed)</span><span>0.08 m (open)</span>
      </div>
    </div>
    <div class="state-card">
      <h3>Cameras</h3>
      <div class="state-row"><span class="state-label">cam_high</span><span class="state-value" id="cam-high">—</span></div>
      <div class="state-row"><span class="state-label">cam_low</span><span class="state-value" id="cam-low">—</span></div>
    </div>
    <div class="state-card">
      <h3>Lease</h3>
      <div class="state-row"><span class="state-label">Holder</span><span class="state-value" id="lease-holder" style="color: #888; font-style: italic;">(none)</span></div>
      <div class="state-row"><span class="state-label">Remaining</span><span class="state-value" id="lease-remaining">—</span></div>
      <div class="state-row"><span class="state-label">Queue</span><span class="state-value" id="lease-queue-len">0</span>&nbsp;waiting</div>
      <div class="state-row"><span class="state-label">Activity</span><span id="robot-activity" class="activity-badge idle" style="font-size: 11px; padding: 2px 8px;">Idle</span></div>
      <div style="margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap;">
        <button id="btn-pause-queue" style="background: #ff9800; color: #000; font-size: 11px; padding: 4px 10px; border: none; border-radius: 4px; cursor: pointer;" onclick="togglePauseQueue(this)">Pause Queue</button>
        <button id="btn-clear-queue" style="background: #b33; color:#fff; font-size: 11px; padding: 4px 10px; border: none; border-radius: 4px; cursor: pointer; display:none;" onclick="clearQueue(this)">Stop &amp; Reset</button>
      </div>
    </div>
  </div>
</div>

<!-- Row 2: Controls -->
<div class="control-section">
  <div class="control-grid">
    <div class="control-card">
      <h3>Reset Arm to Home</h3>
      <div class="control-row">
        <span class="control-label">Calls env.reset() on the arm</span>
      </div>
      <div class="control-row">
        <span class="control-label">Status</span>
        <span id="reset-status" class="state-value">Idle</span>
      </div>
      <button id="btn-reset-home" class="btn-action btn-home" onclick="resetToHome(this)">
        Reset to Home
      </button>
    </div>
    <div class="control-card">
      <h3>Code Execution <span id="code-status-badge" style="font-size: 11px; padding: 2px 8px; border-radius: 4px; display: none;"></span></h3>
      <div id="code-execution-grid" style="max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; white-space: pre-wrap; color: #8b949e;"></div>
    </div>
  </div>
</div>

<!-- Row 3: Logs -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
  <div class="control-card">
    <h3>Server Logs</h3>
    <div id="server-logs" class="log-box" style="height: 300px; font-size: 11px;"></div>
  </div>
  <div class="control-card">
    <h3>Service Logs</h3>
    <div id="service-logs-combined" class="log-box" style="height: 300px; font-size: 11px;"></div>
  </div>
</div>

<!-- Row 4: Services -->
<table>
  <thead><tr><th>Service</th><th>Status</th><th>PID</th><th>Uptime</th><th>Actions</th></tr></thead>
  <tbody id="tbl"><tr><td colspan="5" style="text-align:center;color:#666">Loading...</td></tr></tbody>
</table>

<!-- Port Reference -->
<table style="margin-top: 16px;">
  <thead><tr><th>Port</th><th>Service</th><th>Protocol</th><th>Bind</th></tr></thead>
  <tbody>
    <tr><td style="font-family: monospace;">8888</td><td>Piper Agent Server</td><td>HTTP / WebSocket</td><td>0.0.0.0</td></tr>
    <tr><td style="font-family: monospace;">11311</td><td>ROS Master</td><td>RPC</td><td>localhost</td></tr>
  </tbody>
</table>
<p class="refresh-info">Auto-refreshes every 2 seconds</p>
<script>
let serviceManagerEnabled = true;  // Replaced by server when disabled
let serviceKeys = [];
let queuePaused = false;

function fmt(s) {
  if (s == null) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return h + "h " + m + "m";
  return m + "m " + sec + "s";
}

async function act(method, url, btn) {
  if (btn) btn.disabled = true;
  try {
    const res = await fetch(url, { method });
    const data = await res.json();
    if (!res.ok) alert("Error: " + (data.detail || res.status));
    return data;
  } catch (e) {
    alert("Request failed: " + e);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function poll() {
  if (!serviceManagerEnabled) {
    document.getElementById("tbl").innerHTML =
      '<tr><td colspan="5" style="text-align:center;color:#666">Service manager not enabled</td></tr>';
    return;
  }
  try {
    const data = await (await fetch("/services")).json();
    serviceKeys = Object.keys(data);
    const rows = serviceKeys.map(name => {
      const s = data[name];
      const on = s.running;
      const dot = `<span class="dot ${on ? "on" : "off"}"></span>`;
      const status = `${dot}<span class="status-text ${on ? "on" : "off"}">${on ? "Running" : "Stopped"}</span>`;
      const pid = s.pid || "—";
      const uptime = s.uptime_s != null ? fmt(s.uptime_s) : "—";
      const btns = on
        ? `<div class="actions">
             <button class="btn-stop" onclick="act('POST','/services/${name}/stop',this).then(poll)">Stop</button>
             <button class="btn-restart" onclick="act('POST','/services/${name}/restart',this).then(poll)">Restart</button>
           </div>`
        : `<button class="btn-start" onclick="act('POST','/services/${name}/start',this).then(poll)">Start</button>`;
      return `<tr><td>${name}</td><td>${status}</td><td>${pid}</td><td class="uptime">${uptime}</td><td>${btns}</td></tr>`;
    }).join("");
    document.getElementById("tbl").innerHTML = rows || '<tr><td colspan="5" style="text-align:center;color:#666">No services configured</td></tr>';
  } catch (e) {
    document.getElementById("tbl").innerHTML =
      `<tr><td colspan="5" style="text-align:center;color:#f44">Error: ${e}</td></tr>`;
  }
}

async function pollState() {
  try {
    const data = await (await fetch("/state")).json();
    const joints = (data.arm && data.arm.joint_positions) || [];
    for (let i = 0; i < 6; i++) {
      const el = document.getElementById("j" + i);
      if (el) el.textContent = joints[i] != null ? joints[i].toFixed(3) : "—";
    }
    const ee = (data.arm && data.arm.end_pose) || [];
    ["ee-x","ee-y","ee-z","ee-qx","ee-qy","ee-qz","ee-qw"].forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.textContent = ee[i] != null ? ee[i].toFixed(4) : "—";
    });
    const gp = data.gripper && data.gripper.position != null ? data.gripper.position : null;
    const gEl = document.getElementById("gripper-pos");
    const gBar = document.getElementById("gripper-bar");
    if (gEl) gEl.textContent = gp != null ? gp.toFixed(4) + " m" : "—";
    if (gBar && gp != null) gBar.style.width = Math.min(100, Math.max(0, (gp / 0.08) * 100)) + "%";
    const cams = data.cameras || {};
    ["cam_high","cam_low"].forEach(name => {
      const el = document.getElementById(name.replace("_","-"));
      if (el) { el.textContent = cams[name] ? "✓" : "—"; el.style.color = cams[name] ? "#4caf50" : "#666"; }
    });
  } catch (e) { console.error("State poll error:", e); }
}

async function pollServerLogs() {
  try {
    const data = await (await fetch("/logs")).json();
    const el = document.getElementById("server-logs");
    if (!el) return;
    const logs = (data.logs || []).map(e => {
      const lvl = e.level || "INFO";
      const color = lvl === "ERROR" ? "#f44336" : lvl === "WARNING" ? "#ff9800" : "#8b949e";
      return `<span style="color:${color}">[${lvl}] ${e.message || e}</span>`;
    }).join("\n");
    el.innerHTML = logs;
    el.scrollTop = el.scrollHeight;
  } catch (e) { }
}

async function pollServiceLogs() {
  if (!serviceManagerEnabled || serviceKeys.length === 0) return;
  try {
    const allLogs = [];
    for (const name of serviceKeys) {
      const data = await (await fetch(`/services/${name}/logs?lines=20`)).json();
      (data.lines || []).forEach(l => allLogs.push(`[${name}] ${l}`));
    }
    const el = document.getElementById("service-logs-combined");
    if (el) { el.textContent = allLogs.slice(-100).join("\n"); el.scrollTop = el.scrollHeight; }
  } catch (e) { }
}

let lastCodeId = null;
async function pollCodeLogs() {
  try {
    const status = await (await fetch("/code/status")).json();
    const badge = document.getElementById("code-status-badge");
    const grid  = document.getElementById("code-execution-grid");
    if (!badge || !grid) return;
    const isRunning = status.is_running;
    badge.style.display = "inline";
    badge.textContent = isRunning ? "Running" : "Idle";
    badge.style.background = isRunning ? "#0d47a1" : "#1b5e20";
    badge.style.color = isRunning ? "#42a5f5" : "#4caf50";
    const actEl = document.getElementById("robot-activity");
    if (actEl) {
      actEl.textContent = isRunning ? "Executing Code" : "Idle";
      actEl.className = "activity-badge " + (isRunning ? "executing" : "idle");
    }
    if (status.execution_id) lastCodeId = status.execution_id;
    if (lastCodeId) {
      const stdout = status.stdout || "";
      const stderr = status.stderr || "";
      grid.innerHTML = stdout + (stderr ? '<span style="color:#f44">' + stderr + '</span>' : '');
      grid.scrollTop = grid.scrollHeight;
    }
  } catch (e) { }
}

async function pollLease() {
  try {
    const data = await (await fetch("/lease/status")).json();
    const holderEl = document.getElementById("lease-holder");
    if (holderEl) {
      holderEl.textContent = data.holder || "(none)";
      holderEl.style.fontStyle = data.holder ? "normal" : "italic";
      holderEl.style.color = data.holder ? "#4caf50" : "#888";
    }
    const remEl = document.getElementById("lease-remaining");
    if (remEl) {
      if (data.remaining_s != null && data.holder) {
        const m = Math.floor(data.remaining_s / 60);
        const s = Math.floor(data.remaining_s % 60);
        remEl.textContent = m > 0 ? m + "m " + s + "s" : s + "s";
      } else { remEl.textContent = "—"; }
    }
    const queueLen = data.queue_length || 0;
    const qEl = document.getElementById("lease-queue-len");
    if (qEl) qEl.textContent = queueLen;
    const clearBtn = document.getElementById("btn-clear-queue");
    if (clearBtn) clearBtn.style.display = (queueLen > 0 || data.holder) ? "" : "none";
    queuePaused = !!data.paused;
    const pauseBtn = document.getElementById("btn-pause-queue");
    if (pauseBtn) {
      pauseBtn.textContent = queuePaused ? "Resume Queue" : "Pause Queue";
      pauseBtn.style.background = queuePaused ? "#4caf50" : "#ff9800";
      pauseBtn.style.color = queuePaused ? "#fff" : "#000";
    }
  } catch (e) { }
}

async function resetToHome(btn) {
  if (!confirm("Reset arm to home position?")) return;
  btn.disabled = true;
  const statusEl = document.getElementById("reset-status");
  let leaseId = null;
  try {
    // 1. Acquire a lease
    statusEl.textContent = "Acquiring lease...";
    const acqRes = await fetch("/lease/acquire", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ holder: "dashboard-reset" }),
    });
    const acqData = await acqRes.json();
    if (!acqRes.ok) {
      statusEl.textContent = "Lease error: " + (acqData.detail || acqRes.status);
      return;
    }
    leaseId = acqData.lease_id;
    if (!leaseId) {
      // Queued — not supported in one-click reset
      statusEl.textContent = "Robot busy (lease held by: " + (acqData.holder || "another agent") + ")";
      return;
    }

    // 2. Execute env.reset()
    statusEl.textContent = "Resetting...";
    const execRes = await fetch("/code/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Lease-Id": leaseId },
      body: JSON.stringify({ code: "env.reset()", timeout: 30 }),
    });
    const execData = await execRes.json();
    if (!execRes.ok) {
      statusEl.textContent = "Execute error: " + (execData.detail || execRes.status);
      return;
    }

    // 3. Wait for completion
    statusEl.textContent = "Running...";
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const st = await (await fetch("/code/status")).json();
      if (!st.is_running) {
        statusEl.textContent = st.exit_code === 0 ? "Done ✓" : ("Failed (exit " + st.exit_code + ")");
        break;
      }
    }
  } catch (e) {
    statusEl.textContent = "Error: " + e;
  } finally {
    // 4. Release lease
    if (leaseId) {
      try {
        await fetch("/lease/release", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lease_id: leaseId }),
        });
      } catch (_) {}
    }
    btn.disabled = false;
    setTimeout(() => { statusEl.textContent = "Idle"; }, 4000);
  }
}

async function togglePauseQueue(btn) {
  const action = queuePaused ? "resume" : "pause";
  await act("POST", "/lease/" + action, btn);
  await pollLease();
}

async function clearQueue(btn) {
  if (!confirm("Stop current execution and reset lease?")) return;
  await act("POST", "/code/stop", btn);
  await act("POST", "/lease/release", btn);
  await pollLease();
}

poll();
pollState();
pollServerLogs();
pollServiceLogs();
pollCodeLogs();
pollLease();
setInterval(poll, 3000);
setInterval(pollState, 500);
setInterval(pollServerLogs, 2000);
setInterval(pollServiceLogs, 3000);
setInterval(pollCodeLogs, 1000);
setInterval(pollLease, 1000);
</script></body></html>"""


def create_router(service_mgr: ServiceManager | None, arm_monitor=None):
    """Create the service routes with injected dependencies.

    Args:
        service_mgr: ServiceManager instance, or None if service management is disabled.
                     When None, only the dashboard route is available (without service controls).
        arm_monitor: Ignored (kept for API compatibility).
    """
    service_manager_enabled = service_mgr is not None

    @router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False,
                dependencies=[Depends(require_admin)])
    async def dashboard(request: Request):
        """Web dashboard for service management."""
        # Inject the service_manager_enabled flag into the HTML
        html = DASHBOARD_HTML.replace(
            "let serviceManagerEnabled = true;",
            f"let serviceManagerEnabled = {'true' if service_manager_enabled else 'false'};"
        )
        # Inject API key into JS so fetch() calls include it
        api_key = request.query_params.get("api_key", "")
        auth_snippet = f"""<script>
var __apiKey = "{api_key}";
(function() {{
  var _origFetch = window.fetch;
  window.fetch = function(url, opts) {{
    if (__apiKey && typeof url === 'string' && url.startsWith('/')) {{
      opts = opts || {{}};
      opts.headers = opts.headers || {{}};
      if (opts.headers instanceof Headers) {{
        opts.headers.set('X-API-Key', __apiKey);
      }} else {{
        opts.headers['X-API-Key'] = __apiKey;
      }}
    }}
    return _origFetch.call(this, url, opts);
  }};
}})();
</script>"""
        html = html.replace("<script>", auth_snippet + "\n<script>", 1)
        return html

    @router.get("/config", include_in_schema=False,
                dependencies=[Depends(require_admin)])
    async def get_config():
        """Get dashboard configuration (service manager status, etc.)."""
        return {"service_manager_enabled": service_manager_enabled}

    # Only add service management routes if service manager is enabled
    if service_mgr is not None:
        @router.get("", include_in_schema=False,
                    dependencies=[Depends(require_admin)])
        async def list_services():
            """List all services with status, PID, uptime."""
            return service_mgr.get_status()

        @router.get("/{name}", include_in_schema=False,
                    dependencies=[Depends(require_admin)])
        async def get_service(name: str):
            """Get status of a specific service."""
            result = service_mgr.get_status(name)
            if "error" in result:
                return {"ok": False, **result}
            return result

        @router.post("/{name}/start", include_in_schema=False,
                     dependencies=[Depends(require_admin)])
        async def start_service(name: str):
            """Start a service."""
            result = await service_mgr.start_service(name)
            return result

        @router.post("/{name}/stop", include_in_schema=False,
                     dependencies=[Depends(require_admin)])
        async def stop_service(name: str):
            """Stop a service."""
            return await service_mgr.stop_service(name)

        @router.post("/{name}/restart", include_in_schema=False,
                     dependencies=[Depends(require_admin)])
        async def restart_service(name: str):
            """Restart a service."""
            result = await service_mgr.restart_service(name)
            return result

        @router.get("/{name}/logs", include_in_schema=False,
                    dependencies=[Depends(require_admin)])
        async def get_logs(name: str, lines: int = Query(default=50, ge=1, le=1000)):
            """Get recent log output for a service."""
            return service_mgr.get_logs(name, lines=lines)

    return router

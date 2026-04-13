"""Auto-generated system guide endpoint.

Introspects server config at runtime to generate accurate documentation
for frontend users (agents and humans). Admin-only details (queue ops,
rewind config, safety tuning) are deliberately excluded — see CLAUDE.md.
"""

from __future__ import annotations

import dataclasses
import html as html_mod
import logging
import os
import re

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute, APIWebSocketRoute

from config import LeaseConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/docs", tags=["docs"])


def _lease_field_descriptions() -> dict[str, str]:
    """Human-readable descriptions for LeaseConfig fields."""
    return {
        "max_duration_s": "Maximum lease duration before automatic revocation",
        "idle_timeout_s": "Seconds of inactivity before the lease is revoked",
        "reset_on_release": "Whether the robot returns to home when the lease ends",
    }


def _format_value(val: object) -> str:
    """Format a config value for display."""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _friendly_unit(field_name: str, val: object) -> str:
    """Append a human-friendly unit where appropriate."""
    formatted = _format_value(val)
    if field_name.endswith("_s") and isinstance(val, (int, float)):
        secs = int(val) if isinstance(val, float) and val == int(val) else val
        if secs >= 60 and secs % 60 == 0:
            return f"{formatted}s ({int(secs) // 60} min)"
        return f"{formatted}s"
    return formatted


def _collect_endpoints(app, prefixes: list[str]) -> list[dict]:
    """Collect endpoints from the app matching any of the given path prefixes.

    Skips routes hidden from the schema (admin-only) and the guide itself.
    """
    endpoints = []
    for route in app.routes:
        if isinstance(route, APIWebSocketRoute):
            path = route.path
            if any(path.startswith(p) for p in prefixes):
                summary = ""
                if route.endpoint and route.endpoint.__doc__:
                    summary = route.endpoint.__doc__.strip().split("\n")[0]
                endpoints.append({
                    "method": "WS",
                    "path": path,
                    "description": summary,
                })
            continue
        if not isinstance(route, APIRoute):
            continue
        if not getattr(route, "include_in_schema", True):
            continue
        path = route.path
        if path.startswith("/docs"):
            continue  # skip the guide itself
        if not any(path.startswith(p) for p in prefixes):
            continue
        method = next(iter(route.methods)) if route.methods else "GET"
        summary = route.summary or ""
        if not summary and route.endpoint and route.endpoint.__doc__:
            summary = route.endpoint.__doc__.strip().split("\n")[0]
        endpoints.append({
            "method": method,
            "path": path,
            "description": summary,
        })
    return endpoints


def generate_guide(app=None) -> dict:
    """Generate the system guide by introspecting live config."""
    lease_cfg = LeaseConfig()
    descriptions = _lease_field_descriptions()

    lease_fields = {}
    for f in dataclasses.fields(lease_cfg):
        if f.name in ("check_interval_s", "ticket_ttl_s"):
            continue  # internal implementation details
        val = getattr(lease_cfg, f.name)
        lease_fields[f.name] = {
            "value": val,
            "display": _friendly_unit(f.name, val),
            "description": descriptions.get(f.name, ""),
        }

    # Discover available SDK modules (check file existence, no import needed)
    sdk_modules = []
    _sdk_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_sdk")
    if os.path.exists(os.path.join(_sdk_dir, "piper_sdk.py")):
        sdk_modules.append("env (PiperRobotEnv) — robot arm & gripper control, /usr/bin/python3")
    if os.path.exists(os.path.join(_sdk_dir, "yolo_sdk.py")):
        sdk_modules.append("yolo (YoloSDK) — object detection via HTTP API + ROS 3D projection")
    if os.path.exists(os.path.join(_sdk_dir, "grasp_sdk.py")):
        sdk_modules.append("grasp (GraspSDK) — grasp pose generation via HTTP API")
    if os.path.exists(os.path.join(_sdk_dir, "memory_sdk.py")):
        sdk_modules.append("memory (MemorySDK) — object insert/query via Spatial Memory Hub HTTP API")

    return {
        "title": "Piper Robot Agent Server — Getting Started Guide",
        "sections": {
            "authentication": {
                "title": "Authentication",
                "description": (
                    "Remote clients must authenticate with an API key. "
                    "Localhost requests need no key."
                ),
                "how": [
                    "Pass `X-API-Key: <your-key>` as a header on every request, "
                    "or append `?api_key=<your-key>` as a query parameter.",
                    "Get your API key from the robot operator or ROBOT.md.",
                    "Requests from localhost (127.0.0.1) bypass authentication.",
                ],
            },
            "lease": {
                "title": "Lease System",
                "description": (
                    "The robot is a shared resource — only one agent or human "
                    "controls it at a time. You need a lease to send commands."
                ),
                "config": lease_fields,
                "flow": [
                    'POST /lease/acquire with body {"holder": "your-name"} — if free you get a lease_id immediately; if busy you get a ticket_id',
                    "If queued, poll GET /lease/queue/{ticket_id} until status is 'granted'",
                    "Pass your lease_id in the X-Lease-Id header for code execution and command requests",
                    'Release with POST /lease/release {"lease_id": "..."} (or let it expire)',
                    "Robot arm automatically resets to home position on lease release",
                    "Next agent in queue gets the lease",
                ],
                "endpoints": [
                    {
                        "method": "POST",
                        "path": "/lease/acquire",
                        "description": "Acquire or queue for control lease (never blocks)",
                        "body": '{"holder": "my-agent"}',
                        "note": "Returns lease_id immediately if the robot is free, "
                                "or ticket_id if another agent holds the lease.",
                    },
                    {
                        "method": "GET",
                        "path": "/lease/queue/{ticket_id}",
                        "description": "Check ticket status and queue position",
                        "body": None,
                    },
                    {
                        "method": "DELETE",
                        "path": "/lease/queue/{ticket_id}",
                        "description": "Cancel ticket (leave queue)",
                        "body": None,
                    },
                    {
                        "method": "POST",
                        "path": "/lease/release",
                        "description": "Release your lease",
                        "body": '{"lease_id": "..."}',
                    },
                    {
                        "method": "POST",
                        "path": "/lease/extend",
                        "description": "Reset idle timeout",
                        "body": '{"lease_id": "..."}',
                    },
                    {
                        "method": "GET",
                        "path": "/lease/status",
                        "description": "Current holder, remaining time, queue position",
                        "body": None,
                    },
                ],
                "queue_note": (
                    "POST /lease/acquire never blocks. If the robot is busy, you "
                    "receive a ticket_id. Poll GET /lease/queue/{ticket_id} to "
                    "check your position. When it's your turn, the response "
                    "includes your lease_id."
                ),
                "auto_rewind_note": (
                    "When your lease ends, the robot arm moves to the home position (env.reset()). "
                    "Make sure the arm is in a safe pose before releasing the lease."
                ),
                "idle_tip": (
                    "The idle timeout resets on every API call. If you need time "
                    "between calls (e.g., processing results), call "
                    "`POST /lease/extend` with `{\"lease_id\": \"...\"}` to reset "
                    "the timer. Alternatively, batch your work into a single "
                    "code execution to avoid gaps."
                ),
                "api_docs_note": (
                    "For full request/response schemas for all endpoints, "
                    "see the interactive API reference at [`/docs`](/docs)."
                ),
            },
            "code_execution": {
                "title": "Code Execution",
                "description": (
                    "Control the robot by submitting Python code. The code runs "
                    "under /usr/bin/python3 (ROS/MoveIt) with four pre-created instances: "
                    "`env` (PiperRobotEnv, robot arm & gripper control), "
                    "`yolo` (YoloSDK, object detection via HTTP API + ROS 3D projection), "
                    "`grasp` (GraspSDK, grasp pose generation via HTTP API), "
                    "`memory` (MemorySDK, object insert/query via Spatial Memory Hub HTTP API)."
                ),
                "lease_header_note": (
                    "Pass `X-Lease-Id: <your-lease-id>` as a header on "
                    "`/code/execute` and `/code/stop`. The lease ID is returned "
                    "when you acquire a lease."
                ),
                "submit": {
                    "method": "POST",
                    "path": "/code/execute",
                    "example_curl": (
                        'curl -X POST http://<ROBOT_IP>:8888/code/execute \\\n'
                        '  -H "X-API-Key: <your-key>" \\\n'
                        '  -H "X-Lease-Id: <your-lease-id>" \\\n'
                        '  -H "Content-Type: application/json" \\\n'
                        '  -d \'{"code": "env.reset()", "timeout": 60}\''
                    ),
                    "body": '{"code": "env.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", "timeout": 60}',
                },
                "validate": {
                    "method": "POST",
                    "path": "/code/validate",
                    "body": '{"code": "env.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])"}',
                    "description": (
                        "Validate code without executing it. Checks for dangerous "
                        "patterns (shell commands, network access, file deletion). "
                        "No lease required."
                    ),
                },
                "sdk_modules": sdk_modules,
                "sdk_reference": "/code/sdk/markdown",
                "check_status": {"method": "GET", "path": "/code/status"},
                "get_result": {"method": "GET", "path": "/code/result"},
                "behaviors": [
                    "The wrapper pre-creates `env`, `yolo`, `grasp`, `memory` — no import needed",
                    "env runs in /usr/bin/python3 with ROS/MoveIt; yolo and grasp call remote HTTP APIs",
                    "memory calls Spatial Memory Hub HTTP API to upsert and query object memories",
                    "All hardware drivers (arm, cameras, ROS) must be running externally",
                    "Code runs synchronously — exceptions stop execution",
                    "print() output is captured in the result",
                    "Piper arm has 6 joints; gripper range is 0 (closed) to 0.08 m (open)",
                    "Available cameras: cam_high (overhead), cam_low (wrist) — see config.yaml",
                    "Grasp: move only to translation_base_retreat + quaternion_base; do not move to translation_base directly",
                ],
            },
            "monitoring": {
                "title": "Monitoring During Execution",
                "description": (
                    "While code is running (GET /code/status returns running), "
                    "actively monitor the robot instead of waiting blindly."
                ),
                "tips": [
                    {
                        "name": "Terminal output (preferred)",
                        "description": (
                            "Poll `GET /code/status?stdout_offset=N&stderr_offset=N` "
                            "for incremental output. Use the returned offsets as "
                            "the next values. This is cheap — text only, no images."
                        ),
                    },
                    {
                        "name": "Execution recordings (preferred over live camera)",
                        "description": (
                            "Camera frames (0.5 Hz) and robot state (10 Hz) are "
                            "automatically recorded during every code execution. "
                            "`GET /code/recordings/{execution_id}` returns a "
                            "`timeline` array where each entry has a `frame` "
                            "filename, `timestamp`, and `state` (the nearest "
                            "state sample matched by time — arm joints, EE pose, "
                            "base pose, gripper). Retrieve frame images via "
                            "`GET /code/recordings/{execution_id}/frames/{filename}`. "
                            "This avoids vision token cost from live polling."
                        ),
                    },
                    {
                        "name": "Live camera (use sparingly)",
                        "description": (
                            "`GET /cameras/{id}/frame` returns a live JPEG. "
                            "Each frame costs vision tokens. Only use when you "
                            "need a live view outside of code execution — "
                            "recorded frames are preferred during/after execution."
                        ),
                    },
                    {
                        "name": "Robot state",
                        "description": (
                            "Poll `GET /state` for arm joint positions, end-effector pose, "
                            "and gripper status. Lightweight and cheap."
                        ),
                    },
                ],
                "note": (
                    "If something looks wrong, stop execution with "
                    "`POST /code/stop`. "
                    "Prefer adding print() statements to your code for "
                    "debugging. Use recorded frames after execution to verify "
                    "results instead of polling live camera feeds."
                ),
            },
            "state_observation": {
                "title": "State & Observation",
                "description": "Read robot state and camera feeds. No lease required.",
                "endpoints": _collect_endpoints(app, [
                    "/state", "/health", "/cameras",
                    "/ws/state", "/ws/cameras",
                    "/code/recordings",
                ]) if app else [
                    {"method": "GET", "path": "/state", "description": "Robot state (arm joints, end-effector pose, gripper)"},
                    {"method": "GET", "path": "/health", "description": "Server health status"},
                    {"method": "GET", "path": "/cameras", "description": "List available cameras"},
                    {"method": "GET", "path": "/cameras/{device_id}/frame", "description": "Get a camera frame (JPEG)"},
                    {"method": "WS", "path": "/ws/state", "description": "Streaming robot state"},
                    {"method": "WS", "path": "/ws/cameras", "description": "Streaming camera feeds"},
                    {"method": "GET", "path": "/code/recordings", "description": "List execution IDs with recordings"},
                    {"method": "GET", "path": "/code/recordings/{execution_id}", "description": "Recording timeline: frames matched with nearest state by timestamp"},
                    {"method": "GET", "path": "/code/recordings/{execution_id}/frames/{filename}", "description": "Retrieve a recorded JPEG frame"},
                ],
            },
            "external_services": {
                "title": "External GPU Services",
                "description": (
                    "The `yolo` and `grasp` SDK modules call remote HTTP APIs "
                    "running on GPU compute nodes. Configure their URLs in "
                    "`robot_sdk/config.yaml` under the `yolo.url` and `grasp.url` keys."
                ),
                "flow": [
                    "Start the YOLO detection service on the compute node (see its README)",
                    "Start the AnyGrasp service on the compute node (see its README)",
                    "Update `robot_sdk/config.yaml`: set `yolo.url` and `grasp.url` to the correct addresses",
                    "Restart the agent server — `yolo` and `grasp` will connect automatically",
                ],
                "env_vars": [
                    {
                        "name": "yolo.url (config.yaml)",
                        "description": "URL of the YOLO detection service (e.g. http://10.0.0.5:8017/detect)",
                    },
                    {
                        "name": "grasp.url (config.yaml)",
                        "description": "URL of the AnyGrasp service (e.g. http://10.0.0.5:8018/detect)",
                    },
                ],
                "note": (
                    "Both services accept a base64-encoded image and return detection results. "
                    "3D projection (pixel → camera → robot base frame) is performed locally "
                    "by the SDK using ROS depth images and camera_info topics."
                ),
            },
        },
    }


def _render_markdown(guide: dict) -> str:
    """Render the guide dict as markdown."""
    md = f"# {guide['title']}\n\n"

    # Authentication section
    if "authentication" in guide["sections"]:
        auth = guide["sections"]["authentication"]
        md += f"## {auth['title']}\n\n"
        md += f"{auth['description']}\n\n"
        for item in auth["how"]:
            md += f"- {item}\n"
        md += "\n"

    # Lease section
    lease = guide["sections"]["lease"]
    md += f"## {lease['title']}\n\n"
    md += f"{lease['description']}\n\n"

    md += "### Configuration\n\n"
    md += "| Setting | Value | Description |\n"
    md += "|---------|-------|-------------|\n"
    for name, info in lease["config"].items():
        md += f"| `{name}` | {info['display']} | {info['description']} |\n"
    md += "\n"

    md += "### How It Works\n\n"
    for i, step in enumerate(lease["flow"], 1):
        md += f"{i}. {step}\n"
    md += "\n"

    if "queue_note" in lease:
        md += f"> {lease['queue_note']}\n\n"
    md += f"> {lease['auto_rewind_note']}\n\n"
    if "idle_tip" in lease:
        md += f"> {lease['idle_tip']}\n\n"
    if "api_docs_note" in lease:
        md += f"> {lease['api_docs_note']}\n\n"

    md += "### Endpoints\n\n"
    md += "| Method | Path | Description |\n"
    md += "|--------|------|-------------|\n"
    for ep in lease["endpoints"]:
        md += f"| `{ep['method']}` | `{ep['path']}` | {ep['description']} |\n"
    md += "\n"

    md += "When a lease is granted (via `/lease/acquire` or `/lease/queue/{ticket_id}`), "
    md += "the response includes `max_duration_s` and `idle_timeout_s` so you know your time limits.\n\n"

    # Code execution section
    code = guide["sections"]["code_execution"]
    md += f"## {code['title']}\n\n"
    md += f"{code['description']}\n\n"

    if "lease_header_note" in code:
        md += f"> {code['lease_header_note']}\n\n"

    md += "### Submit Code\n\n"
    md += f"**`{code['submit']['method']} {code['submit']['path']}`**\n\n"
    if "example_curl" in code["submit"]:
        md += "```bash\n"
        md += code["submit"]["example_curl"]
        md += "\n```\n\n"
    md += "```json\n"
    md += code["submit"]["body"].replace("\\n", "\n")
    md += "\n```\n\n"

    if "validate" in code:
        v = code["validate"]
        md += "### Validate Code\n\n"
        md += f"{v['description']}\n\n"
        md += f"**`{v['method']} {v['path']}`**\n\n"
        md += "```json\n"
        md += v["body"].replace("\\n", "\n")
        md += "\n```\n\n"

    md += "### Available SDK Modules\n\n"
    for mod in code["sdk_modules"]:
        md += f"- `{mod}`\n"
    md += f"\nFull SDK reference: [`{code['sdk_reference']}`]({code['sdk_reference']})\n\n"

    md += "### Key Behaviors\n\n"
    for behavior in code["behaviors"]:
        md += f"- {behavior}\n"
    md += "\n"

    md += "### Check Results\n\n"
    md += f"- `{code['check_status']['method']} {code['check_status']['path']}` — Live execution status with real-time stdout/stderr. Pass `?stdout_offset=N&stderr_offset=N` (from previous response) to get only new output since last poll.\n"
    md += f"- `{code['get_result']['method']} {code['get_result']['path']}` — Final result with full stdout, stderr, exit code (after execution completes)\n\n"

    # Monitoring section
    if "monitoring" in guide["sections"]:
        mon = guide["sections"]["monitoring"]
        md += f"## {mon['title']}\n\n"
        md += f"{mon['description']}\n\n"
        for tip in mon["tips"]:
            md += f"- **{tip['name']}:** {tip['description']}\n"
        md += "\n"
        md += f"> {mon['note']}\n\n"

    # State & Observation section
    state = guide["sections"]["state_observation"]
    md += f"## {state['title']}\n\n"
    md += f"{state['description']}\n\n"

    md += "| Method | Path | Description |\n"
    md += "|--------|------|-------------|\n"
    for ep in state["endpoints"]:
        md += f"| `{ep['method']}` | `{ep['path']}` | {ep['description']} |\n"
    md += "\n"

    # External services section
    if "external_services" in guide["sections"]:
        ext = guide["sections"]["external_services"]
        md += f"## {ext['title']}\n\n"
        md += f"{ext['description']}\n\n"

        md += "### How to Deploy\n\n"
        for i, step in enumerate(ext["flow"], 1):
            md += f"{i}. {step}\n"
        md += "\n"

        if ext.get("env_vars"):
            md += "### Environment Variables\n\n"
            md += "| Variable | Description |\n"
            md += "|----------|-------------|\n"
            for ev in ext["env_vars"]:
                md += f"| `{ev['name']}` | {ev['description']} |\n"
            md += "\n"

        md += f"> {ext['note']}\n\n"

    # Links
    md += "## More Documentation\n\n"
    md += "- [`/code/sdk/markdown`](/code/sdk/markdown) — Full SDK reference (env / yolo / grasp / memory)\n"
    md += "- [`/docs`](/docs) — Interactive API reference (Swagger UI)\n"
    md += "- `robot_sdk/config.yaml` — Robot configuration (camera topics, ROS frames, API URLs)\n"

    return md


def _md_to_html(raw_md: str) -> str:
    """Convert markdown to HTML (zero-dependency, same pattern as sdk_docs.py)."""
    lines = raw_md.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_list = False
    in_ordered_list = False
    in_table = False
    table_header_done = False

    for line in lines:
        if line.startswith("```"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if in_ordered_list:
                html_lines.append("</ol>")
                in_ordered_list = False
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False
                table_header_done = False
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                lang = line[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html_mod.escape(line))
            continue

        stripped = line.strip()

        # Table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            # Skip separator rows
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not in_table:
                html_lines.append('<table><thead><tr>')
                for cell in cells:
                    cell = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell)
                    cell = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", cell)
                    html_lines.append(f"<th>{cell}</th>")
                html_lines.append("</tr></thead><tbody>")
                in_table = True
                table_header_done = True
                continue
            else:
                html_lines.append("<tr>")
                for cell in cells:
                    cell = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell)
                    cell = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", cell)
                    cell = re.sub(
                        r"\[([^\]]+)\]\(([^)]+)\)",
                        r'<a href="\2">\1</a>',
                        cell,
                    )
                    html_lines.append(f"<td>{cell}</td>")
                html_lines.append("</tr>")
                continue

        if in_table and not stripped.startswith("|"):
            html_lines.append("</tbody></table>")
            in_table = False
            table_header_done = False

        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if in_ordered_list:
                html_lines.append("</ol>")
                in_ordered_list = False
            html_lines.append("")
            continue

        # Blockquote
        if stripped.startswith("> "):
            content = stripped[2:]
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
            html_lines.append(f"<blockquote>{content}</blockquote>")
            continue

        # Unordered list
        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = stripped[2:]
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            content = re.sub(
                r"\[([^\]]+)\]\(([^)]+)\)",
                r'<a href="\2">\1</a>',
                content,
            )
            html_lines.append(f"<li>{content}</li>")
            continue

        # Ordered list
        ol_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if ol_match:
            if not in_ordered_list:
                html_lines.append("<ol>")
                in_ordered_list = True
            content = ol_match.group(2)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False
        if in_ordered_list:
            html_lines.append("</ol>")
            in_ordered_list = False

        # Headings
        if stripped.startswith("#### "):
            content = stripped[5:]
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_lines.append(f"<h4>{content}</h4>")
        elif stripped.startswith("### "):
            content = stripped[4:]
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_lines.append(f"<h3>{content}</h3>")
        elif stripped.startswith("## "):
            content = stripped[3:]
            html_lines.append(f"<h2>{content}</h2>")
        elif stripped.startswith("# "):
            content = stripped[2:]
            html_lines.append(f"<h1>{content}</h1>")
        else:
            content = stripped
            content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            content = re.sub(
                r"\[([^\]]+)\]\(([^)]+)\)",
                r'<a href="\2">\1</a>',
                content,
            )
            html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")
    if in_ordered_list:
        html_lines.append("</ol>")
    if in_table:
        html_lines.append("</tbody></table>")
    if in_code_block:
        html_lines.append("</code></pre>")

    return "\n".join(html_lines)


@router.get("/guide")
async def get_system_guide(request: Request):
    """Get auto-generated system guide.

    Returns documentation for the lease system, code execution, and
    state observation. Values are introspected from live config.

    No lease required.
    """
    return generate_guide(app=request.app)


@router.get("/guide/html", response_class=HTMLResponse)
async def get_system_guide_html(request: Request):
    """Get system guide as rendered HTML.

    Opens nicely in a browser. Also usable by agents via curl.

    No lease required.
    """
    guide = generate_guide(app=request.app)
    md = _render_markdown(guide)
    body = _md_to_html(md)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{guide['title']}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; line-height: 1.6; color: #24292e; }}
  h1 {{ border-bottom: 2px solid #e1e4e8; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #e1e4e8; padding-bottom: 0.3em; margin-top: 2em; }}
  h3 {{ margin-top: 1.5em; }}
  h4 {{ margin-top: 1em; color: #0366d6; }}
  pre {{ background: #f6f8fa; border-radius: 6px; padding: 16px; overflow-x: auto; }}
  code {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9em; }}
  p > code, li > code, td > code, th > code, h3 > code, h4 > code, blockquote > code {{ background: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; }}
  ul, ol {{ padding-left: 1.5em; }}
  li {{ margin: 0.25em 0; }}
  strong {{ font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #e1e4e8; padding: 0.5em 0.75em; text-align: left; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  blockquote {{ border-left: 4px solid #0366d6; padding: 0.5em 1em; margin: 1em 0; background: #f1f8ff; color: #24292e; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

"""Auto-generated SDK documentation endpoint.

Parses piper_sdk.py via AST to extract method signatures and docstrings.
This avoids importing rospy/ROS dependencies that may not be available
in the server's Python environment.
"""

from __future__ import annotations

import ast
import logging
import os
import textwrap

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code", tags=["code"])

_SDK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_sdk")


def _parse_class_from_file(filepath: str, class_name: str) -> dict:
    """Parse a Python file with AST and extract public methods from a class."""
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return _extract_class_info(node)

    return {"docstring": "", "methods": {}}


def _extract_class_info(cls_node: ast.ClassDef) -> dict:
    """Extract docstring and public methods from an AST ClassDef node."""
    docstring = ast.get_docstring(cls_node) or ""
    methods = {}

    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name.startswith("_"):
            continue

        # Build signature string from args
        sig = _build_signature(item)
        doc = ast.get_docstring(item) or ""

        methods[item.name] = {
            "signature": sig,
            "docstring": doc,
        }

    return {"docstring": docstring, "methods": methods}


def _build_signature(func_node: ast.FunctionDef) -> str:
    """Build a human-readable signature string from a FunctionDef AST node."""
    args = func_node.args
    parts = []

    # Positional args (skip 'self')
    all_args = [a.arg for a in args.args]
    defaults = [None] * (len(all_args) - len(args.defaults)) + list(args.defaults)

    for arg_name, default in zip(all_args, defaults):
        if arg_name == "self":
            continue
        if default is not None:
            default_str = _ast_value_to_str(default)
            parts.append(f"{arg_name}={default_str}")
        else:
            parts.append(arg_name)

    return f"({', '.join(parts)})"


def _ast_value_to_str(node: ast.expr) -> str:
    """Convert an AST default value node to a readable string."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.List, ast.Tuple)):
        elts = ", ".join(_ast_value_to_str(e) for e in node.elts)
        if isinstance(node, ast.List):
            return f"[{elts}]"
        return f"({elts})"
    if isinstance(node, ast.Dict):
        return "{}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_ast_value_to_str(node.operand)}"
    if isinstance(node, ast.Attribute):
        return f"{_ast_value_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.NameConstant):
        return repr(node.value)
    return "..."


def generate_sdk_docs() -> dict:
    """Generate SDK documentation by parsing SDK source files via AST."""
    docs = {
        "version": "3.0.0",
        "description": (
            "Piper Robot SDK. "
            "The wrapper pre-creates four instances: "
            "`env` (PiperRobotEnv, /usr/bin/python3 + ROS/MoveIt), "
            "`yolo` (YoloSDK, HTTP API + ROS depth), "
            "`grasp` (GraspSDK, HTTP API + ROS image), "
            "`memory` (MemorySDK, Spatial Memory Hub HTTP API). "
            "No import needed in submitted code."
        ),
        "modules": {},
        "usage": {
            "example": """# All instances are pre-created — no import needed

# ---- Robot control (env, ROS/MoveIt) ----
state = env.get_robot_state()
print(f"Joint positions: {state['joint_positions']}")
print(f"Gripper: {state['gripper_position']}")

env.move_joints([0.0, 0.08, -0.32, -0.02, 1.06, -0.034])
env.set_gripper(0.04)
env.move_to_pose([0.2, -0.05, 0.1, 0.0, 1.0, 0.0, 0.0])  # [x,y,z, qx,qy,qz,qw]

pose = env.get_robot_end_pose()
if pose:
    print(f"Position: {pose['position']}")
    print(f"Euler: {pose['orientation_euler']}")

images, timestamps = env.read_cameras()
env.reset()

# ---- Object detection (yolo, HTTP API + ROS depth/TF) ----
labels = yolo.detect_env()
detections = yolo.segment_3d("bottle")
for d in detections:
    print(f"{d['label']}: base={d['position_base']}, depth={d['depth_m']:.3f}m")

# ---- Grasp pose (grasp, HTTP API + ROS image) ----
# Use translation_base_retreat + quaternion_base as the MoveIt endpose.
# Do NOT move to translation_base (AnyGrasp geometric center only).
results = grasp.get_grasp_pose("bottle", top_k=5)
if results and results[0]["grasps"]:
    best = results[0]["grasps"][0]
    endpose = best["translation_base_retreat"] + best["quaternion_base"]
    env.move_to_pose(endpose)
""",
            "notes": [
                "Four pre-created instances: `env`, `yolo`, `grasp`, `memory`",
                "env runs under /usr/bin/python3 (ROS Noetic + MoveIt)",
                "yolo calls a remote YOLO HTTP service, then projects bboxes to 3D via ROS depth/camera_info/TF",
                "grasp calls a remote Grasp HTTP service (YOLO + AnyGrasp) and returns 6DoF poses",
                "memory calls Spatial Memory Hub HTTP API to upsert and query object memories",
                "All methods are synchronous (blocking)",
                "Robot has 6 arm joints + 1 gripper",
                "Gripper range: 0.0 (closed) to 0.06 (fully open) meters",
                "Arm control uses MoveIt services (trajectory planning + collision avoidance)",
                "End-effector pose is read from ROS topic /end_pose",
                "Camera names: 'left_camera_0_left' (cam_high), 'wrist_camera_0_left' (cam_low)",
                "yolo/grasp return plain dicts (JSON-serializable), not dataclass objects",
                "Grasp execution: move to translation_base_retreat + quaternion_base only; "
                "do not add a second move to translation_base.",
            ],
        },
    }

    # --- env (PiperRobotEnv) ---
    piper_sdk_path = os.path.join(_SDK_DIR, "piper_sdk.py")
    if os.path.exists(piper_sdk_path):
        class_info = _parse_class_from_file(piper_sdk_path, "PiperRobotEnv")
        docs["modules"]["env (PiperRobotEnv)"] = {
            "import": "# pre-created as `env`",
            "description": "Piper robot control via MoveIt — arm joints, gripper, end-effector, cameras",
            "environment": "/usr/bin/python3 (ROS/MoveIt)",
            **class_info,
        }

    # --- yolo (YoloSDK) ---
    yolo_sdk_path = os.path.join(_SDK_DIR, "yolo_sdk.py")
    if os.path.exists(yolo_sdk_path):
        class_info = _parse_class_from_file(yolo_sdk_path, "YoloSDK")
        docs["modules"]["yolo (YoloSDK)"] = {
            "import": "# pre-created as `yolo`",
            "description": (
                "YOLOv5 object detection + 3D localization. "
                "Calls a remote YOLO HTTP service for pixel bboxes, "
                "then reprojects to 3D using ROS depth image + camera_info + TF."
            ),
            "environment": "/usr/bin/python3 (HTTP API + ROS)",
            **class_info,
        }

    # --- grasp (GraspSDK) ---
    grasp_sdk_path = os.path.join(_SDK_DIR, "grasp_sdk.py")
    if os.path.exists(grasp_sdk_path):
        class_info = _parse_class_from_file(grasp_sdk_path, "GraspSDK")
        docs["modules"]["grasp (GraspSDK)"] = {
            "import": "# pre-created as `grasp`",
            "description": (
                "AnyGrasp 6DoF grasp pose generation via remote HTTP service. "
                "For arm motion: use translation_base_retreat + quaternion_base as env.move_to_pose; "
                "do not add a second move to translation_base (retreat is the final grasp pose)."
            ),
            "environment": "/usr/bin/python3 (HTTP API + ROS)",
            **class_info,
        }

    # --- memory (MemorySDK) ---
    memory_sdk_path = os.path.join(_SDK_DIR, "memory_sdk.py")
    if os.path.exists(memory_sdk_path):
        class_info = _parse_class_from_file(memory_sdk_path, "MemorySDK")
        docs["modules"]["memory (MemorySDK)"] = {
            "import": "# pre-created as `memory`",
            "description": "Spatial Memory Hub object memory upsert + name-based query",
            "environment": "/usr/bin/python3 (HTTP API)",
            **class_info,
        }

    docs["constants"] = {
        "piper_config": {
            "num_joints": 6,
            "description": "6 arm joints, controlled via MoveIt",
        },
        "gripper_range": {
            "min": 0.0,
            "max": 0.06,
            "description": "0.0 = fully closed, 0.06 = fully open (meters)",
        },
        "camera_mapping": {
            "cam_high": "left_camera_0_left",
            "cam_low": "wrist_camera_0_left",
            "description": "ROS camera topics mapped to SDK camera names",
        },
        "environments": {
            "robot_control": "/usr/bin/python3 (ROS Noetic + MoveIt)",
            "perception": "HTTP API services (YOLO + AnyGrasp on remote compute node)",
            "description": "Robot control in ROS process; perception via HTTP API calls to remote services",
        },
        "grasp_execution": {
            "move_to_pose": "translation_base_retreat + quaternion_base",
            "do_not_move_to": "translation_base (AnyGrasp geometric center only; not the arm target)",
            "description": "Retreat pose is the final end-effector pose for grasping on Piper; no second approach to translation_base.",
        },
    }

    return docs


@router.get("/sdk")
async def get_sdk_documentation():
    """Get auto-generated SDK documentation.

    Returns documentation for all available SDK modules, methods, and their
    signatures. This is generated by introspecting the actual code, so it's
    always accurate.

    No lease required.
    """
    return generate_sdk_docs()


@router.get("/sdk/markdown", response_class=HTMLResponse)
async def get_sdk_markdown():
    """Get SDK documentation as rendered HTML.

    Opens nicely in a browser. Also usable by agents via curl.

    No lease required.
    """
    docs = generate_sdk_docs()

    md = f"# Robot SDK Documentation\n\n"
    md += f"**Version:** {docs['version']}\n\n"
    md += f"{docs['description']}\n\n"

    # Usage
    md += "## Quick Start\n\n"
    md += "```python\n"
    md += docs["usage"]["example"]
    md += "```\n\n"

    md += "**Notes:**\n"
    for note in docs["usage"]["notes"]:
        md += f"- {note}\n"
    md += "\n"

    # Modules
    md += "## Modules\n\n"
    for module_name, module_info in docs.get("modules", {}).items():
        md += f"### `{module_name}`\n\n"
        md += f"**Import:** `{module_info['import']}`\n\n"
        md += f"{module_info['description']}\n\n"

        if module_info.get("docstring"):
            md += f"{module_info['docstring']}\n\n"

        md += "**Methods:**\n\n"
        for method_name, method_info in module_info.get("methods", {}).items():
            sig = method_info.get("signature", "()")
            md += f"#### `{method_name}{sig}`\n\n"
            if method_info.get("docstring"):
                md += f"{method_info['docstring']}\n\n"

    # YOLO result types
    if "yolo_types" in docs:
        md += "## YOLO Result Types\n\n"
        for type_name, type_info in docs["yolo_types"].items():
            md += f"### `{type_name}`\n\n"
            md += f"{type_info['description']}\n\n"
            md += "**Fields:**\n"
            for field_name, field_desc in type_info["fields"].items():
                md += f"- `{field_name}`: {field_desc}\n"
            md += "\n"
            if type_info.get("methods"):
                md += "**Methods:**\n"
                for method_name, method_info in type_info["methods"].items():
                    if isinstance(method_info, dict):
                        kind = method_info.get("kind", "method")
                        doc = method_info.get("docstring", "")
                        sig = method_info.get("signature", "()")
                        if kind == "property":
                            md += f"- `{method_name}` (property) — {doc}\n"
                        else:
                            first_line = doc.split("\n")[0] if doc else ""
                            md += f"- `{method_name}{sig}` — {first_line}\n"
                    else:
                        md += f"- `{method_name}` — {method_info}\n"
                md += "\n"

    # Constants
    if "constants" in docs:
        md += "## Constants\n\n"
        for const_name, const_info in docs["constants"].items():
            md += f"### {const_name}\n\n"
            if isinstance(const_info, dict):
                if "description" in const_info:
                    md += f"{const_info['description']}\n\n"
                for k, v in const_info.items():
                    if k != "description":
                        md += f"- `{k}`: {v}\n"
            md += "\n"

    # Advanced
    if "advanced" in docs:
        md += "## Advanced (Direct Backend Access)\n\n"
        for backend_name, backend_info in docs["advanced"].items():
            md += f"### `{backend_name}`\n\n"
            md += f"{backend_info['description']}\n\n"
            if "example" in backend_info:
                md += "```python\n"
                md += backend_info["example"]
                md += "```\n\n"
            if "methods" in backend_info:
                md += "**Methods:**\n"
                for method in backend_info["methods"]:
                    md += f"- `{method}`\n"
                md += "\n"

    # Render markdown to HTML using zero-dependency approach
    import html as html_mod
    raw_md = md

    # Convert markdown to HTML (lightweight, no external deps)
    lines = raw_md.split("\n")
    html_lines = []
    in_code_block = False
    in_list = False

    for line in lines:
        if line.startswith("```"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
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

        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
            continue

        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = stripped[2:]
            # Inline code
            import re
            content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
            html_lines.append(f"<li>{content}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        if stripped.startswith("#### "):
            content = stripped[5:]
            content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
            html_lines.append(f"<h4>{content}</h4>")
        elif stripped.startswith("### "):
            content = stripped[4:]
            content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
            html_lines.append(f"<h3>{content}</h3>")
        elif stripped.startswith("## "):
            content = stripped[3:]
            html_lines.append(f"<h2>{content}</h2>")
        elif stripped.startswith("# "):
            content = stripped[2:]
            html_lines.append(f"<h1>{content}</h1>")
        else:
            import re
            content = stripped
            content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
            html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")
    if in_code_block:
        html_lines.append("</code></pre>")

    body = "\n".join(html_lines)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot SDK Documentation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; line-height: 1.6; color: #24292e; }}
  h1 {{ border-bottom: 2px solid #e1e4e8; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #e1e4e8; padding-bottom: 0.3em; margin-top: 2em; }}
  h3 {{ margin-top: 1.5em; }}
  h4 {{ margin-top: 1em; color: #0366d6; }}
  pre {{ background: #f6f8fa; border-radius: 6px; padding: 16px; overflow-x: auto; }}
  code {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9em; }}
  p > code, li > code, h3 > code, h4 > code {{ background: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; }}
  ul {{ padding-left: 1.5em; }}
  li {{ margin: 0.25em 0; }}
  strong {{ font-weight: 600; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

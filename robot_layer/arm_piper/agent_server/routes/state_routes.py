"""GET /state, /health, /cameras endpoints for Piper robot."""

from __future__ import annotations

import logging
import base64
from typing import Optional

from fastapi import APIRouter, Response, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_no_cache_headers = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


def _build_camera_maps(cameras_cfg: dict) -> tuple[dict, dict]:
    """Build bidirectional alias <-> config-name maps from cameras config."""
    alias_to_cfg = {}
    cfg_to_alias = {}
    for name, info in cameras_cfg.items():
        alias = info.get("alias", name)
        alias_to_cfg[alias] = name
        cfg_to_alias[name] = alias
    return alias_to_cfg, cfg_to_alias


def _resolve_device_id(device_id: str, cameras_cfg: dict) -> str | None:
    """Accept either config name (cam_low) or alias (wrist_camera_0_left)."""
    if device_id in cameras_cfg:
        return device_id
    alias_to_cfg, _ = _build_camera_maps(cameras_cfg)
    return alias_to_cfg.get(device_id)


def create_router(state_agg, *args, **kwargs):
    """Create state router.

    Only state_agg and lease_mgr are used; extra positional/keyword args from
    the old multi-backend signature are accepted and ignored for compatibility.
    """
    robot_env = args[0] if len(args) > 0 else kwargs.get("robot_env")
    lease_mgr = args[1] if len(args) > 1 else kwargs.get("lease_mgr")

    @router.get("/state")
    async def get_state():
        """Get current robot state (arm joints, end-effector pose, gripper)."""
        return state_agg.state

    @router.get("/cameras")
    async def list_cameras():
        """List available cameras from ROS topics (configured in config.yaml)."""
        try:
            from robot_sdk.config import get_config
            cfg = get_config()
            cameras_cfg = cfg.get("cameras", {})
            _, cfg_to_alias = _build_camera_maps(cameras_cfg)
            cameras = [
                {
                    "device_id": name,
                    "alias": cfg_to_alias.get(name, name),
                    "name": name,
                    "enabled": info.get("enabled", True),
                    "frame_endpoint": f"/cameras/{name}/frame",
                }
                for name, info in cameras_cfg.items()
                if info.get("enabled", True)
            ]
        except Exception as e:
            logger.warning("Could not load camera config: %s", e)
            cameras = []
        return {"cameras": cameras}

    def _get_frame_from_recorder(cam_name: str):
        """Try to get a cached BGR frame from robot_env's ImageRecorder."""
        if robot_env is None:
            return None
        rec = getattr(robot_env, "image_recorder", None)
        if rec is None:
            return None
        img = getattr(rec, f"{cam_name}_rgb_image", None)
        return img

    @router.get("/cameras/realtime")
    def get_realtime_cameras():
        """Get latest frames for all enabled cameras as base64 JPEG."""
        try:
            import cv2
            from robot_sdk.config import get_config

            cfg = get_config()
            cameras_cfg = cfg.get("cameras", {})
            enabled = [
                (name, info.get("topic"))
                for name, info in cameras_cfg.items()
                if info.get("enabled", True) and info.get("topic")
            ]

            frames = {}
            errors = {}

            for name, topic in enabled:
                try:
                    img = _get_frame_from_recorder(name)
                    if img is None:
                        errors[name] = "no frame cached yet"
                        continue

                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        errors[name] = "failed to encode frame"
                        continue

                    import time
                    frames[name] = {
                        "topic": topic,
                        "timestamp": time.time(),
                        "jpeg_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
                    }
                except Exception as e:
                    errors[name] = str(e)

            return {"frames": frames, "errors": errors}
        except Exception as e:
            logger.warning("Realtime cameras error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=503)

    @router.get("/cameras/{device_id}/frame")
    def get_device_frame(device_id: str):
        """Get latest frame from camera as JPEG.

        Reads from the in-process ImageRecorder cache (subscribed via ROS).
        Falls back to rospy.wait_for_message if no cached frame is available.
        """
        try:
            import cv2
            from robot_sdk.config import get_config

            cfg = get_config()
            cameras_cfg = cfg.get("cameras", {})

            resolved = _resolve_device_id(device_id, cameras_cfg)
            if resolved is None:
                return JSONResponse({"error": f"unknown camera: {device_id}"}, status_code=404)

            topic = cameras_cfg[resolved].get("topic")
            if not topic:
                return JSONResponse({"error": f"no topic configured for camera: {device_id}"}, status_code=503)

            img = _get_frame_from_recorder(resolved)

            if img is None:
                try:
                    import rospy
                    from sensor_msgs.msg import Image as RosImage
                    from cv_bridge import CvBridge
                    msg = rospy.wait_for_message(topic, RosImage, timeout=3.0)
                    bridge = CvBridge()
                    try:
                        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    except Exception:
                        img = bridge.imgmsg_to_cv2(msg)
                except Exception as e:
                    return JSONResponse(
                        {"error": f"no cached frame and ROS fallback failed: {e}"},
                        status_code=503,
                    )

            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return JSONResponse({"error": "failed to encode frame"}, status_code=500)

            return Response(content=buf.tobytes(), media_type="image/jpeg", headers=_no_cache_headers)

        except Exception as e:
            logger.warning("Camera frame error (%s): %s", device_id, e)
            return JSONResponse({"error": str(e)}, status_code=503)

    @router.get("/health")
    async def health():
        """Server health and lease status."""
        result: dict = {"status": "ok"}
        if lease_mgr is not None:
            result["lease"] = lease_mgr.status()
        return result

    @router.get("/debug/ros")
    def debug_ros():
        """Diagnostic: check rospy node and topic status."""
        info = {}
        try:
            import rospy
            info["node_uri"] = rospy.get_node_uri()
            info["is_initialized"] = rospy.core.is_initialized()
            info["is_shutdown"] = rospy.core.is_shutdown()
            info["node_name"] = rospy.get_name()
        except Exception as e:
            info["rospy_error"] = str(e)

        try:
            import rospy
            from sensor_msgs.msg import Image as RosImage
            msg = rospy.wait_for_message(
                "/wrist_camera/color/image_raw", RosImage, timeout=5.0)
            info["wait_for_message"] = f"OK {msg.width}x{msg.height}"
        except Exception as e:
            info["wait_for_message"] = f"FAIL: {e}"

        try:
            import xmlrpc.client
            master = xmlrpc.client.ServerProxy("http://localhost:11311")
            code, _, state = master.getSystemState("/debug")
            _, subs, _ = state
            for topic, nodes in subs:
                if "image_raw" in topic:
                    info[f"master_subs_{topic}"] = nodes
        except Exception as e:
            info["master_error"] = str(e)

        return info

    @router.get("/logs", include_in_schema=False)
    async def get_server_logs(limit: int = Query(default=100, ge=1, le=500)):
        """Get recent server logs for dashboard display."""
        from logging_config import get_log_buffer
        buf = get_log_buffer()
        if buf is None:
            return {"logs": []}
        return {"logs": buf.get_logs(limit)}

    return router

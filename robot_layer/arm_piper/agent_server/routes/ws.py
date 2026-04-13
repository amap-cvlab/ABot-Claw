"""WebSocket handlers — /ws/state, /ws/feedback, and /ws/cameras."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth import KeyStore, check_ws_auth

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)


class CameraSubscription:
    """Tracks a camera WebSocket client's subscription."""
    
    def __init__(self, fps: int = 15, quality: int = 80, streams: list = None):
        self.fps = fps
        self.quality = quality
        self.streams = streams or ["color"]
        self.devices: list[str] = []  # Empty = all devices


def create_router(state_agg, config, camera_backend=None, key_store: KeyStore | None = None):
    """Create WebSocket router.

    Args:
        state_agg: StateAggregator instance
        config: ServerConfig
        camera_backend: Optional camera backend for /ws/cameras (pass None to disable)
        key_store: Optional KeyStore for WebSocket auth
    """

    async def _ws_auth(ws: WebSocket) -> bool:
        """Check auth and close with 4001 if unauthorized. Returns True if ok."""
        if key_store is None:
            return True
        ok, _name, _role = check_ws_auth(ws, key_store)
        if not ok:
            await ws.close(code=4001, reason="Unauthorized")
            return False
        return True

    @router.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        if not await _ws_auth(ws):
            return
        await ws.accept()
        interval = 1.0 / config.observer_state_hz
        try:
            while True:
                await ws.send_json(state_agg.state)
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("ws/state error")

    @router.websocket("/ws/cameras")
    async def ws_cameras(ws: WebSocket):
        """WebSocket endpoint for camera streaming.

        Clients can send JSON messages to configure streaming:
        - {"action": "subscribe", "streams": ["color"], "fps": 15, "quality": 80}
        - {"action": "unsubscribe"}
        - {"action": "get_state"}

        Server sends:
        - Binary frames: [4-byte header len][JSON header][JPEG data]
        - JSON state messages
        """
        if not await _ws_auth(ws):
            return
        await ws.accept()
        
        if camera_backend is None:
            await ws.send_json({"error": "Camera backend not available"})
            await ws.close()
            return

        subscription = CameraSubscription(
            fps=getattr(getattr(config, "cameras", None), "stream_fps", 15),
            quality=getattr(getattr(config, "cameras", None), "quality", 80),
            streams=getattr(getattr(config, "cameras", None), "streams", ["color"]),
        )
        streaming = True
        
        logger.info("Camera WebSocket client connected")
        
        async def send_frames():
            """Background task to send frames at configured FPS."""
            interval = 1.0 / subscription.fps
            last_send = 0
            
            while streaming:
                try:
                    now = time.time()
                    if now - last_send < interval:
                        await asyncio.sleep(0.01)
                        continue
                    
                    for stream_type in subscription.streams:
                        device_ids = subscription.devices or [None]
                        for device_id in device_ids:
                            frame = camera_backend.get_latest_decoded_frame(stream_type, device_id=device_id)
                            if frame is None:
                                continue
                            
                            # Encode frame
                            if stream_type == "color" and CV2_AVAILABLE:
                                encode_params = [cv2.IMWRITE_JPEG_QUALITY, subscription.quality]
                                _, encoded = cv2.imencode(".jpg", frame.frame, encode_params)
                                data = encoded.tobytes()
                                fmt = "jpeg"
                            elif stream_type == "depth" and CV2_AVAILABLE:
                                _, encoded = cv2.imencode(".png", frame.frame)
                                data = encoded.tobytes()
                                fmt = "png"
                            elif stream_type in ("infrared_left", "infrared_right") and CV2_AVAILABLE:
                                encode_params = [cv2.IMWRITE_JPEG_QUALITY, subscription.quality]
                                _, encoded = cv2.imencode(".jpg", frame.frame, encode_params)
                                data = encoded.tobytes()
                                fmt = "jpeg"
                            else:
                                data = frame.frame.tobytes()
                                fmt = "raw"
                            
                            # Build message: [header_len][header_json][binary_data]
                            header = json.dumps({
                                "type": "frame",
                                "device_id": frame.device_id,
                                "stream_type": stream_type,
                                "timestamp": frame.timestamp,
                                "width": frame.width,
                                "height": frame.height,
                                "format": fmt,
                                "depth_scale": frame.depth_scale,
                            }).encode()
                            
                            header_len = struct.pack(">I", len(header))
                            message = header_len + header + data
                            
                            await ws.send_bytes(message)
                    
                    last_send = now
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error("Camera stream error: %s", e)
                    await asyncio.sleep(0.1)
        
        # Start frame sending task
        send_task = asyncio.create_task(send_frames())
        
        try:
            # Handle incoming messages
            while True:
                try:
                    message = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                    data = json.loads(message)
                    action = data.get("action")
                    
                    if action == "subscribe":
                        subscription.streams = data.get("streams", ["color"])
                        subscription.fps = data.get("fps", 15)
                        subscription.quality = data.get("quality", 80)
                        subscription.devices = data.get("devices", [])
                        await ws.send_json({
                            "type": "ack",
                            "action": "subscribe",
                            "streams": subscription.streams,
                            "fps": subscription.fps,
                        })
                        logger.info("Camera subscription updated: %s at %d fps",
                                   subscription.streams, subscription.fps)
                    
                    elif action == "unsubscribe":
                        streaming = False
                        await ws.send_json({"type": "ack", "action": "unsubscribe"})
                    
                    elif action == "get_state":
                        state = camera_backend.get_state()
                        await ws.send_json({"type": "state", "data": state})
                    
                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError:
                    await ws.send_json({"error": "Invalid JSON"})
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.exception("Camera WebSocket error: %s", e)
        finally:
            streaming = False
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            logger.info("Camera WebSocket client disconnected")

    return router

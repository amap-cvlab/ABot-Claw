"""Display state manager — holds face GUI state and broadcasts via WebSocket."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

VALID_EXPRESSIONS = {"happy", "thinking", "sad", "neutral", "excited", "concerned"}
DEFAULT_EXPRESSION = "happy"


@dataclass
class DisplayState:
    """Current state of the face display."""

    face: str = DEFAULT_EXPRESSION
    text: str = ""
    text_size: str = "large"
    image_b64: str = ""
    image_mime: str = ""
    robot_status: str = "idle"
    queue_length: int = 0
    current_holder: str = ""

    def snapshot(self) -> dict:
        """Full state snapshot for new WebSocket connections."""
        return {"type": "snapshot", **asdict(self)}


class DisplayBroadcaster:
    """Manages display state and broadcasts updates to face GUI clients."""

    def __init__(self) -> None:
        self._state = DisplayState()
        self._connections: list[WebSocket] = []
        self._face_override: bool = False  # True if code set face explicitly

    @property
    def state(self) -> DisplayState:
        return self._state

    async def connect(self, ws: WebSocket) -> None:
        """Accept WebSocket and send full state snapshot."""
        await ws.accept()
        self._connections.append(ws)
        try:
            await ws.send_json(self._state.snapshot())
        except Exception:
            self._connections.remove(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    def _broadcast(self, message: dict) -> None:
        """Non-async broadcast — schedules sends on the event loop."""
        for ws in list(self._connections):
            try:
                asyncio.get_event_loop().create_task(ws.send_json(message))
            except Exception:
                self._connections.remove(ws)

    def set_face(self, expression: str) -> None:
        """Set face expression (from SDK code)."""
        if expression not in VALID_EXPRESSIONS:
            raise ValueError(f"Invalid expression: {expression}. Valid: {VALID_EXPRESSIONS}")
        self._state.face = expression
        self._face_override = True
        self._broadcast({"type": "face", "face": expression})

    def set_text(self, text: str, size: str = "large") -> None:
        """Show text on display."""
        self._state.text = text
        self._state.text_size = size
        self._broadcast({"type": "text", "text": text, "text_size": size})

    def set_image(self, image_b64: str, mime_type: str = "image/png") -> None:
        """Show image on display."""
        self._state.image_b64 = image_b64
        self._state.image_mime = mime_type
        self._broadcast({"type": "image", "image_b64": image_b64, "image_mime": mime_type})

    def clear_content(self) -> None:
        """Clear text + image, revert face to default."""
        self._state.text = ""
        self._state.text_size = "large"
        self._state.image_b64 = ""
        self._state.image_mime = ""
        self._state.face = DEFAULT_EXPRESSION
        self._face_override = False
        self._broadcast({"type": "clear"})

    def update_robot_status(self, status: str, queue_length: int, holder: str) -> None:
        """Update robot status (called by polling task)."""
        changed = (
            self._state.robot_status != status
            or self._state.queue_length != queue_length
            or self._state.current_holder != holder
        )
        self._state.robot_status = status
        self._state.queue_length = queue_length
        self._state.current_holder = holder

        # Auto-set face based on status (unless code has overridden it)
        if not self._face_override:
            face_map = {
                "idle": "happy",
                "executing": "thinking",
                "rewinding": "concerned",
                "error": "sad",
            }
            new_face = face_map.get(status, "neutral")
            if self._state.face != new_face:
                self._state.face = new_face
                changed = True

        if changed:
            self._broadcast({
                "type": "status",
                "robot_status": status,
                "queue_length": queue_length,
                "current_holder": holder,
                "face": self._state.face,
            })

    def announce(self, text: str) -> None:
        """Send a voice announcement to all connected display clients (fire-and-forget)."""
        self._broadcast({"type": "announce", "text": text})

    def on_execution_ended(self) -> None:
        """Called when code execution ends — clears content and resets face override."""
        self._face_override = False
        self._state.text = ""
        self._state.text_size = "large"
        self._state.image_b64 = ""
        self._state.image_mime = ""
        self._broadcast({"type": "clear"})

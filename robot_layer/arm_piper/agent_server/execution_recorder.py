"""Records camera snapshots and state data during code execution."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state import StateAggregator

logger = logging.getLogger(__name__)

_CODE_DIR = Path(__file__).resolve().parent.parent / "logs" / "code_executions"

# Capture interval (0.5 Hz = every 2 seconds)
_CAPTURE_INTERVAL = 2.0

# State capture interval (10 Hz)
_STATE_INTERVAL = 0.1


class ExecutionRecorder:
    """Records camera snapshots and state data during code execution.

    Spawns daemon threads that capture JPEG frames from all connected
    cameras at 0.5 Hz and state data at 10 Hz, writing both to disk.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._execution_id: Optional[str] = None
        self._camera_backend = None
        self._state_agg: Optional[StateAggregator] = None
        self._output_dir: Optional[Path] = None
        self._name_map: Dict[str, str] = {}  # device_id -> friendly name
        self._timestamps: List[float] = []
        self._frame_index = 0
        self._state_count = 0
        self._started_at = 0.0

    # -- lifecycle -----------------------------------------------------------

    def start(self, execution_id: str, camera_backend=None, state_agg: Optional[StateAggregator] = None) -> None:
        """Start recording frames and state for an execution.

        No-op if already recording.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ExecutionRecorder: already recording, ignoring start()")
            return

        # Build device_id -> friendly name map
        self._name_map = {}
        if camera_backend is not None:
            cameras = camera_backend.get_cameras()
            for cam in cameras:
                name = cam.get("name") or cam.get("device_id", "unknown")
                device_id = cam.get("device_id", "unknown")
                # Sanitize name for filesystem
                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
                self._name_map[device_id] = safe_name

        has_cameras = bool(self._name_map)
        has_state = state_agg is not None

        if not has_cameras and not has_state:
            logger.debug("ExecutionRecorder: no cameras or state available, skipping recording")
            return

        self._execution_id = execution_id
        self._camera_backend = camera_backend
        self._state_agg = state_agg
        self._output_dir = _CODE_DIR / execution_id
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamps = []
        self._frame_index = 0
        self._state_count = 0
        self._started_at = time.time()
        self._stop_event.clear()

        if has_cameras:
            self._thread = threading.Thread(
                target=self._capture_loop,
                name=f"exec-recorder-{execution_id}",
                daemon=True,
            )
            self._thread.start()

        if has_state:
            self._state_thread = threading.Thread(
                target=self._state_capture_loop,
                name=f"exec-state-{execution_id}",
                daemon=True,
            )
            self._state_thread.start()

        logger.info(
            "ExecutionRecorder: started for %s (%d cameras, state=%s)",
            execution_id, len(self._name_map), has_state,
        )

    def stop(self) -> Dict[str, Any]:
        """Stop recording and write metadata.

        Returns:
            Summary dict with frame_count, duration, cameras.
            Empty dict if recorder was not running.
        """
        camera_alive = self._thread is not None and self._thread.is_alive()
        state_alive = self._state_thread is not None and self._state_thread.is_alive()

        if not camera_alive and not state_alive:
            return {}

        self._stop_event.set()

        if camera_alive:
            self._thread.join(timeout=5.0)
            self._thread = None

        if state_alive:
            self._state_thread.join(timeout=5.0)
            self._state_thread = None

        stopped_at = time.time()
        duration = stopped_at - self._started_at
        frame_count = self._frame_index
        state_samples = self._state_count

        # Build frames list from files on disk
        frames_list = []
        if self._output_dir and self._output_dir.exists():
            frames_list = sorted(
                f.name for f in self._output_dir.iterdir()
                if f.suffix == ".jpg"
            )

        # Write metadata
        metadata = {
            "execution_id": self._execution_id,
            "started_at": self._started_at,
            "stopped_at": stopped_at,
            "duration": round(duration, 2),
            "interval": _CAPTURE_INTERVAL,
            "frame_count": frame_count,
            "cameras": [
                {"device_id": did, "name": name}
                for did, name in self._name_map.items()
            ],
            "timestamps": [round(t, 3) for t in self._timestamps],
            "frames": frames_list,
        }

        # Add state recording metadata if state was captured
        if state_samples > 0:
            metadata["state_log"] = "state_log.jsonl"
            metadata["state_interval"] = _STATE_INTERVAL
            metadata["state_samples"] = state_samples

        has_data = frame_count > 0 or state_samples > 0

        if self._output_dir and has_data:
            meta_path = self._output_dir / "metadata.json"
            try:
                meta_path.write_text(json.dumps(metadata, indent=2))
            except Exception as e:
                logger.error("ExecutionRecorder: failed to write metadata: %s", e)
        elif self._output_dir and not has_data:
            # No data captured — remove empty directory
            try:
                shutil.rmtree(self._output_dir, ignore_errors=True)
            except Exception:
                pass

        logger.info(
            "ExecutionRecorder: stopped for %s (%d frames, %d state samples, %.1fs)",
            self._execution_id, frame_count, state_samples, duration,
        )

        self._camera_backend = None
        self._state_agg = None
        self._execution_id = None
        self._output_dir = None

        return metadata

    # -- capture thread ------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: capture frames at 0.5 Hz."""
        while not self._stop_event.is_set():
            try:
                self._capture_once()
            except Exception as e:
                logger.error("ExecutionRecorder: capture error: %s", e)
            self._stop_event.wait(timeout=_CAPTURE_INTERVAL)

    def _capture_once(self) -> None:
        """Capture one set of frames from all cameras."""
        if self._camera_backend is None or self._output_dir is None:
            return

        frames = self._camera_backend.get_all_frames()
        if not frames:
            return

        now = time.time()
        idx = self._frame_index
        wrote_any = False

        for device_id, jpeg_bytes in frames.items():
            name = self._name_map.get(device_id, device_id)
            filename = f"{idx:04d}_{name}.jpg"
            filepath = self._output_dir / filename
            try:
                filepath.write_bytes(jpeg_bytes)
                wrote_any = True
            except Exception as e:
                logger.error("ExecutionRecorder: failed to write %s: %s", filename, e)

        if wrote_any:
            self._timestamps.append(now)
            self._frame_index += 1

    def _state_capture_loop(self) -> None:
        """Background thread: capture state at 10 Hz to JSONL."""
        state_path = self._output_dir / "state_log.jsonl"
        with open(state_path, "w") as f:
            while not self._stop_event.is_set():
                try:
                    state = self._state_agg.state  # thread-safe copy
                    f.write(json.dumps(state, separators=(",", ":")) + "\n")
                    f.flush()
                    self._state_count += 1
                except Exception as e:
                    logger.error("ExecutionRecorder: state capture error: %s", e)
                self._stop_event.wait(timeout=_STATE_INTERVAL)

    # -- queries -------------------------------------------------------------

    def get_recording(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a past recording.

        Returns:
            Metadata dict or None if not found.
        """
        meta_path = _CODE_DIR / execution_id / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception as e:
            logger.error("ExecutionRecorder: failed to read metadata for %s: %s",
                         execution_id, e)
            return None

        # Backfill frames list for older recordings that lack it
        if "frames" not in metadata:
            rec_dir = _CODE_DIR / execution_id
            metadata["frames"] = sorted(
                f.name for f in rec_dir.iterdir()
                if f.suffix == ".jpg"
            )

        return metadata

    def list_recordings(self) -> List[str]:
        """List execution IDs that have recordings (newest first)."""
        if not _CODE_DIR.exists():
            return []
        dirs = [
            d for d in _CODE_DIR.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return [d.name for d in dirs]

    def cleanup_old_recordings(self, keep: int = 20) -> None:
        """Delete oldest recording directories beyond *keep* limit."""
        if not _CODE_DIR.exists():
            return
        dirs = [
            d for d in _CODE_DIR.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]
        if len(dirs) <= keep:
            return
        dirs.sort(key=lambda d: d.stat().st_mtime)
        for old_dir in dirs[:-keep]:
            try:
                shutil.rmtree(old_dir)
                logger.info("ExecutionRecorder: cleaned up %s", old_dir.name)
            except Exception as e:
                logger.warning("ExecutionRecorder: failed to clean %s: %s",
                               old_dir.name, e)

"""System Logger - Unified state recording from all robot subsystems.

This module provides the SystemLogger class that records state from:
- Base server (pose, velocity)
- Franka server (joint positions, EE pose, wrench)
- Gripper server (position, width, object detection)

The logger runs as a background task, sampling state at configurable intervals
and storing waypoints that represent significant movement.

Example:
    logger = SystemLogger(config)

    # Start recording (provide state function)
    await logger.start(get_state_fn)

    # ... robot operates ...

    # Get recorded trajectory
    waypoints = logger.get_waypoints()

    # Stop recording
    await logger.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from system_logger.waypoint import UnifiedWaypoint
from system_logger.config import LoggerConfig

logger = logging.getLogger(__name__)


class SystemLogger:
    """Unified state recorder for all robot subsystems.

    Records state from base, arm, and gripper at configurable intervals.
    Only records waypoints when movement exceeds configured thresholds.

    The logger maintains a FIFO buffer of waypoints with configurable maximum size.
    """

    def __init__(self, config: Optional[LoggerConfig] = None):
        """Initialize the system logger.

        Args:
            config: Logger configuration. Uses defaults if not provided.
        """
        self._config = config or LoggerConfig()
        self._waypoints: List[UnifiedWaypoint] = []
        self._next_index = 0

        # Background task state
        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self._state_fn: Optional[Callable[[], Dict[str, Any]]] = None

        # Statistics
        self._total_samples = 0
        self._recorded_samples = 0
        self._start_time: Optional[float] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def config(self) -> LoggerConfig:
        """Return the logger configuration."""
        return self._config

    @property
    def is_recording(self) -> bool:
        """Return True if recording is active (running and not paused)."""
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        """Return True if recording is paused."""
        return self._paused

    @property
    def waypoint_count(self) -> int:
        """Return the number of stored waypoints."""
        return len(self._waypoints)

    @property
    def duration(self) -> float:
        """Return the duration of the recorded trajectory in seconds."""
        if len(self._waypoints) < 2:
            return 0.0
        return self._waypoints[-1].t - self._waypoints[0].t

    # -------------------------------------------------------------------------
    # Recording control
    # -------------------------------------------------------------------------

    async def start(self, state_fn: Callable[[], Dict[str, Any]]) -> None:
        """Start recording state.

        Args:
            state_fn: Function that returns current robot state dict.
                      Expected format matches /state endpoint response.
        """
        if self._running:
            logger.warning("[SystemLogger] Already recording")
            return

        self._state_fn = state_fn
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._record_loop())
        logger.info(f"[SystemLogger] Started recording (interval={self._config.record_interval}s)")

    async def stop(self) -> None:
        """Stop recording state."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info(
            f"[SystemLogger] Stopped recording "
            f"({self._recorded_samples}/{self._total_samples} samples, "
            f"{len(self._waypoints)} waypoints)"
        )

    def pause(self) -> None:
        """Pause recording (e.g. during rewind)."""
        self._paused = True
        logger.info("[SystemLogger] Recording paused")

    def resume(self) -> None:
        """Resume recording after pause."""
        self._paused = False
        logger.info("[SystemLogger] Recording resumed")

    async def _record_loop(self) -> None:
        """Main recording loop - samples state at regular intervals."""
        while self._running:
            try:
                if self._state_fn is not None and not self._paused:
                    state = self._state_fn()
                    self._record_state(state)

                await asyncio.sleep(self._config.record_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SystemLogger] Error in record loop: {e}")
                await asyncio.sleep(self._config.record_interval)

    def _record_state(self, state: Dict[str, Any]) -> None:
        """Record a state snapshot as a waypoint if movement exceeds thresholds.

        Args:
            state: Current robot state from state function.
        """
        self._total_samples += 1
        t = time.time()

        # Create waypoint from state
        wp = UnifiedWaypoint.from_state(state, t, index=self._next_index)

        # Debug: log arm data availability every 100 samples
        if self._total_samples % 100 == 1:
            arm_joints = len(wp.arm_q)
            logger.debug(
                f"[SystemLogger] Sample {self._total_samples}: "
                f"base=[{wp.x:.3f}, {wp.y:.3f}], arm_q has {arm_joints} joints, "
                f"recorded={len(self._waypoints)}"
            )

        # Check if we should record this waypoint
        if len(self._waypoints) > 0:
            last_wp = self._waypoints[-1]

            # Skip if movement is below all thresholds
            if wp.is_similar_to(
                last_wp,
                base_pos_threshold=self._config.base_position_threshold,
                base_orient_threshold=self._config.base_orientation_threshold,
                arm_threshold=self._config.arm_threshold,
            ):
                return

        # Record the waypoint
        wp.index = self._next_index
        self._waypoints.append(wp)
        self._next_index += 1
        self._recorded_samples += 1

        # Debug: log when waypoint is recorded
        logger.debug(
            f"[SystemLogger] Recorded waypoint {wp.index}: "
            f"base=[{wp.x:.3f}, {wp.y:.3f}, {wp.theta:.3f}], "
            f"arm_q={len(wp.arm_q)} joints"
        )

        # Enforce max length (FIFO)
        if len(self._waypoints) > self._config.max_waypoints:
            self._waypoints = self._waypoints[-self._config.max_waypoints:]

    def record_manual(self, state: Dict[str, Any], tags: Optional[List[str]] = None) -> UnifiedWaypoint:
        """Manually record a waypoint (bypasses thresholds).

        Args:
            state: Current robot state.
            tags: Optional tags for the waypoint.

        Returns:
            The recorded waypoint.
        """
        t = time.time()
        wp = UnifiedWaypoint.from_state(state, t, index=self._next_index)
        if tags:
            wp.tags = tags
        self._waypoints.append(wp)
        self._next_index += 1
        self._recorded_samples += 1

        # Enforce max length
        if len(self._waypoints) > self._config.max_waypoints:
            self._waypoints = self._waypoints[-self._config.max_waypoints:]

        return wp

    # -------------------------------------------------------------------------
    # Waypoint access
    # -------------------------------------------------------------------------

    def get_waypoints(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> List[UnifiedWaypoint]:
        """Get waypoints in the specified range.

        Args:
            start_idx: Start index (inclusive). None = from beginning.
            end_idx: End index (exclusive). None = to end.

        Returns:
            List of waypoints.
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self._waypoints)
        return self._waypoints[start_idx:end_idx]

    def get_waypoint(self, idx: int) -> Optional[UnifiedWaypoint]:
        """Get a specific waypoint by index.

        Args:
            idx: Waypoint index (0 = oldest).

        Returns:
            Waypoint or None if index is invalid.
        """
        if 0 <= idx < len(self._waypoints):
            return self._waypoints[idx]
        return None

    def get_latest_waypoint(self) -> Optional[UnifiedWaypoint]:
        """Get the most recent waypoint.

        Returns:
            Latest waypoint or None if empty.
        """
        if self._waypoints:
            return self._waypoints[-1]
        return None

    def get_waypoints_by_tag(self, tag: str) -> List[UnifiedWaypoint]:
        """Get all waypoints with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of matching waypoints.
        """
        return [wp for wp in self._waypoints if tag in wp.tags]

    # -------------------------------------------------------------------------
    # Trajectory manipulation
    # -------------------------------------------------------------------------

    def truncate(self, keep_n: int) -> None:
        """Keep only the first N waypoints.

        Used after rewind to remove waypoints beyond the rewind target.

        Args:
            keep_n: Number of waypoints to keep from the beginning.
        """
        if keep_n < len(self._waypoints):
            self._waypoints = self._waypoints[:keep_n]
            logger.info(f"[SystemLogger] Truncated to {keep_n} waypoints")

    def clear(self) -> None:
        """Clear all waypoints."""
        self._waypoints.clear()
        self._next_index = 0
        self._total_samples = 0
        self._recorded_samples = 0
        logger.info("[SystemLogger] Cleared all waypoints")

    # -------------------------------------------------------------------------
    # Status and info
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get recorder status and statistics.

        Returns:
            Status dictionary.
        """
        return {
            "is_recording": self._running,
            "waypoint_count": len(self._waypoints),
            "max_waypoints": self._config.max_waypoints,
            "duration_sec": self.duration,
            "record_interval": self._config.record_interval,
            "total_samples": self._total_samples,
            "recorded_samples": self._recorded_samples,
            "efficiency": (
                self._recorded_samples / self._total_samples
                if self._total_samples > 0 else 0.0
            ),
            "thresholds": {
                "base_position": self._config.base_position_threshold,
                "base_orientation": self._config.base_orientation_threshold,
                "arm": self._config.arm_threshold,
            },
        }

    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get information about the recorded trajectory.

        Returns:
            Trajectory information.
        """
        if len(self._waypoints) == 0:
            return {
                "length": 0,
                "duration_sec": 0.0,
                "first_waypoint": None,
                "last_waypoint": None,
            }

        return {
            "length": len(self._waypoints),
            "duration_sec": self.duration,
            "first_waypoint": self._waypoints[0].to_dict(),
            "last_waypoint": self._waypoints[-1].to_dict(),
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_to_file(self, path: str) -> None:
        """Save waypoints to a JSON file.

        Args:
            path: File path to save to.
        """
        data = {
            "version": "1.0",
            "config": {
                "record_interval": self._config.record_interval,
                "base_position_threshold": self._config.base_position_threshold,
                "base_orientation_threshold": self._config.base_orientation_threshold,
                "arm_threshold": self._config.arm_threshold,
            },
            "waypoints": [wp.to_dict() for wp in self._waypoints],
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[SystemLogger] Saved {len(self._waypoints)} waypoints to {path}")

    def load_from_file(self, path: str) -> None:
        """Load waypoints from a JSON file.

        Args:
            path: File path to load from.
        """
        with open(path, "r") as f:
            data = json.load(f)

        self._waypoints = [
            UnifiedWaypoint.from_dict(wp_data)
            for wp_data in data.get("waypoints", [])
        ]

        if self._waypoints:
            self._next_index = max(wp.index for wp in self._waypoints) + 1
        else:
            self._next_index = 0

        logger.info(f"[SystemLogger] Loaded {len(self._waypoints)} waypoints from {path}")

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of waypoints."""
        return len(self._waypoints)

    def __getitem__(self, idx: int) -> UnifiedWaypoint:
        """Get waypoint by index."""
        return self._waypoints[idx]

    def __iter__(self):
        """Iterate over waypoints."""
        return iter(self._waypoints)

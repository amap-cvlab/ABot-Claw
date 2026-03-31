"""Rewind Orchestrator - Coordinated rewind across multiple robot subsystems.

This module provides the RewindOrchestrator class that coordinates rewind
operations across base, arm, and gripper servers.

The orchestrator:
1. Reads waypoints from the SystemLogger
2. Sends coordinated commands to each backend
3. Handles timing and synchronization
4. Provides dry-run capability for testing

Example:
    orchestrator = RewindOrchestrator(logger, config)
    orchestrator.set_backends(base_backend, arm_backend, gripper_backend)

    # Rewind 10% of trajectory
    result = await orchestrator.rewind_percentage(10.0)

    # Rewind to specific waypoint
    result = await orchestrator.rewind_to_waypoint(idx=50)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol

from system_logger.waypoint import UnifiedWaypoint
from system_logger.logger import SystemLogger
from system_logger.config import RewindConfig, WorkspaceBounds

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Log buffer for dashboard display
# -----------------------------------------------------------------------------

class RewindLogBuffer(logging.Handler):
    """Captures rewind-related logs for dashboard display."""

    def __init__(self, max_entries: int = 100):
        super().__init__()
        self.max_entries = max_entries
        self._buffer: deque = deque(maxlen=max_entries)
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        """Capture log record to buffer."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": self.format(record),
            }
            self._buffer.append(entry)
        except Exception:
            pass  # Don't let logging errors break things

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent log entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of log entries, oldest first (new stuff at bottom).
        """
        entries = list(self._buffer)
        return entries[-limit:]  # Chronological order (oldest first)

    def clear(self) -> None:
        """Clear the log buffer."""
        self._buffer.clear()


# Global log buffer instance
_rewind_log_buffer = RewindLogBuffer()
logger.addHandler(_rewind_log_buffer)


def get_rewind_log_buffer() -> RewindLogBuffer:
    """Get the global rewind log buffer."""
    return _rewind_log_buffer


# -----------------------------------------------------------------------------
# Backend protocols (interfaces)
# -----------------------------------------------------------------------------

class BaseBackendProtocol(Protocol):
    """Protocol for base backend interface."""

    def execute_action(self, x: float, y: float, theta: float) -> None:
        """Move base to position."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current base state."""
        ...


class ArmBackendProtocol(Protocol):
    """Protocol for arm backend interface."""

    def send_joint_position(self, q: List[float], blocking: bool = True) -> bool:
        """Move arm to joint positions."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current arm state."""
        ...

    def set_control_mode(self, mode: int) -> bool:
        """Set arm control mode (1 = JOINT_POSITION)."""
        ...

    def set_gains(self, **kwargs) -> bool:
        """Set impedance control gains."""
        ...


class GripperBackendProtocol(Protocol):
    """Protocol for gripper backend interface."""

    def move(self, position: int, speed: int = 255, force: int = 255) -> tuple:
        """Move gripper to position (0-255)."""
        ...

    def open(self, speed: int = 255, force: int = 255) -> tuple:
        """Open the gripper fully."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current gripper state."""
        ...


# -----------------------------------------------------------------------------
# Rewind result
# -----------------------------------------------------------------------------

@dataclass
class RewindResult:
    """Result of a rewind operation."""

    success: bool
    steps_rewound: int = 0
    start_waypoint_idx: int = 0
    end_waypoint_idx: int = 0
    error: str = ""
    waypoints_executed: List[Dict[str, Any]] = field(default_factory=list)
    components_rewound: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Rewind Orchestrator
# -----------------------------------------------------------------------------

class RewindOrchestrator:
    """Coordinates rewind operations across all robot subsystems.

    The orchestrator manages the rewind process:
    1. Determines which waypoints to traverse
    2. Sends commands to each backend in sequence
    3. Waits for settling between waypoints
    4. Truncates the trajectory after successful rewind

    Supports selective component rewind (base only, arm only, or both).
    """

    def __init__(
        self,
        system_logger: SystemLogger,
        config: Optional[RewindConfig] = None,
        workspace_bounds: Optional[WorkspaceBounds] = None,
    ):
        """Initialize the rewind orchestrator.

        Args:
            system_logger: SystemLogger instance with recorded waypoints.
            config: Rewind configuration.
            workspace_bounds: Workspace boundary definitions.
        """
        self._logger = system_logger
        self._config = config or RewindConfig()
        self._bounds = workspace_bounds or WorkspaceBounds()

        # Backends (set via set_backends)
        self._base_backend: Optional[BaseBackendProtocol] = None
        self._arm_backend: Optional[ArmBackendProtocol] = None
        self._gripper_backend: Optional[GripperBackendProtocol] = None
        self._mocap_backend: Any = None  # Optional MocapBackend for odom correction

        # State
        self._is_rewinding = False
        self._cancel_requested = False

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    @property
    def bounds(self) -> "WorkspaceBounds":
        """Return the workspace bounds object."""
        return self._bounds

    @property
    def config(self) -> RewindConfig:
        """Return the rewind configuration."""
        return self._config

    @property
    def is_rewinding(self) -> bool:
        """Return True if a rewind operation is in progress."""
        return self._is_rewinding

    def cancel_rewind(self) -> None:
        """Request cancellation of any in-progress rewind."""
        if self._is_rewinding:
            self._cancel_requested = True
            logger.warning("[RewindOrchestrator] Rewind cancellation requested")

    @property
    def trajectory_length(self) -> int:
        """Return the number of waypoints in the trajectory."""
        return len(self._logger)

    def set_backends(
        self,
        base_backend: Optional[BaseBackendProtocol] = None,
        arm_backend: Optional[ArmBackendProtocol] = None,
        gripper_backend: Optional[GripperBackendProtocol] = None,
        mocap_backend: Any = None,
    ) -> None:
        """Set the backend interfaces for rewind commands.

        Args:
            base_backend: Base control backend.
            arm_backend: Arm control backend.
            gripper_backend: Gripper control backend.
            mocap_backend: Optional MocapBackend for odom↔mocap correction.
        """
        self._base_backend = base_backend
        self._arm_backend = arm_backend
        self._gripper_backend = gripper_backend
        self._mocap_backend = mocap_backend

    def _world_to_odom(self, wx: float, wy: float, wtheta: float) -> Optional[tuple]:
        """Convert a world-frame (mocap) target to odom-frame command.

        Uses the current odom↔mocap readings to compute the rigid transform
        (rotation + translation) between frames. This correctly handles
        heading drift — a pure additive offset fails when theta drifts.

        Args:
            wx, wy, wtheta: Target pose in world/mocap frame.

        Returns:
            (odom_x, odom_y, odom_theta) command, or None if mocap unavailable.
        """
        if self._mocap_backend is None or self._base_backend is None:
            return None
        try:
            mocap_state = self._mocap_backend.get_state()
        except Exception:
            return None
        if not mocap_state.get("tracking_valid", False):
            return None
        mocap = mocap_state.get("mocap_pose")
        if mocap is None:
            return None
        try:
            base_state = self._base_backend.get_state()
        except Exception:
            return None
        odom = base_state.get("base_pose", [0.0, 0.0, 0.0])

        import math
        alpha = odom[2] - mocap[2]  # heading drift
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        # Displacement from current mocap to target (world frame)
        dx_w = wx - mocap[0]
        dy_w = wy - mocap[1]

        # Rotate into odom frame and add to current odom position
        return (
            odom[0] + cos_a * dx_w - sin_a * dy_w,
            odom[1] + sin_a * dx_w + cos_a * dy_w,
            wtheta + alpha,
        )

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------

    def is_base_out_of_bounds(self, state: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the base is currently outside the workspace.

        Args:
            state: Current robot state. If None, queries base backend.

        Returns:
            True if base is out of bounds.
        """
        if state is None and self._base_backend:
            state = self._base_backend.get_state()

        if state is None:
            return False

        pose = state.get("base_pose", [0, 0, 0])
        x, y = pose[0], pose[1]

        return not self._bounds.is_base_in_bounds(x, y)

    def get_boundary_status(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed status of current position relative to workspace boundary.

        Args:
            state: Current robot state. If None, queries base backend.

        Returns:
            Boundary status information.
        """
        if state is None and self._base_backend:
            state = self._base_backend.get_state()

        if state is None:
            return {"error": "No state available"}

        pose = state.get("base_pose", [0, 0, 0])
        x, y = pose[0], pose[1]

        distances = self._bounds.base_distance_to_boundary(x, y)

        result = {
            "x": x,
            "y": y,
            "out_of_bounds": not self._bounds.is_base_in_bounds(x, y),
            "boundary_type": "hull" if self._bounds.has_hull else "aabb",
            "bounds": {
                "x_min": self._bounds.base_x_min,
                "x_max": self._bounds.base_x_max,
                "y_min": self._bounds.base_y_min,
                "y_max": self._bounds.base_y_max,
            },
            "distances": distances,
        }

        if self._bounds.has_hull:
            result["hull_vertices"] = self._bounds.hull_vertices

        return result

    def find_last_safe_waypoint(self) -> Optional[int]:
        """Find the index of the last waypoint where base was within bounds.

        Returns:
            Index of last safe waypoint, or None if none found.
        """
        margin = self._config.safety_margin

        for i in range(len(self._logger) - 1, -1, -1):
            wp = self._logger[i]
            if self._bounds.is_base_in_bounds(wp.x, wp.y, margin=margin):
                return i

        return None

    # -------------------------------------------------------------------------
    # Rewind operations
    # -------------------------------------------------------------------------

    async def rewind_steps(
        self,
        steps: int,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind the robot by a specified number of steps backward.

        Args:
            steps: Number of waypoints to rewind.
            dry_run: If True, only return what would happen.
            components: List of components to rewind ("base", "arm", "gripper").
                        None = use config defaults.

        Returns:
            RewindResult with operation details.
        """
        if self._is_rewinding:
            return RewindResult(success=False, error="Rewind already in progress")

        trajectory_len = len(self._logger)
        if trajectory_len == 0:
            return RewindResult(success=False, error="No trajectory history available")

        if steps <= 0:
            return RewindResult(success=False, error=f"Invalid steps: {steps}, must be positive")

        # Calculate waypoint indices
        start_idx = trajectory_len - 1
        end_idx = max(0, trajectory_len - 1 - steps)
        actual_steps = start_idx - end_idx

        if actual_steps == 0:
            return RewindResult(
                success=True,
                steps_rewound=0,
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                error="Already at beginning of trajectory",
            )

        return await self._execute_rewind(start_idx, end_idx, dry_run, components)

    async def rewind_percentage(
        self,
        percentage: float,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind the robot by a percentage of the trajectory.

        Args:
            percentage: Percentage of trajectory to rewind (0-100).
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        if percentage < 0 or percentage > 100:
            return RewindResult(
                success=False,
                error=f"Invalid percentage: {percentage}, must be 0-100",
            )

        trajectory_len = len(self._logger)
        if trajectory_len == 0:
            return RewindResult(success=False, error="No trajectory history available")

        steps = max(1, int(trajectory_len * percentage / 100))
        return await self.rewind_steps(steps, dry_run, components)

    async def rewind_to_safe(
        self,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind to the last safe waypoint (inside workspace boundary).

        Args:
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        safe_idx = self.find_last_safe_waypoint()

        if safe_idx is None:
            return RewindResult(
                success=False,
                error="No safe waypoint found in trajectory history",
            )

        trajectory_len = len(self._logger)
        start_idx = trajectory_len - 1

        if safe_idx >= start_idx:
            return RewindResult(
                success=True,
                steps_rewound=0,
                start_waypoint_idx=start_idx,
                end_waypoint_idx=safe_idx,
                error="Already at or past the last safe waypoint",
            )

        return await self._execute_rewind(start_idx, safe_idx, dry_run, components)

    async def rewind_to_waypoint(
        self,
        waypoint_idx: int,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind to a specific waypoint index.

        Args:
            waypoint_idx: Target waypoint index (0 = oldest).
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        trajectory_len = len(self._logger)

        if waypoint_idx < 0 or waypoint_idx >= trajectory_len:
            return RewindResult(
                success=False,
                error=f"Invalid waypoint index: {waypoint_idx}",
            )

        start_idx = trajectory_len - 1
        return await self._execute_rewind(start_idx, waypoint_idx, dry_run, components)

    # Convergence thresholds for base homing
    BASE_HOME_POS_THRESHOLD = 0.05  # meters (5cm)
    BASE_HOME_THETA_THRESHOLD = 0.087  # rad (~5 deg)
    BASE_HOME_TIMEOUT = 10.0  # seconds
    BASE_HOME_CMD_RATE = 10.0  # Hz

    # Convergence thresholds for arm homing
    ARM_HOME_JOINT_THRESHOLD = 0.05  # rad (~3 deg)
    ARM_HOME_TIMEOUT = 10.0  # seconds
    ARM_HOME_CMD_RATE = 50.0  # Hz

    async def reset_to_home(
        self,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Reset to home by rewinding 100% of the trajectory, then
        converging the base and arm to their home poses.

        After the trajectory rewind, feedback loops keep commanding the
        target poses until errors are below threshold (corrects for
        tracking drift accumulated during open-loop rewind).

        Args:
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        logger.info("[RewindOrchestrator] Reset to home triggered")

        arm_connected = self._arm_backend and getattr(self._arm_backend, "is_connected", True)

        trajectory_len = len(self._logger)
        if trajectory_len > 0 and arm_connected:
            # Full trajectory rewind (safe retraction for arm)
            result = await self.rewind_percentage(100.0, dry_run, components)
        elif trajectory_len > 0 and not arm_connected:
            # No arm — skip slow trajectory replay, just clear trajectory
            logger.info("[RewindOrchestrator] Arm not connected, skipping trajectory rewind")
            if not dry_run:
                self._logger.truncate(0)
            result = RewindResult(
                success=True,
                steps_rewound=0,
                components_rewound=["base"],
            )
        else:
            # No trajectory — skip rewind, go straight to convergence
            logger.info("[RewindOrchestrator] No trajectory to rewind, going directly to home")
            result = RewindResult(
                success=True,
                steps_rewound=0,
                components_rewound=[],
            )

        if not dry_run and result.success:
            if arm_connected:
                await self._converge_arm_to_target(self._config.arm_home_q)
            if self._base_backend:
                await self._converge_base_to_target((0.0, 0.0, 0.0))

        return result

    async def go_home(self) -> None:
        """Converge arm and base to their home poses without rewinding.

        Skips trajectory rewind — just moves directly to home.
        Opens the gripper before moving arm/base.
        """
        logger.info("[RewindOrchestrator] Go home triggered (no rewind)")
        if self._gripper_backend:
            try:
                await asyncio.to_thread(self._gripper_backend.open)
                logger.info("[RewindOrchestrator] Opened gripper")
            except Exception as e:
                logger.warning(f"[RewindOrchestrator] Failed to open gripper: {e}")
        if self._arm_backend:
            await self._converge_arm_to_target(self._config.arm_home_q)
        if self._base_backend:
            await self._converge_base_to_target((0.0, 0.0, 0.0))
        logger.info("[RewindOrchestrator] Go home complete")

    async def _converge_base_to_target(
        self, target: tuple[float, float, float]
    ) -> None:
        """Keep commanding ``(x, y, theta)`` until the base is within threshold.

        When mocap is available, applies odom↔mocap offset correction so
        the base converges on the real-world target despite odometry drift.

        Acts like an integral correction — the base controller (Ruckig OTG)
        plans a trajectory to the target each time, so any residual error is
        corrected on the next command cycle.
        """
        tx, ty, ttheta = target
        interval = 1.0 / self.BASE_HOME_CMD_RATE
        deadline = asyncio.get_event_loop().time() + self.BASE_HOME_TIMEOUT

        logger.info(
            "[RewindOrchestrator] Converging base to (%.3f, %.3f, %.3f) "
            "(pos_thr=%.3fm, theta_thr=%.3frad, timeout=%.1fs)",
            tx, ty, ttheta,
            self.BASE_HOME_POS_THRESHOLD,
            self.BASE_HOME_THETA_THRESHOLD,
            self.BASE_HOME_TIMEOUT,
        )

        last_cmd = None  # Last commanded odom-frame target
        converged = False
        while asyncio.get_event_loop().time() < deadline:
            # Convert world-frame target to odom-frame command
            odom_cmd = self._world_to_odom(tx, ty, ttheta)
            if odom_cmd is not None:
                last_cmd = odom_cmd

            if last_cmd is not None:
                cmd_x, cmd_y, cmd_theta = last_cmd
            else:
                # No mocap — send target directly (pure odom)
                cmd_x, cmd_y, cmd_theta = tx, ty, ttheta

            try:
                self._base_backend.execute_action(cmd_x, cmd_y, cmd_theta)
            except Exception as e:
                logger.warning("[RewindOrchestrator] Base homing command failed: %s", e)
                break

            # Check convergence
            try:
                state = self._base_backend.get_state()
                odom = state.get("base_pose", [])
                if odom and len(odom) >= 3:
                    # Check odom convergence against commanded target
                    dx = abs(odom[0] - cmd_x)
                    dy = abs(odom[1] - cmd_y)
                    dtheta = abs(odom[2] - cmd_theta)
                    if dx < self.BASE_HOME_POS_THRESHOLD and \
                       dy < self.BASE_HOME_POS_THRESHOLD and \
                       dtheta < self.BASE_HOME_THETA_THRESHOLD:
                        converged = True
                        logger.info(
                            "[RewindOrchestrator] Base home reached "
                            "(err: x=%.4f y=%.4f theta=%.4f)",
                            dx, dy, dtheta,
                        )
                        break
            except Exception:
                pass

            await asyncio.sleep(interval)

        if not converged:
            try:
                state = self._base_backend.get_state()
                odom = state.get("base_pose", [0, 0, 0])
                logger.warning(
                    "[RewindOrchestrator] Base homing timeout — "
                    "odom=(%.3f, %.3f, %.3f) cmd=(%.3f, %.3f, %.3f) "
                    "target=(%.3f, %.3f, %.3f)",
                    odom[0], odom[1], odom[2],
                    cmd_x, cmd_y, cmd_theta,
                    tx, ty, ttheta,
                )
            except Exception:
                logger.warning("[RewindOrchestrator] Base homing timeout (could not read state)")

    async def _converge_arm_to_target(self, target_q: List[float]) -> None:
        """Keep commanding arm joint positions until within threshold.

        Uses smooth interpolation from current position to target to avoid
        velocity violations (same approach as robot_sdk arm.move_joints).
        """
        if not self._arm_backend or not target_q:
            return

        # Read current arm position
        try:
            state = self._arm_backend.get_state()
            current_q = state.get("q", [])
            if not current_q or len(current_q) != len(target_q):
                logger.warning("[RewindOrchestrator] Cannot read arm state for homing")
                return
        except Exception as e:
            logger.warning("[RewindOrchestrator] Arm state read failed: %s", e)
            return

        # Calculate max joint delta to determine duration
        max_delta = max(abs(c - t) for c, t in zip(current_q, target_q))

        # Check if already at target
        if max_delta < self.ARM_HOME_JOINT_THRESHOLD:
            logger.info(
                "[RewindOrchestrator] Arm already at home (max_delta=%.4f rad)",
                max_delta,
            )
            return

        # Auto-calculate duration: scale with distance, clamp to [2, 10] seconds
        duration = max(2.0, min(10.0, max_delta / 0.5 * 4.0))

        logger.info(
            "[RewindOrchestrator] Converging arm to home "
            "(max_delta=%.3f rad, duration=%.1fs)",
            max_delta, duration,
        )

        # Ensure arm is in JOINT_POSITION mode
        try:
            self._arm_backend.set_control_mode(1)
        except Exception as e:
            logger.warning("[RewindOrchestrator] Failed to set arm control mode: %s", e)

        interval = 1.0 / self.ARM_HOME_CMD_RATE
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + duration
        deadline = start_time + self.ARM_HOME_TIMEOUT

        while asyncio.get_event_loop().time() < deadline:
            now = asyncio.get_event_loop().time()
            t = min((now - start_time) / duration, 1.0)

            # Cubic ease-in-out interpolation
            if t < 0.5:
                s = 4 * t * t * t
            else:
                s = 1 - (-2 * t + 2) ** 3 / 2

            q_cmd = [c + s * (tgt - c) for c, tgt in zip(current_q, target_q)]

            try:
                self._arm_backend.send_joint_position(q_cmd, blocking=False)
            except Exception as e:
                logger.warning("[RewindOrchestrator] Arm homing command failed: %s", e)
                break

            # After interpolation finishes, check convergence
            if t >= 1.0:
                try:
                    state = self._arm_backend.get_state()
                    actual_q = state.get("q", [])
                    if actual_q and len(actual_q) == len(target_q):
                        max_err = max(abs(a - tgt) for a, tgt in zip(actual_q, target_q))
                        if max_err < self.ARM_HOME_JOINT_THRESHOLD:
                            logger.info(
                                "[RewindOrchestrator] Arm home reached (max_err=%.4f rad)",
                                max_err,
                            )
                            return
                except Exception:
                    pass

            await asyncio.sleep(interval)

        logger.warning("[RewindOrchestrator] Arm homing timeout")

    # -------------------------------------------------------------------------
    # Rewind execution (chunked smooth version)
    # -------------------------------------------------------------------------

    def _interpolate_joints(
        self,
        q_start: List[float],
        q_end: List[float],
        t: float,
    ) -> List[float]:
        """Cubic interpolation between two joint configurations.

        Args:
            q_start: Starting joint positions.
            q_end: Ending joint positions.
            t: Interpolation parameter [0, 1].

        Returns:
            Interpolated joint positions.
        """
        # Cubic ease-in-out for smooth motion
        if t < 0.5:
            s = 4 * t * t * t
        else:
            s = 1 - (-2 * t + 2) ** 3 / 2

        return [qs + s * (qe - qs) for qs, qe in zip(q_start, q_end)]

    def _interpolate_waypoint_sequence(
        self,
        waypoints: List[UnifiedWaypoint],
        t: float,
    ) -> List[float]:
        """Interpolate through a sequence of waypoints.

        Args:
            waypoints: List of waypoints to interpolate through.
            t: Interpolation parameter [0, 1] for entire sequence.

        Returns:
            Interpolated joint positions.
        """
        if len(waypoints) < 2:
            return waypoints[0].arm_q if waypoints and waypoints[0].arm_q else []

        # Map t to segment index and local t
        n_segments = len(waypoints) - 1
        segment_t = t * n_segments
        segment_idx = min(int(segment_t), n_segments - 1)
        local_t = segment_t - segment_idx

        q_start = waypoints[segment_idx].arm_q
        q_end = waypoints[segment_idx + 1].arm_q

        if not q_start or not q_end:
            return q_start or q_end or []

        return self._interpolate_joints(q_start, q_end, local_t)

    async def _execute_rewind(
        self,
        start_idx: int,
        end_idx: int,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Execute the rewind operation from start_idx to end_idx.

        Uses chunked smooth interpolation for arm motion to reduce jitter.
        Waypoints are grouped into chunks, and the arm interpolates smoothly
        through each chunk while the base moves to the chunk endpoint.

        Args:
            start_idx: Starting waypoint index (more recent).
            end_idx: Ending waypoint index (older, target).
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        # Determine which components to rewind
        if components is None:
            components = []
            if self._config.rewind_base and self._base_backend:
                components.append("base")
            if self._config.rewind_arm and self._arm_backend:
                components.append("arm")
            if self._config.rewind_gripper and self._gripper_backend:
                components.append("gripper")

        # Build list of waypoints to traverse (in reverse order)
        waypoints_to_execute: List[UnifiedWaypoint] = []
        for i in range(start_idx, end_idx - 1, -1):
            if 0 <= i < len(self._logger):
                waypoints_to_execute.append(self._logger[i])

        if dry_run:
            return RewindResult(
                success=True,
                steps_rewound=len(waypoints_to_execute),
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                waypoints_executed=[wp.to_dict() for wp in waypoints_to_execute],
                components_rewound=components,
            )

        self._is_rewinding = True
        self._logger.pause()
        executed_waypoints: List[Dict[str, Any]] = []

        # Open gripper before rewinding arm/base
        if "gripper" in components and self._gripper_backend:
            try:
                await asyncio.to_thread(self._gripper_backend.open)
                logger.info("[RewindOrchestrator] Opened gripper before rewind")
            except Exception as e:
                logger.warning(f"[RewindOrchestrator] Failed to open gripper: {e}")

        try:
            chunk_size = self._config.chunk_size
            chunk_duration = self._config.chunk_duration
            n_waypoints = len(waypoints_to_execute)
            n_chunks = (n_waypoints + chunk_size - 1) // chunk_size  # Ceiling division

            logger.info(
                f"[RewindOrchestrator] Starting chunked rewind from waypoint {start_idx} to {end_idx} "
                f"({n_waypoints} waypoints in {n_chunks} chunks, components: {components})"
            )

            # Set arm to JOINT_POSITION mode with lower gains for smoother rewind
            if "arm" in components and self._arm_backend:
                try:
                    self._arm_backend.set_control_mode(1)  # 1 = JOINT_POSITION
                    self._arm_backend.set_gains(
                        joint_stiffness=[200, 200, 200, 200, 100, 75, 25],
                        joint_damping=[30, 30, 30, 30, 15, 12, 8],
                    )
                    logger.info("[RewindOrchestrator] Set arm to JOINT_POSITION with low rewind gains")
                except Exception as e:
                    logger.warning(f"[RewindOrchestrator] Failed to set control mode/gains: {e}")
                    logger.warning("[RewindOrchestrator] Skipping arm component for this rewind")
                    components.remove("arm")

            command_interval = 1.0 / self._config.command_rate

            # Process waypoints in chunks
            for chunk_idx in range(n_chunks):
                if self._cancel_requested:
                    logger.warning("[RewindOrchestrator] Rewind cancelled")
                    break

                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, n_waypoints)
                chunk_waypoints = waypoints_to_execute[chunk_start:chunk_end]

                if not chunk_waypoints:
                    continue

                # Final waypoint of this chunk (target for base)
                final_wp = chunk_waypoints[-1]

                logger.info(
                    f"[RewindOrchestrator] Chunk {chunk_idx + 1}/{n_chunks} "
                    f"({len(chunk_waypoints)} waypoints)"
                )

                # Execute chunk with smooth arm interpolation
                await self._execute_chunk(
                    chunk_waypoints,
                    components,
                    chunk_duration,
                    command_interval,
                )

                # Record executed waypoints
                for wp in chunk_waypoints:
                    executed_waypoints.append(wp.to_dict())

                # Settle time between chunks (for base to catch up)
                if chunk_idx < n_chunks - 1 and self._config.settle_time > 0:
                    settle_end = asyncio.get_event_loop().time() + self._config.settle_time
                    while asyncio.get_event_loop().time() < settle_end:
                        # Keep sending final position to prevent arm timeout
                        if "arm" in components and self._arm_backend and final_wp.arm_q:
                            self._arm_backend.send_joint_position(final_wp.arm_q, blocking=False)
                        await asyncio.sleep(command_interval)

            logger.info(
                f"[RewindOrchestrator] Completed rewind: "
                f"{len(executed_waypoints)} waypoints executed in {n_chunks} chunks"
            )

            # Truncate trajectory to the target waypoint
            self._logger.truncate(end_idx + 1)

            return RewindResult(
                success=True,
                steps_rewound=len(executed_waypoints),
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                waypoints_executed=executed_waypoints,
                components_rewound=components,
            )

        except Exception as e:
            logger.error(f"[RewindOrchestrator] Error during rewind: {e}")
            return RewindResult(
                success=False,
                steps_rewound=len(executed_waypoints),
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                error=str(e),
                waypoints_executed=executed_waypoints,
                components_rewound=components,
            )
        finally:
            # Restore default gains
            if "arm" in components and self._arm_backend:
                try:
                    self._arm_backend.set_gains(
                        joint_stiffness=[600, 600, 600, 600, 250, 150, 50],
                        joint_damping=[50, 50, 50, 50, 30, 25, 15],
                    )
                    logger.info("[RewindOrchestrator] Restored default arm gains")
                except Exception as e:
                    logger.warning(f"[RewindOrchestrator] Failed to restore gains: {e}")
            self._is_rewinding = False
            self._cancel_requested = False
            self._logger.resume()

    def _interpolate_base_pose(
        self,
        waypoints: List[UnifiedWaypoint],
        t: float,
    ) -> tuple:
        """Interpolate base pose through a sequence of waypoints.

        Args:
            waypoints: List of waypoints to interpolate through.
            t: Interpolation parameter [0, 1] for entire sequence.

        Returns:
            Tuple of (x, y, theta).
        """
        if len(waypoints) < 2:
            wp = waypoints[0] if waypoints else None
            return (wp.x, wp.y, wp.theta) if wp else (0, 0, 0)

        # Map t to segment index and local t
        n_segments = len(waypoints) - 1
        segment_t = t * n_segments
        segment_idx = min(int(segment_t), n_segments - 1)
        local_t = segment_t - segment_idx

        wp_start = waypoints[segment_idx]
        wp_end = waypoints[segment_idx + 1]

        # Linear interpolation for base (Ruckig will smooth it)
        x = wp_start.x + local_t * (wp_end.x - wp_start.x)
        y = wp_start.y + local_t * (wp_end.y - wp_start.y)
        theta = wp_start.theta + local_t * (wp_end.theta - wp_start.theta)

        return (x, y, theta)

    async def _execute_chunk(
        self,
        chunk_waypoints: List[UnifiedWaypoint],
        components: List[str],
        duration: float,
        command_interval: float,
    ) -> None:
        """Execute a chunk of waypoints with smooth interpolation for both arm and base.

        Args:
            chunk_waypoints: Waypoints in this chunk.
            components: Components to command.
            duration: Total duration for this chunk (seconds).
            command_interval: Time between commands (seconds).
        """
        if not chunk_waypoints:
            return

        final_wp = chunk_waypoints[-1]

        # Send gripper to final position (no interpolation)
        if "gripper" in components and self._gripper_backend:
            self._gripper_backend.move(final_wp.gripper_position)

        # Interpolate both arm and base through all waypoints in chunk.
        # Arm streams at command_rate (50 Hz). Base sends interpolated
        # positions at 10 Hz to avoid Ruckig replanning jitter.
        #
        # Waypoint base_pose values may be in mocap frame (when mocap was
        # tracking during recording). We translate them to odom frame using
        # the current odom↔mocap offset so the base follows the real-world
        # trajectory even if odometry has drifted.
        base_interval = 0.1  # 10 Hz for base
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + duration
        last_base_cmd_time = 0.0
        last_base_cmd = None  # Last commanded odom-frame target

        while True:
            now = asyncio.get_event_loop().time()
            if now >= end_time:
                break

            # Calculate interpolation parameter [0, 1]
            t = (now - start_time) / duration
            t = min(max(t, 0.0), 1.0)

            # Interpolate and send base position at 10 Hz
            if "base" in components and self._base_backend:
                if now - last_base_cmd_time >= base_interval:
                    x, y, theta = self._interpolate_base_pose(chunk_waypoints, t)
                    # Convert from world/mocap frame to odom frame
                    odom_cmd = self._world_to_odom(x, y, theta)
                    if odom_cmd is not None:
                        last_base_cmd = odom_cmd
                    if last_base_cmd is not None:
                        self._base_backend.execute_action(*last_base_cmd)
                    else:
                        self._base_backend.execute_action(x, y, theta)
                    last_base_cmd_time = now

            # Interpolate and send arm position at command_rate
            if "arm" in components and self._arm_backend:
                q_interp = self._interpolate_waypoint_sequence(chunk_waypoints, t)
                if q_interp:
                    self._arm_backend.send_joint_position(q_interp, blocking=False)

            await asyncio.sleep(command_interval)

        # Ensure we end at the final position
        if "base" in components and self._base_backend:
            odom_cmd = self._world_to_odom(final_wp.x, final_wp.y, final_wp.theta)
            if odom_cmd is not None:
                last_base_cmd = odom_cmd
            if last_base_cmd is not None:
                self._base_backend.execute_action(*last_base_cmd)
            else:
                self._base_backend.execute_action(final_wp.x, final_wp.y, final_wp.theta)
        if "arm" in components and self._arm_backend and final_wp.arm_q:
            self._arm_backend.send_joint_position(final_wp.arm_q, blocking=False)

    async def _execute_waypoint(
        self,
        wp: UnifiedWaypoint,
        components: List[str],
    ) -> None:
        """Execute commands for a single waypoint.

        Args:
            wp: Waypoint to execute.
            components: Components to command.
        """
        # Base — convert waypoint pose from world/mocap frame to odom frame
        if "base" in components and self._base_backend:
            odom_cmd = self._world_to_odom(wp.x, wp.y, wp.theta)
            if odom_cmd is not None:
                self._base_backend.execute_action(*odom_cmd)
            else:
                self._base_backend.execute_action(wp.x, wp.y, wp.theta)

        # Arm - use non-blocking (streaming) for faster command rate
        if "arm" in components and self._arm_backend and wp.arm_q:
            self._arm_backend.send_joint_position(wp.arm_q, blocking=False)

        # Gripper
        if "gripper" in components and self._gripper_backend:
            self._gripper_backend.move(wp.gripper_position)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current rewind status.

        Returns:
            Status dictionary.
        """
        return {
            "is_rewinding": self._is_rewinding,
            "trajectory_length": len(self._logger),
            "config": {
                "settle_time": self._config.settle_time,
                "command_rate": self._config.command_rate,
                "rewind_base": self._config.rewind_base,
                "rewind_arm": self._config.rewind_arm,
                "rewind_gripper": self._config.rewind_gripper,
                "safety_margin": self._config.safety_margin,
            },
            "backends": {
                "base": self._base_backend is not None,
                "arm": self._arm_backend is not None,
                "gripper": self._gripper_backend is not None,
            },
        }

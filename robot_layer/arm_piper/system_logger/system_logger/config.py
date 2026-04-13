"""Configuration for the system logger.

This module contains all configuration options for state recording and rewind.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Convex hull utilities (pure Python, no external deps)
# ---------------------------------------------------------------------------

def convex_hull_2d(points: List[List[float]]) -> List[List[float]]:
    """Compute 2D convex hull using Andrew's monotone chain algorithm.

    Args:
        points: List of [x, y] pairs.

    Returns:
        Hull vertices in counter-clockwise order (no repeated endpoint).
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    if len(pts) <= 1:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove last point of each half because it's repeated
    return [list(p) for p in lower[:-1] + upper[:-1]]


def _point_in_convex_polygon(
    x: float, y: float, vertices: List[List[float]], margin: float = 0.0
) -> bool:
    """Test if (x, y) is inside a convex polygon using cross-product sign test.

    Args:
        x, y: Point to test.
        vertices: Hull vertices in CCW order.
        margin: Shrink boundary inward by this amount (meters).

    Returns:
        True if point is inside (with margin).
    """
    n = len(vertices)
    if n < 3:
        return False

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # Edge vector
        ex, ey = x2 - x1, y2 - y1
        # Cross product: positive means left of edge (inside for CCW polygon)
        cross = ex * (y - y1) - ey * (x - x1)
        # Inward normal length for margin check
        edge_len = math.hypot(ex, ey)
        if edge_len == 0:
            continue
        # Signed distance from edge (positive = inside)
        signed_dist = cross / edge_len
        if signed_dist < margin:
            return False
    return True


def _point_to_hull_min_distance(
    x: float, y: float, vertices: List[List[float]]
) -> float:
    """Signed distance from point to nearest hull edge.

    Positive = inside, negative = outside.
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    min_dist = float("inf")
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        edge_len = math.hypot(ex, ey)
        if edge_len == 0:
            continue
        # Signed distance (positive = inside for CCW polygon)
        signed_dist = (ex * (y - y1) - ey * (x - x1)) / edge_len
        min_dist = min(min_dist, signed_dist)

    return min_dist


@dataclass
class LoggerConfig:
    """Configuration for the SystemLogger.

    Attributes:
        max_waypoints: Maximum number of waypoints to store (FIFO buffer).
        record_interval: Time between recordings in seconds.
        base_position_threshold: Minimum base position change to record (meters).
        base_orientation_threshold: Minimum base orientation change to record (radians).
        arm_threshold: Minimum arm joint change to record (radians).
        record_base: Whether to record base state.
        record_arm: Whether to record arm state.
        record_gripper: Whether to record gripper state.
        record_cameras: Whether to record camera frames (not implemented yet).
    """

    # Buffer settings
    max_waypoints: int = 10000

    # Recording rate
    record_interval: float = 0.1  # 10 Hz

    # Movement thresholds for recording (skip if below all thresholds)
    base_position_threshold: float = 0.05    # 5 cm
    base_orientation_threshold: float = 0.1  # ~5.7 degrees
    arm_threshold: float = 0.05              # ~3 degrees

    # What to record
    record_base: bool = True
    record_arm: bool = True
    record_gripper: bool = True
    record_cameras: bool = False  # Not implemented yet

    # Auto-save settings
    auto_save: bool = False
    auto_save_path: Optional[str] = None
    auto_save_interval: float = 60.0  # seconds


ARM_HOME_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785]


@dataclass
class RewindConfig:
    """Configuration for the RewindOrchestrator.

    Attributes:
        settle_time: Time to wait between chunks during rewind (seconds).
        command_rate: Rate to send commands during rewind (Hz).
        chunk_size: Number of waypoints per chunk for smooth interpolation.
        chunk_duration: Duration to execute each chunk (seconds).
        rewind_base: Whether to rewind base.
        rewind_arm: Whether to rewind arm.
        rewind_gripper: Whether to rewind gripper.
        arm_velocity_scale: Scale factor for arm velocity during rewind (0-1).
        base_velocity_scale: Scale factor for base velocity during rewind (0-1).
        safety_margin: Stop rewind this far inside workspace boundary (meters).
    """

    # Timing
    settle_time: float = 0.0       # Time between chunks (reduced from 0.5)
    command_rate: float = 50.0     # Hz - must be > 10 Hz for arm (100ms timeout)

    # Chunked smooth rewind
    chunk_size: int = 30            # Waypoints per chunk (tune for smoothness vs responsiveness)
    chunk_duration: float = 3.0    # Seconds to execute each chunk (arm interpolation time)

    # What to rewind
    rewind_base: bool = True
    rewind_arm: bool = True
    rewind_gripper: bool = True    # Open gripper before rewinding arm/base

    # Velocity scaling (slower = safer)
    arm_velocity_scale: float = 0.3
    base_velocity_scale: float = 0.5

    # Safety
    safety_margin: float = 0.1     # meters inside workspace boundary

    # Arm home position (Franka Panda default)
    arm_home_q: List[float] = field(default_factory=lambda: list(ARM_HOME_Q))

    # Auto-rewind settings
    auto_rewind_enabled: bool = False
    auto_rewind_percentage: float = 10.0    # % of trajectory to rewind
    monitor_interval: float = 0.1           # seconds between boundary checks

    # Collision detection (active when auto_rewind_enabled is True)
    # Dual-threshold: arm when ratio exceeds arm_threshold, trigger when below trigger_threshold
    # Normal cruising ratio: 0.70-0.90 (never below 0.60)
    # Kick/collision ratio: drops to 0.20-0.40
    collision_arm_threshold: float = 0.6       # ratio must exceed this to arm detector
    collision_trigger_threshold: float = 0.5   # ratio must drop below this to trigger
    collision_min_cmd_speed: float = 0.05      # m/s minimum commanded speed to consider
    collision_grace_period: float = 0.3        # seconds below trigger threshold before firing


@dataclass
class WorkspaceBounds:
    """Workspace boundary definition for safety checks.

    Supports both axis-aligned bounding box (AABB) and convex hull boundaries.
    When a hull is set, ``is_base_in_bounds`` checks the hull instead of the AABB.

    Attributes:
        base_x_min: Minimum x position for base (meters).
        base_x_max: Maximum x position for base (meters).
        base_y_min: Minimum y position for base (meters).
        base_y_max: Maximum y position for base (meters).
        hull_vertices: Optional convex hull vertices [[x,y], ...] in CCW order.
        arm_q_min: Minimum joint angles (radians).
        arm_q_max: Maximum joint angles (radians).
    """

    # Base workspace (rectangle)
    base_x_min: float = -2.0
    base_x_max: float = 2.0
    base_y_min: float = -2.0
    base_y_max: float = 2.0

    # Convex hull boundary (overrides AABB when set)
    hull_vertices: Optional[List[List[float]]] = None

    # Arm joint limits (Franka Panda defaults)
    arm_q_min: List[float] = field(default_factory=lambda: [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])
    arm_q_max: List[float] = field(default_factory=lambda: [
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ])

    @property
    def has_hull(self) -> bool:
        """True if a convex hull boundary is active."""
        return self.hull_vertices is not None and len(self.hull_vertices) >= 3

    def set_hull(self, vertices: List[List[float]]) -> None:
        """Set convex hull boundary and update AABB to match hull bounding box.

        Args:
            vertices: Hull vertices [[x,y], ...] in CCW order.
        """
        self.hull_vertices = [list(v) for v in vertices]
        if vertices:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            self.base_x_min = min(xs)
            self.base_x_max = max(xs)
            self.base_y_min = min(ys)
            self.base_y_max = max(ys)

    def clear_hull(self) -> None:
        """Remove hull boundary, reverting to AABB-only checks."""
        self.hull_vertices = None

    def is_base_in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if base position is within bounds.

        Uses hull if present, otherwise falls back to AABB.

        Args:
            x: Base x position.
            y: Base y position.
            margin: Safety margin (shrinks bounds).

        Returns:
            True if within bounds.
        """
        if self.has_hull:
            return _point_in_convex_polygon(x, y, self.hull_vertices, margin)
        return (
            self.base_x_min + margin <= x <= self.base_x_max - margin and
            self.base_y_min + margin <= y <= self.base_y_max - margin
        )

    def is_arm_in_bounds(self, q: List[float], margin: float = 0.0) -> bool:
        """Check if arm joints are within bounds.

        Args:
            q: Joint angles (7 values).
            margin: Safety margin (shrinks bounds).

        Returns:
            True if within bounds.
        """
        if len(q) != 7:
            return True  # Can't check, assume OK

        for i, (qi, qmin, qmax) in enumerate(zip(q, self.arm_q_min, self.arm_q_max)):
            if not (qmin + margin <= qi <= qmax - margin):
                return False
        return True

    def base_distance_to_boundary(self, x: float, y: float) -> dict:
        """Get distances from base position to boundary.

        Uses hull signed distance if present, otherwise AABB edge distances.

        Args:
            x: Base x position.
            y: Base y position.

        Returns:
            Dict with distance information.
        """
        if self.has_hull:
            min_dist = _point_to_hull_min_distance(x, y, self.hull_vertices)
            return {"min_distance": min_dist}
        return {
            "x_min": x - self.base_x_min,
            "x_max": self.base_x_max - x,
            "y_min": y - self.base_y_min,
            "y_max": self.base_y_max - y,
            "min_distance": min(
                x - self.base_x_min,
                self.base_x_max - x,
                y - self.base_y_min,
                self.base_y_max - y,
            ),
        }

    def to_dict(self) -> dict:
        """Serialize bounds to a dictionary."""
        d = {
            "base_x_min": self.base_x_min,
            "base_x_max": self.base_x_max,
            "base_y_min": self.base_y_min,
            "base_y_max": self.base_y_max,
        }
        if self.hull_vertices is not None:
            d["hull_vertices"] = self.hull_vertices
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceBounds":
        """Create WorkspaceBounds from a dictionary."""
        bounds = cls(
            base_x_min=data.get("base_x_min", -2.0),
            base_x_max=data.get("base_x_max", 2.0),
            base_y_min=data.get("base_y_min", -2.0),
            base_y_max=data.get("base_y_max", 2.0),
        )
        hull = data.get("hull_vertices")
        if hull:
            bounds.hull_vertices = hull
        return bounds

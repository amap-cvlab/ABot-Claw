"""Unified waypoint dataclass for state recording.

This module defines the UnifiedWaypoint class that stores complete robot state
including base, arm, gripper, and optionally camera data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UnifiedWaypoint:
    """Complete robot state at a point in time.

    This dataclass stores the full state of all robot subsystems:
    - Base: pose [x, y, theta] and velocity [vx, vy, wz]
    - Arm: joint positions [q0..q6], end-effector pose, wrench
    - Gripper: position, width, object detection
    - Metadata: timestamp, index, tags

    Attributes:
        t: Timestamp (Unix time)
        index: Waypoint index in trajectory
        base_pose: Base pose [x, y, theta] in meters and radians
        base_velocity: Base velocity [vx, vy, wz] in m/s and rad/s
        arm_q: Arm joint positions [q0..q6] in radians
        arm_dq: Arm joint velocities [dq0..dq6] in rad/s
        ee_pose: End-effector pose as 4x4 matrix (16 floats, column-major)
        ee_pose_world: EE pose in world frame (16 floats)
        ee_wrench: Force/torque at EE [fx, fy, fz, tx, ty, tz]
        gripper_position: Gripper position (0-255)
        gripper_width: Gripper width in meters
        gripper_object_detected: Whether object is grasped
        tags: Optional tags for filtering/searching
        metadata: Optional additional metadata
    """

    # Timestamp and index
    t: float
    index: int = 0

    # Base state
    base_pose: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    odom_pose: Optional[List[float]] = None   # Raw wheel odometry [x, y, theta]
    mocap_pose: Optional[List[float]] = None  # Raw mocap pose [x, y, theta] or None

    # Arm state
    arm_q: List[float] = field(default_factory=list)
    arm_dq: List[float] = field(default_factory=list)
    ee_pose: List[float] = field(default_factory=list)
    ee_pose_world: List[float] = field(default_factory=list)
    ee_wrench: List[float] = field(default_factory=list)

    # Gripper state
    gripper_position: int = 0
    gripper_width: float = 0.0
    gripper_object_detected: bool = False

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert waypoint to dictionary.

        Returns:
            Dictionary representation of the waypoint.
        """
        return {
            "t": self.t,
            "index": self.index,
            "base_pose": self.base_pose,
            "base_velocity": self.base_velocity,
            "odom_pose": self.odom_pose,
            "mocap_pose": self.mocap_pose,
            "arm_q": self.arm_q,
            "arm_dq": self.arm_dq,
            "ee_pose": self.ee_pose,
            "ee_pose_world": self.ee_pose_world,
            "ee_wrench": self.ee_wrench,
            "gripper_position": self.gripper_position,
            "gripper_width": self.gripper_width,
            "gripper_object_detected": self.gripper_object_detected,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedWaypoint":
        """Create waypoint from dictionary.

        Args:
            data: Dictionary with waypoint data.

        Returns:
            UnifiedWaypoint instance.
        """
        return cls(
            t=data.get("t", 0.0),
            index=data.get("index", 0),
            base_pose=data.get("base_pose", [0.0, 0.0, 0.0]),
            base_velocity=data.get("base_velocity", [0.0, 0.0, 0.0]),
            odom_pose=data.get("odom_pose"),
            mocap_pose=data.get("mocap_pose"),
            arm_q=data.get("arm_q", []),
            arm_dq=data.get("arm_dq", []),
            ee_pose=data.get("ee_pose", []),
            ee_pose_world=data.get("ee_pose_world", []),
            ee_wrench=data.get("ee_wrench", []),
            gripper_position=data.get("gripper_position", 0),
            gripper_width=data.get("gripper_width", 0.0),
            gripper_object_detected=data.get("gripper_object_detected", False),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_state(cls, state: Dict[str, Any], t: float, index: int = 0) -> "UnifiedWaypoint":
        """Create waypoint from agent server state response.

        Args:
            state: State dict from /state endpoint.
            t: Timestamp.
            index: Waypoint index.

        Returns:
            UnifiedWaypoint instance.
        """
        base = state.get("base", {})
        arm = state.get("arm", {})
        gripper = state.get("gripper", {})

        raw_odom = base.get("odom_pose")
        raw_mocap = base.get("mocap_pose")

        return cls(
            t=t,
            index=index,
            base_pose=list(base.get("pose", [0.0, 0.0, 0.0])),
            base_velocity=list(base.get("velocity", [0.0, 0.0, 0.0])),
            odom_pose=list(raw_odom) if raw_odom else None,
            mocap_pose=list(raw_mocap) if raw_mocap else None,
            arm_q=list(arm.get("q", [])),
            arm_dq=list(arm.get("dq", [])),
            ee_pose=list(arm.get("ee_pose", [])),
            ee_pose_world=list(arm.get("ee_pose_world", [])),
            ee_wrench=list(arm.get("ee_wrench", [])),
            gripper_position=gripper.get("position", 0),
            gripper_width=gripper.get("position_mm", 0.0) / 1000.0,
            gripper_object_detected=gripper.get("object_detected", False),
        )

    # -------------------------------------------------------------------------
    # Convenience properties
    # -------------------------------------------------------------------------

    @property
    def x(self) -> float:
        """Base x position in meters."""
        return self.base_pose[0] if self.base_pose else 0.0

    @property
    def y(self) -> float:
        """Base y position in meters."""
        return self.base_pose[1] if len(self.base_pose) > 1 else 0.0

    @property
    def theta(self) -> float:
        """Base orientation in radians."""
        return self.base_pose[2] if len(self.base_pose) > 2 else 0.0

    @property
    def ee_position(self) -> List[float]:
        """End-effector position [x, y, z] from pose matrix."""
        if len(self.ee_pose) >= 16:
            return [self.ee_pose[12], self.ee_pose[13], self.ee_pose[14]]
        return [0.0, 0.0, 0.0]

    @property
    def ee_position_world(self) -> List[float]:
        """End-effector position in world frame [x, y, z]."""
        if len(self.ee_pose_world) >= 16:
            return [self.ee_pose_world[12], self.ee_pose_world[13], self.ee_pose_world[14]]
        return [0.0, 0.0, 0.0]

    # -------------------------------------------------------------------------
    # Distance calculations
    # -------------------------------------------------------------------------

    def base_distance_to(self, other: "UnifiedWaypoint") -> float:
        """Calculate Euclidean distance to another waypoint (base position only).

        Args:
            other: Another waypoint.

        Returns:
            Distance in meters.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def base_orientation_distance_to(self, other: "UnifiedWaypoint") -> float:
        """Calculate orientation difference to another waypoint.

        Args:
            other: Another waypoint.

        Returns:
            Angle difference in radians (0 to pi).
        """
        diff = self.theta - other.theta
        # Normalize to [-pi, pi]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return abs(diff)

    def arm_distance_to(self, other: "UnifiedWaypoint") -> float:
        """Calculate joint-space distance to another waypoint (arm only).

        Args:
            other: Another waypoint.

        Returns:
            L2 norm of joint angle differences in radians.
        """
        if not self.arm_q or not other.arm_q:
            return 0.0
        if len(self.arm_q) != len(other.arm_q):
            return 0.0

        sum_sq = 0.0
        for q1, q2 in zip(self.arm_q, other.arm_q):
            sum_sq += (q1 - q2) ** 2
        return math.sqrt(sum_sq)

    def ee_distance_to(self, other: "UnifiedWaypoint") -> float:
        """Calculate Cartesian distance between end-effectors.

        Args:
            other: Another waypoint.

        Returns:
            Distance in meters.
        """
        p1 = self.ee_position
        p2 = other.ee_position
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # -------------------------------------------------------------------------
    # Comparison helpers
    # -------------------------------------------------------------------------

    def is_similar_to(
        self,
        other: "UnifiedWaypoint",
        base_pos_threshold: float = 0.05,
        base_orient_threshold: float = 0.1,
        arm_threshold: float = 0.05,
    ) -> bool:
        """Check if this waypoint is similar to another.

        Used for filtering redundant waypoints during recording.
        Returns False (not similar) if ANY component moved significantly.

        Args:
            other: Another waypoint.
            base_pos_threshold: Position threshold in meters.
            base_orient_threshold: Orientation threshold in radians.
            arm_threshold: Arm joint threshold in radians.

        Returns:
            True if waypoints are similar (below all thresholds).
            False if any component moved significantly (should record).
        """
        # Check base position - if moved, not similar
        if self.base_distance_to(other) >= base_pos_threshold:
            return False

        # Check base orientation - if rotated, not similar
        if self.base_orientation_distance_to(other) >= base_orient_threshold:
            return False

        # Check arm - if joints moved, not similar
        # Only check if BOTH have arm data
        if self.arm_q and other.arm_q:
            if self.arm_distance_to(other) >= arm_threshold:
                return False
        # If one has arm data and other doesn't, consider not similar (state changed)
        elif bool(self.arm_q) != bool(other.arm_q):
            return False

        return True

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnifiedWaypoint(t={self.t:.2f}, "
            f"base=[{self.x:.3f}, {self.y:.3f}, {self.theta:.3f}], "
            f"arm={len(self.arm_q)} joints, "
            f"gripper={self.gripper_position})"
        )

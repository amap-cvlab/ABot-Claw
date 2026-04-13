"""Safety envelope checks for Piper robot arm."""

from __future__ import annotations

from dataclasses import dataclass

from config import SafetyConfig


@dataclass
class SafetyResult:
    ok: bool
    reason: str = ""
    detail: str = ""


class SafetyEnvelope:
    """Basic safety checks for Piper arm workspace and joint velocities."""

    def __init__(self, config: SafetyConfig) -> None:
        self._cfg = config

    def check_arm_cartesian(self, xyz: list[float]) -> SafetyResult:
        """Check that end-effector [x, y, z] is within the configured workspace."""
        if len(xyz) < 3:
            return SafetyResult(False, "invalid_input", "need at least 3 floats [x, y, z]")
        x, y, z = xyz[0], xyz[1], xyz[2]
        mn, mx = self._cfg.arm_workspace_min, self._cfg.arm_workspace_max
        if not (mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]):
            return SafetyResult(
                False,
                "out_of_bounds",
                f"arm EE position ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace",
            )
        return SafetyResult(True)

    def check_arm_joint_velocity(self, dq: list[float]) -> SafetyResult:
        """Check that all joint velocities are within limits."""
        for i, v in enumerate(dq):
            if abs(v) > self._cfg.arm_max_joint_vel:
                return SafetyResult(
                    False,
                    "velocity_limit",
                    f"joint {i} velocity {abs(v):.2f} rad/s exceeds limit {self._cfg.arm_max_joint_vel}",
                )
        return SafetyResult(True)

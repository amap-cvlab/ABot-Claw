"""Configuration for the hardware server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional  # noqa: F401 kept for external compatibility


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ServiceDefinition:
    """Definition of a managed backend service."""
    name: str                          # display name
    cmd: str                           # command to run
    cwd: str                           # working directory
    shell_prefix: str = ""             # e.g., "source /opt/ros/..."
    kill_patterns: list[str] = field(default_factory=list)
    auto_restart: bool = False
    depends_on: list[str] = field(default_factory=list)  # service keys this depends on


@dataclass
class ServiceManagerConfig:
    """Configuration for the service manager."""
    enabled: bool = True
    auto_start: bool = False           # start backends on server startup
    log_max_lines: int = 100
    health_check_interval_s: float = 5.0
    pid_file: str = ".agent_server_pids.json"
    services: dict[str, ServiceDefinition] = field(default_factory=dict)


def default_services() -> dict[str, ServiceDefinition]:
    """Return the default service definitions for Piper robot.

    These are ROS-based services. Customize commands and paths to match
    your actual launch scripts.
    """
    return {
        "roscore": ServiceDefinition(
            name="ROS Master",
            cmd="roscore",
            cwd=_PROJECT_ROOT,
            kill_patterns=["roscore", "rosmaster"],
        ),
        "piper_ros": ServiceDefinition(
            name="Piper ROS Driver",
            cmd="bash -c 'source /opt/ros/noetic/setup.bash && roslaunch piper_description piper_ctrl_single_arm.launch'",
            cwd=_PROJECT_ROOT,
            kill_patterns=["piper_ctrl_single_arm"],
            depends_on=["roscore"],
        ),
    }


@dataclass
class PollConfig:
    """Polling rate configuration."""
    poll_hz: float = 10.0


@dataclass
class SafetyConfig:
    # Arm workspace bounding box in base frame [min, max] for x, y, z (meters)
    arm_workspace_min: list[float] = field(default_factory=lambda: [-0.8, -0.8, 0.0])
    arm_workspace_max: list[float] = field(default_factory=lambda: [0.8, 0.8, 1.2])
    arm_max_joint_vel: float = 2.0  # rad/s per joint


@dataclass
class TimingConfig:
    """Central timing constants for the entire agent server stack.

    All timeouts, rates, and durations in one place so interactions between
    layers are visible.  Constants flow into the SDK subprocess via env vars
    (see CodeExecutor._create_temp_file).

    Timeout budget for a single blocking SDK call
    ──────────────────────────────────────────────
    ┌─ code_execution_timeout_s (300 s) ──────────────────────────────┐
    │  ┌─ motion_timeout_s (30 s) ─────────────────────────────────┐  │
    │  │  interpolation (2–15 s auto-calc)                         │  │
    │  │  ── then ──                                               │  │
    │  │  settle_timeout_s (3 s)  ← converge or raise ArmError    │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  Lease keeps alive while robot moves (movement detection).      │
    │  If arm is stuck and not moving:                                │
    │    settle_timeout_s (3 s) fires BEFORE lease_idle_timeout_s     │
    │    (15 s), so code gets a clean ArmError.                       │
    │                                                                 │
    │  ┌─ lease_idle_timeout_s (15 s) ──┐                             │
    │  │  lease revoked, code killed    │                             │
    │  └────────────────────────────────┘                             │
    └─────────────────────────────────────────────────────────────────┘

    Command rates
    ─────────────
    arm_command_rate_hz   50 Hz   joint/cartesian command streaming
    base_command_rate_hz  10 Hz   base velocity resend (must beat 250 ms hw timeout)
    """

    # -- Lease --
    lease_idle_timeout_s: float = 60.0       # idle time before revoke
    lease_max_duration_s: float = 300.0      # hard cap on any single lease
    lease_check_interval_s: float = 1.0      # how often the lease watchdog ticks

    # -- Code execution --
    code_execution_timeout_s: float = 300.0  # subprocess wall-clock limit

    # -- Arm motion --
    motion_timeout_s: float = 30.0           # overall timeout per blocking arm call
    settle_timeout_s: float = 3.0            # post-interpolation convergence window
    arm_command_rate_hz: float = 50.0        # streaming rate for arm commands
    arm_converge_pos_m: float = 0.03         # cartesian convergence threshold (meters)
    arm_converge_joint_rad: float = 0.02     # joint convergence threshold (radians)
    arm_converge_vel: float = 0.05           # max joint velocity to declare settled (rad/s)

    # -- Base motion --
    base_timeout_s: float = 30.0             # overall timeout per blocking base call
    base_command_rate_hz: float = 10.0       # resend rate (must beat 250 ms hw timeout)
    base_position_tolerance_m: float = 0.05  # convergence threshold (meters)
    base_angle_tolerance_rad: float = 0.05   # convergence threshold (radians)


# Singleton default — importable by anyone in the server process.
TIMING = TimingConfig()


@dataclass
class LeaseConfig:
    idle_timeout_s: float = TIMING.lease_idle_timeout_s
    max_duration_s: float = TIMING.lease_max_duration_s
    check_interval_s: float = TIMING.lease_check_interval_s
    reset_on_release: bool = True  # Auto-rewind to home when lease ends
    ticket_ttl_s: float = 60.0    # How long granted/cancelled tickets stay in memory


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8888
    dry_run: bool = False
    observer_state_hz: float = 10.0
    operator_state_hz: float = 100.0

    base: PollConfig = field(default_factory=PollConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    lease: LeaseConfig = field(default_factory=LeaseConfig)
    service_manager: ServiceManagerConfig = field(default_factory=ServiceManagerConfig)
    dashboard: bool = True

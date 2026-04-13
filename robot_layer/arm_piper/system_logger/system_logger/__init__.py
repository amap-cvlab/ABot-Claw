"""System Logger - Unified state recording and rewind orchestration.

This package provides:
- Unified state recording from all robot subsystems (base, arm, gripper, cameras)
- Coordinated rewind across multiple systems
- Waypoint storage and retrieval
"""

from system_logger.waypoint import UnifiedWaypoint
from system_logger.logger import SystemLogger
from system_logger.rewind_orchestrator import RewindOrchestrator, get_rewind_log_buffer
from system_logger.config import LoggerConfig, RewindConfig

__version__ = "0.1.0"
__all__ = [
    "UnifiedWaypoint",
    "SystemLogger",
    "RewindOrchestrator",
    "LoggerConfig",
    "RewindConfig",
    "get_rewind_log_buffer",
]

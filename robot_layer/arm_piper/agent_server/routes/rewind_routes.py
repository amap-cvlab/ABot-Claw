"""API routes for rewind operations."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from auth import require_admin

router = APIRouter(prefix="/rewind", tags=["rewind"])


class RewindStepsRequest(BaseModel):
    steps: int = Field(..., ge=1)
    dry_run: bool = False
    components: Optional[List[str]] = None  # ["base", "arm", "gripper"]


class RewindPercentageRequest(BaseModel):
    percentage: float = Field(..., ge=0, le=100)
    dry_run: bool = False
    components: Optional[List[str]] = None


class RewindToWaypointRequest(BaseModel):
    waypoint_idx: int = Field(..., ge=0)
    dry_run: bool = False
    components: Optional[List[str]] = None


class RewindConfigUpdate(BaseModel):
    settle_time: Optional[float] = Field(None, ge=0)
    safety_margin: Optional[float] = Field(None, ge=0)
    rewind_base: Optional[bool] = None
    rewind_arm: Optional[bool] = None
    rewind_gripper: Optional[bool] = None
    chunk_size: Optional[int] = Field(None, ge=1, le=50)  # Waypoints per chunk
    chunk_duration: Optional[float] = Field(None, ge=0.1, le=10.0)  # Seconds per chunk


class MonitorConfigUpdate(BaseModel):
    auto_rewind_enabled: Optional[bool] = None
    auto_rewind_percentage: Optional[float] = None
    manual_rewind_percentage: Optional[float] = None
    monitor_interval: Optional[float] = None
    collision_arm_threshold: Optional[float] = Field(None, ge=0, le=1)
    collision_trigger_threshold: Optional[float] = Field(None, ge=0, le=1)
    collision_min_cmd_speed: Optional[float] = Field(None, ge=0)
    collision_grace_period: Optional[float] = Field(None, ge=0)


class ResetToHomeRequest(BaseModel):
    dry_run: bool = False
    components: Optional[List[str]] = None


def _format_result(result) -> dict:
    """Format RewindResult to dict."""
    return {
        "success": result.success,
        "steps_rewound": result.steps_rewound,
        "start_waypoint_idx": result.start_waypoint_idx,
        "end_waypoint_idx": result.end_waypoint_idx,
        "waypoints_executed": result.waypoints_executed,
        "components_rewound": result.components_rewound,
        "error": result.error,
    }


def create_router(rewind_orchestrator, lease_mgr, system_logger, safety_monitor=None, arm_monitor=None, state_agg=None):
    """Create rewind routes with injected dependencies."""

    def _check_lease(lease_id: Optional[str]) -> None:
        if lease_id is None:
            raise HTTPException(status_code=401, detail="Lease required")
        if not lease_mgr.validate_lease(lease_id):
            raise HTTPException(status_code=403, detail="Invalid lease")

    @router.get("/status")
    async def get_status():
        """Get rewind status and trajectory info."""
        try:
            boundary = rewind_orchestrator.get_boundary_status()
        except Exception as e:
            boundary = {"error": str(e)}

        result = {
            "is_rewinding": rewind_orchestrator.is_rewinding,
            "trajectory_length": rewind_orchestrator.trajectory_length,
            "trajectory_info": system_logger.get_trajectory_info(),
            "base_boundary_status": boundary,  # Dashboard expects this name
            "collision_detected": False,
        }
        if safety_monitor is not None:
            result["collision_detected"] = safety_monitor.collision_detected
        if arm_monitor is not None:
            result["arm_recovering"] = arm_monitor.is_recovering
        return result

    @router.get("/logs", include_in_schema=False)
    async def get_logs(limit: int = 50):
        """Get recent rewind logs for dashboard display.

        Args:
            limit: Maximum number of log entries to return (default 50).

        Returns:
            List of log entries with timestamp, level, and message.
        """
        from system_logger import get_rewind_log_buffer
        log_buffer = get_rewind_log_buffer()
        return {"logs": log_buffer.get_logs(limit)}

    @router.get("/trajectory")
    async def get_trajectory():
        """Get trajectory info."""
        info = system_logger.get_trajectory_info()
        info["last_safe_waypoint_idx"] = rewind_orchestrator.find_last_safe_waypoint()
        return info

    @router.post("/trajectory/clear", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def clear_trajectory():
        """Clear all recorded trajectory waypoints."""
        system_logger.clear()
        return {"success": True, "message": "Trajectory cleared"}

    @router.get("/trajectory/{idx}", include_in_schema=False)
    async def get_waypoint(idx: int):
        """Get waypoint by index."""
        wp = system_logger.get_waypoint(idx)
        if wp is None:
            raise HTTPException(status_code=404, detail=f"Waypoint {idx} not found")
        return wp.to_dict()

    @router.get("/boundary", include_in_schema=False)
    async def get_boundary():
        """Get boundary status."""
        try:
            return rewind_orchestrator.get_boundary_status()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/check", include_in_schema=False)
    async def check_bounds():
        """Quick out-of-bounds check."""
        try:
            oob = rewind_orchestrator.is_base_out_of_bounds()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

        result = {"out_of_bounds": oob}
        if oob:
            safe_idx = rewind_orchestrator.find_last_safe_waypoint()
            result["last_safe_waypoint_idx"] = safe_idx
            if safe_idx is not None:
                result["steps_to_safe"] = rewind_orchestrator.trajectory_length - 1 - safe_idx
        return result

    @router.post("/steps")
    async def rewind_steps(
        req: RewindStepsRequest,
        x_lease_id: Optional[str] = Header(None, alias="X-Lease-Id"),
    ):
        """Rewind by N steps."""
        if not req.dry_run:
            _check_lease(x_lease_id)
        result = await rewind_orchestrator.rewind_steps(
            req.steps, dry_run=req.dry_run, components=req.components
        )
        return _format_result(result)

    @router.post("/percentage")
    async def rewind_percentage(
        req: RewindPercentageRequest,
        x_lease_id: Optional[str] = Header(None, alias="X-Lease-Id"),
    ):
        """Rewind by percentage."""
        if not req.dry_run:
            _check_lease(x_lease_id)
        result = await rewind_orchestrator.rewind_percentage(
            req.percentage, dry_run=req.dry_run, components=req.components
        )
        return _format_result(result)

    @router.post("/to-safe")
    async def rewind_to_safe(
        req: RewindStepsRequest = RewindStepsRequest(steps=1),
        x_lease_id: Optional[str] = Header(None, alias="X-Lease-Id"),
    ):
        """Rewind to last safe waypoint."""
        if not req.dry_run:
            _check_lease(x_lease_id)
        result = await rewind_orchestrator.rewind_to_safe(
            dry_run=req.dry_run, components=req.components
        )
        return _format_result(result)

    @router.post("/to-waypoint")
    async def rewind_to_waypoint(
        req: RewindToWaypointRequest,
        x_lease_id: Optional[str] = Header(None, alias="X-Lease-Id"),
    ):
        """Rewind to specific waypoint."""
        if not req.dry_run:
            _check_lease(x_lease_id)
        result = await rewind_orchestrator.rewind_to_waypoint(
            req.waypoint_idx, dry_run=req.dry_run, components=req.components
        )
        return _format_result(result)

    @router.post("/reset-to-home")
    async def reset_to_home(
        req: ResetToHomeRequest = ResetToHomeRequest(),
        x_lease_id: Optional[str] = Header(None, alias="X-Lease-Id"),
    ):
        """Full 100% rewind."""
        if not req.dry_run:
            _check_lease(x_lease_id)
        result = await rewind_orchestrator.reset_to_home(
            dry_run=req.dry_run, components=req.components
        )
        return _format_result(result)

    @router.get("/config")
    async def get_config():
        """Get rewind config."""
        cfg = rewind_orchestrator.config
        return {
            "settle_time": cfg.settle_time,
            "command_rate": cfg.command_rate,
            "safety_margin": cfg.safety_margin,
            "rewind_base": cfg.rewind_base,
            "rewind_arm": cfg.rewind_arm,
            "rewind_gripper": cfg.rewind_gripper,
            "chunk_size": cfg.chunk_size,
            "chunk_duration": cfg.chunk_duration,
        }

    @router.put("/config", include_in_schema=False,
                dependencies=[Depends(require_admin)])
    async def update_config(req: RewindConfigUpdate):
        """Update rewind config.

        Tuning tips for chunk_size and chunk_duration:
        - chunk_size: Number of waypoints per chunk (default 5)
          - Smaller = more frequent base sync, slightly more jittery
          - Larger = smoother arm motion, but base might lag
        - chunk_duration: Seconds to execute each chunk (default 1.0)
          - Shorter = faster rewind, might be jerky
          - Longer = smoother motion, slower overall
        """
        cfg = rewind_orchestrator.config
        if req.settle_time is not None:
            cfg.settle_time = req.settle_time
        if req.safety_margin is not None:
            cfg.safety_margin = req.safety_margin
        if req.rewind_base is not None:
            cfg.rewind_base = req.rewind_base
        if req.rewind_arm is not None:
            cfg.rewind_arm = req.rewind_arm
        if req.rewind_gripper is not None:
            cfg.rewind_gripper = req.rewind_gripper
        if req.chunk_size is not None:
            cfg.chunk_size = req.chunk_size
        if req.chunk_duration is not None:
            cfg.chunk_duration = req.chunk_duration
        return await get_config()

    # -------------------------------------------------------------------------
    # Monitor endpoints (for dashboard compatibility)
    # -------------------------------------------------------------------------

    # Store manual rewind percentage in config
    _manual_rewind_pct = {"value": 5.0}

    @router.get("/monitor/status")
    async def get_monitor_status():
        """Get safety monitor status (dashboard compatibility)."""
        cfg = rewind_orchestrator.config
        result = {
            "is_running": True,
            "auto_rewind_enabled": cfg.auto_rewind_enabled,
            "auto_rewind_percentage": cfg.auto_rewind_percentage,
            "manual_rewind_percentage": _manual_rewind_pct["value"],
            "monitor_interval": cfg.monitor_interval,
            "auto_rewind_count": 0,
            "last_auto_rewind_time": None,
            "is_currently_rewinding": rewind_orchestrator.is_rewinding,
            "collision_detected": False,
            "collision_arm_threshold": cfg.collision_arm_threshold,
            "collision_trigger_threshold": cfg.collision_trigger_threshold,
            "collision_min_cmd_speed": cfg.collision_min_cmd_speed,
            "collision_grace_period": cfg.collision_grace_period,
            "manual_reset_required": False,
            "manual_reset_reason": "",
        }
        if safety_monitor is not None:
            result["auto_rewind_count"] = safety_monitor.auto_rewind_count
            result["last_auto_rewind_time"] = safety_monitor.last_auto_rewind_time
            result["collision_detected"] = safety_monitor.collision_detected
            result["manual_reset_required"] = safety_monitor.manual_reset_required
        if arm_monitor is not None:
            result["arm_monitor"] = arm_monitor.get_status()
        return result

    @router.put("/monitor/config", include_in_schema=False,
                dependencies=[Depends(require_admin)])
    async def update_monitor_config(req: MonitorConfigUpdate):
        """Update monitor config (dashboard compatibility)."""
        cfg = rewind_orchestrator.config
        if req.auto_rewind_enabled is not None:
            if req.auto_rewind_enabled and state_agg is not None:
                base_info = state_agg.state.get("base", {})
                if base_info.get("pose_source") != "mocap":
                    raise HTTPException(
                        status_code=409,
                        detail="Cannot enable auto-rewind without mocap tracking.",
                    )
            cfg.auto_rewind_enabled = req.auto_rewind_enabled
        if req.auto_rewind_percentage is not None:
            cfg.auto_rewind_percentage = req.auto_rewind_percentage
        if req.manual_rewind_percentage is not None:
            _manual_rewind_pct["value"] = req.manual_rewind_percentage
        if req.monitor_interval is not None:
            cfg.monitor_interval = req.monitor_interval
        if req.collision_arm_threshold is not None:
            cfg.collision_arm_threshold = req.collision_arm_threshold
        if req.collision_trigger_threshold is not None:
            cfg.collision_trigger_threshold = req.collision_trigger_threshold
        if req.collision_min_cmd_speed is not None:
            cfg.collision_min_cmd_speed = req.collision_min_cmd_speed
        if req.collision_grace_period is not None:
            cfg.collision_grace_period = req.collision_grace_period
        return await get_monitor_status()

    @router.post("/monitor/enable", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def enable_auto_rewind():
        """Enable auto-rewind. Requires mocap tracking (odom drifts)."""
        if state_agg is not None:
            base_info = state_agg.state.get("base", {})
            if base_info.get("pose_source") != "mocap":
                raise HTTPException(
                    status_code=409,
                    detail="Cannot enable auto-rewind without mocap tracking. "
                           "Odom drifts and boundary checks would be unreliable.",
                )
        rewind_orchestrator.config.auto_rewind_enabled = True
        return {"auto_rewind_enabled": True}

    @router.post("/monitor/disable", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def disable_auto_rewind():
        """Disable auto-rewind."""
        rewind_orchestrator.config.auto_rewind_enabled = False
        return {"auto_rewind_enabled": False}

    @router.post("/monitor/reset", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def reset_safety_monitor():
        """Clear manual-reset-required state (admin only)."""
        if safety_monitor is None:
            raise HTTPException(status_code=503, detail="Safety monitor not available")
        safety_monitor.reset_manual()
        return {"manual_reset_required": False}

    @router.post("/manual")
    async def trigger_manual_rewind(
        dry_run: bool = False,
        x_lease_id: Optional[str] = Header(None, alias="X-Lease-Id"),
    ):
        """Manual rewind using configured percentage."""
        if not dry_run:
            _check_lease(x_lease_id)
        result = await rewind_orchestrator.rewind_percentage(
            _manual_rewind_pct["value"], dry_run=dry_run
        )
        return _format_result(result)

    return router

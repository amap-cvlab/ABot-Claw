"""API routes for workspace boundary teaching."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth import require_admin

router = APIRouter(prefix="/workspace", tags=["workspace"])


class StopTeachRequest(BaseModel):
    margin: float = Field(0.0, ge=0.0, description="Extra margin to expand hull (meters)")
    save: bool = True


def create_router(workspace_teacher):
    """Create workspace routes with injected WorkspaceTeacher.

    Args:
        workspace_teacher: WorkspaceTeacher instance.
    """

    @router.post("/teach/start", dependencies=[Depends(require_admin)])
    async def teach_start():
        """Start recording base positions for workspace boundary."""
        return await workspace_teacher.start_teaching()

    @router.post("/teach/stop", dependencies=[Depends(require_admin)])
    async def teach_stop(req: StopTeachRequest = StopTeachRequest()):
        """Stop recording, compute convex hull, update and save bounds."""
        return await workspace_teacher.stop_teaching(margin=req.margin, save=req.save)

    @router.get("/teach/status")
    async def teach_status():
        """Get teaching status and current bounds info."""
        return workspace_teacher.get_status()

    @router.get("/bounds")
    async def get_bounds():
        """Get current workspace boundary (hull or AABB)."""
        return workspace_teacher.get_status()

    @router.post("/bounds/reset", dependencies=[Depends(require_admin)])
    async def reset_bounds():
        """Clear hull boundary, revert to AABB, delete saved file."""
        workspace_teacher._bounds.clear_hull()
        deleted = workspace_teacher.delete_saved_bounds()
        return {
            "ok": True,
            "message": "Hull cleared, reverted to AABB",
            "file_deleted": deleted,
            "bounds": workspace_teacher._bounds.to_dict(),
        }

    return router

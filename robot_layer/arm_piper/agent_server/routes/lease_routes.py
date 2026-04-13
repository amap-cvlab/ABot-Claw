"""Lease management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth import require_admin

router = APIRouter(prefix="/lease")


class AcquireRequest(BaseModel):
    holder: str
    rewind_on_release: bool = False  # if True, rewind trajectory before going home on release


class LeaseIdRequest(BaseModel):
    lease_id: str


def create_router(lease_mgr):
    @router.post("/acquire")
    async def acquire(req: AcquireRequest):
        return await lease_mgr.acquire(req.holder, rewind_on_release=req.rewind_on_release)

    @router.post("/release")
    async def release(req: LeaseIdRequest):
        return await lease_mgr.release(req.lease_id)

    @router.post("/extend")
    async def extend(req: LeaseIdRequest):
        return await lease_mgr.extend(req.lease_id)

    @router.get("/status")
    async def status():
        return lease_mgr.status()

    @router.get("/queue/{ticket_id}")
    async def check_ticket(ticket_id: str):
        """Check the status and position of a queue ticket."""
        return lease_mgr.check_ticket(ticket_id)

    @router.delete("/queue/{ticket_id}")
    async def cancel_ticket(ticket_id: str):
        """Cancel a queue ticket (leave the queue)."""
        return lease_mgr.cancel_ticket(ticket_id)

    @router.post("/clear-queue", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def clear_queue():
        return await lease_mgr.clear_queue()

    @router.post("/pause-queue", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def pause_queue():
        return await lease_mgr.pause_queue()

    @router.post("/resume-queue", include_in_schema=False,
                 dependencies=[Depends(require_admin)])
    async def resume_queue():
        return await lease_mgr.resume_queue()

    return router

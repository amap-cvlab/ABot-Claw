"""Lease manager — non-blocking ticket queue, acquire, release, idle detection."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from config import LeaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Lease:
    lease_id: str
    holder: str  # client identifier
    granted_at: float
    last_cmd_at: float
    rewind_on_release: bool = False  # if True, rewind trajectory before going home


@dataclass
class QueueTicket:
    ticket_id: str
    holder: str
    created_at: float
    status: str = "waiting"          # "waiting" | "granted" | "expired" | "cancelled"
    lease_id: str | None = None      # set when auto-granted
    granted_at: float | None = None  # when status changed to "granted"
    rewind_on_release: bool = False  # forwarded to Lease when granted


class LeaseManager:
    """Manages operator lease with idle detection and non-blocking ticket queue."""

    def __init__(
        self,
        config: LeaseConfig,
        last_moved_at_fn: Callable[[], float],
    ) -> None:
        self._cfg = config
        self._last_moved_at = last_moved_at_fn
        self._current: Lease | None = None
        self._queue: deque[QueueTicket] = deque()
        self._tickets: dict[str, QueueTicket] = {}  # all tickets by ID
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._paused: bool = False

        # Reset-on-release state
        self._resetting: bool = False
        self._reset_task: asyncio.Task | None = None
        self._on_lease_end_async: Callable[[bool], Awaitable[None]] | None = None
        self._on_lease_start: Callable[[], None] | None = None

    @property
    def current_lease(self) -> Lease | None:
        return self._current

    def set_on_lease_end(self, callback: Callable[[bool], Awaitable[None]]) -> None:
        """Set async callback invoked when a lease ends.

        Args:
            callback: Called with ``rewind`` bool — True to rewind trajectory
                before going home, False to go home directly.
        """
        self._on_lease_end_async = callback

    def set_on_lease_start(self, callback: Callable[[str], None]) -> None:
        """Set callback invoked when a lease is granted.

        Args:
            callback: Called with the holder name as argument.
        """
        self._on_lease_start = callback

    async def start(self) -> None:
        self._task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel()
            try:
                await self._reset_task
            except asyncio.CancelledError:
                pass
            self._resetting = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Mark all waiting tickets as cancelled
        async with self._lock:
            for ticket in list(self._queue):
                ticket.status = "cancelled"
            self._queue.clear()

    async def acquire(self, holder: str, rewind_on_release: bool = False) -> dict:
        async with self._lock:
            # Lease free → grant immediately
            if self._current is None and not self._resetting and not self._paused:
                return self._grant(holder, rewind_on_release=rewind_on_release)
            # Already holder?
            if self._current and self._current.holder == holder:
                return {
                    "status": "already_held",
                    "lease_id": self._current.lease_id,
                    "remaining_s": self._remaining(),
                    "max_duration_s": self._cfg.max_duration_s,
                    "idle_timeout_s": self._cfg.idle_timeout_s,
                }
            # Same holder already waiting → return existing ticket (idempotent)
            for i, ticket in enumerate(self._queue):
                if ticket.holder == holder and ticket.status == "waiting":
                    return {
                        "status": "queued",
                        "ticket_id": ticket.ticket_id,
                        "position": i + 1,
                        "queue_length": len(self._queue),
                    }
            # New holder → create ticket, return immediately
            ticket = QueueTicket(
                ticket_id=str(uuid.uuid4()),
                holder=holder,
                created_at=time.time(),
                rewind_on_release=rewind_on_release,
            )
            self._queue.append(ticket)
            self._tickets[ticket.ticket_id] = ticket
            position = len(self._queue)
            logger.info("Queued %s at position %d (ticket %s)",
                        holder, position, ticket.ticket_id)
            return {
                "status": "queued",
                "ticket_id": ticket.ticket_id,
                "position": position,
                "queue_length": len(self._queue),
            }

    def check_ticket(self, ticket_id: str) -> dict:
        """Check the status of a queue ticket."""
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return {"status": "not_found"}
        if ticket.status == "granted":
            return {
                "status": "granted",
                "lease_id": ticket.lease_id,
                "holder": ticket.holder,
                "max_duration_s": self._cfg.max_duration_s,
                "idle_timeout_s": self._cfg.idle_timeout_s,
            }
        if ticket.status == "waiting":
            # Find position in queue
            position = None
            for i, t in enumerate(self._queue):
                if t.ticket_id == ticket_id:
                    position = i + 1
                    break
            return {
                "status": "waiting",
                "holder": ticket.holder,
                "position": position,
                "queue_length": len(self._queue),
            }
        # cancelled or expired
        return {"status": ticket.status, "holder": ticket.holder}

    def cancel_ticket(self, ticket_id: str) -> dict:
        """Cancel a queue ticket (leave queue)."""
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return {"status": "not_found"}
        if ticket.status == "granted":
            return {
                "status": "error",
                "message": "Ticket already granted — use POST /lease/release instead",
            }
        if ticket.status != "waiting":
            return {"status": ticket.status}
        # Remove from queue
        ticket.status = "cancelled"
        self._queue = deque(t for t in self._queue if t.ticket_id != ticket_id)
        logger.info("Cancelled ticket %s for %s", ticket_id, ticket.holder)
        return {"status": "cancelled"}

    async def release(self, lease_id: str) -> dict:
        async with self._lock:
            if self._current and self._current.lease_id == lease_id:
                rewind = self._current.rewind_on_release
                self._current = None
                if self._cfg.reset_on_release and self._on_lease_end_async:
                    self._resetting = True
                    self._reset_task = asyncio.create_task(
                        self._do_reset_and_grant(reason="released", rewind=rewind)
                    )
                    return {"status": "released", "resetting": True}
                else:
                    self._try_grant_next()
                    return {"status": "released", "resetting": False}
            return {"status": "not_found"}

    async def extend(self, lease_id: str) -> dict:
        async with self._lock:
            if self._current and self._current.lease_id == lease_id:
                self._current.last_cmd_at = time.time()
                return {"status": "extended", "remaining_s": self._remaining()}
            return {"status": "not_found"}

    async def clear_queue(self) -> dict:
        """Clear queue, revoke current lease, stop code, and trigger rewind."""
        async with self._lock:
            # Mark all queued tickets as cancelled
            removed = 0
            while self._queue:
                ticket = self._queue.popleft()
                ticket.status = "cancelled"
                removed += 1

            had_lease = self._current is not None
            if had_lease:
                self._revoke("queue_cleared")

            logger.info("Cleared queue (%d removed), revoked lease: %s",
                         removed, had_lease)
            return {
                "status": "cleared",
                "removed": removed,
                "lease_revoked": had_lease,
                "resetting": self._resetting,
            }

    def record_command(self) -> None:
        """Called when operator sends a command."""
        if self._current:
            self._current.last_cmd_at = time.time()

    def validate_lease(self, lease_id: str) -> bool:
        return self._current is not None and self._current.lease_id == lease_id

    def status(self) -> dict:
        # Build queue list with ticket IDs
        queue_list = [
            {"position": i + 1, "holder": t.holder, "ticket_id": t.ticket_id}
            for i, t in enumerate(self._queue)
        ]

        config = {
            "max_duration_s": self._cfg.max_duration_s,
            "idle_timeout_s": self._cfg.idle_timeout_s,
            "reset_on_release": self._cfg.reset_on_release,
        }

        if self._current is None:
            return {
                "holder": None,
                "queue_length": len(self._queue),
                "queue": queue_list,
                "resetting": self._resetting,
                "paused": self._paused,
                "config": config,
            }
        return {
            "holder": self._current.holder,
            "remaining_s": self._remaining(),
            "queue_length": len(self._queue),
            "queue": queue_list,
            "resetting": self._resetting,
            "paused": self._paused,
            "config": config,
        }

    async def pause_queue(self) -> dict:
        """Pause queue progression — no queued holders will be granted."""
        async with self._lock:
            self._paused = True
            logger.info("Lease queue paused")
            return {"status": "paused"}

    async def resume_queue(self) -> dict:
        """Resume queue progression and grant next if nobody holds the lease."""
        async with self._lock:
            self._paused = False
            logger.info("Lease queue resumed")
            if self._current is None and not self._resetting:
                self._try_grant_next()
            return {"status": "resumed"}

    # -- internals -----------------------------------------------------------

    def _grant(self, holder: str, rewind_on_release: bool = False) -> dict:
        now = time.time()
        lease = Lease(
            lease_id=str(uuid.uuid4()),
            holder=holder,
            granted_at=now,
            last_cmd_at=now,
            rewind_on_release=rewind_on_release,
        )
        self._current = lease
        if self._on_lease_start:
            self._on_lease_start()
        logger.info("Lease granted to %s (%s)", holder, lease.lease_id)
        return {
            "status": "granted",
            "type": "lease_granted",
            "lease_id": lease.lease_id,
            "max_duration_s": self._cfg.max_duration_s,
            "idle_timeout_s": self._cfg.idle_timeout_s,
        }

    def _remaining(self) -> float:
        if not self._current:
            return 0.0
        elapsed = time.time() - self._current.granted_at
        return max(0.0, self._cfg.max_duration_s - elapsed)

    def _try_grant_next(self) -> None:
        if self._paused:
            return
        while self._queue:
            ticket = self._queue.popleft()
            if ticket.status == "waiting":
                result = self._grant(ticket.holder, rewind_on_release=ticket.rewind_on_release)
                ticket.status = "granted"
                ticket.lease_id = result["lease_id"]
                ticket.granted_at = time.time()
                return
        # Queue empty — nothing to grant

    def _revoke(self, reason: str) -> None:
        if not self._current:
            return
        rewind = self._current.rewind_on_release
        logger.info("Lease revoked from %s: %s", self._current.holder, reason)
        self._current = None
        if self._cfg.reset_on_release and self._on_lease_end_async:
            self._resetting = True
            self._reset_task = asyncio.create_task(
                self._do_reset_and_grant(reason=reason, rewind=rewind)
            )
        else:
            self._try_grant_next()

    async def _do_reset_and_grant(self, reason: str = "released", rewind: bool = False) -> None:
        """Reset robot to home, optionally rewinding trajectory first."""
        try:
            logger.info("Lease ended — resetting robot to home (reason: %s, rewind: %s)", reason, rewind)

            # Stop any running code execution
            try:
                from routes.code_routes import get_executor
                executor = get_executor()
                if executor.is_running:
                    logger.info("Stopping running code execution before reset (reason: %s)", reason)
                    executor.stop(reason=reason)
            except Exception as e:
                logger.warning("Failed to stop code executor: %s", e)

            # Perform reset (rewind + go home, or just go home)
            await self._on_lease_end_async(rewind)

            logger.info("Reset to home complete")
        except asyncio.CancelledError:
            logger.info("Reset to home cancelled")
            raise
        except Exception as e:
            logger.error("Reset to home failed: %s", e)
        finally:
            async with self._lock:
                self._resetting = False
                self._try_grant_next()

    async def _check_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cfg.check_interval_s)
            async with self._lock:
                # Clean up stale tickets (granted/cancelled older than TTL)
                now = time.time()
                stale_ids = [
                    tid for tid, t in self._tickets.items()
                    if t.status in ("granted", "cancelled", "expired")
                    and (t.granted_at or t.created_at) + self._cfg.ticket_ttl_s < now
                ]
                for tid in stale_ids:
                    del self._tickets[tid]

                if not self._current or self._resetting:
                    continue

                # Hard max duration
                if now - self._current.granted_at >= self._cfg.max_duration_s:
                    self._revoke("max_duration")
                    continue

                # Idle check — revoke immediately after timeout
                last_activity = max(self._current.last_cmd_at, self._last_moved_at())
                idle_s = now - last_activity

                if idle_s >= self._cfg.idle_timeout_s:
                    self._revoke("idle_timeout")

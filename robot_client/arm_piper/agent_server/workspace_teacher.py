"""Workspace boundary teaching — record base positions, compute convex hull."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from system_logger.config import WorkspaceBounds, convex_hull_2d

logger = logging.getLogger(__name__)


class WorkspaceTeacher:
    """Records base positions while the user pushes the robot around,
    then computes a convex hull to define the workspace boundary.

    Usage:
        teacher = WorkspaceTeacher(state_fn, workspace_bounds)
        teacher.load_bounds()          # Restore saved hull on startup

        await teacher.start_teaching() # Begin recording
        # ... user pushes robot ...
        await teacher.stop_teaching()  # Compute hull, update bounds, save
    """

    RECORD_HZ = 10.0
    MIN_POSITION_CHANGE = 0.01  # 1 cm dedup threshold

    def __init__(
        self,
        state_fn: Callable[[], Dict[str, Any]],
        workspace_bounds: WorkspaceBounds,
        save_path: str = "workspace_bounds.json",
    ) -> None:
        self._state_fn = state_fn
        self._bounds = workspace_bounds
        self._save_path = save_path

        # Teaching state
        self._is_teaching = False
        self._recorded_points: List[List[float]] = []
        self._task: Optional[asyncio.Task] = None
        self._teach_start_time: Optional[float] = None

    @property
    def is_teaching(self) -> bool:
        return self._is_teaching

    # ------------------------------------------------------------------
    # Teaching flow
    # ------------------------------------------------------------------

    async def start_teaching(self) -> dict:
        """Begin recording base [x, y] positions at 10 Hz."""
        if self._is_teaching:
            return {"ok": False, "error": "Already teaching"}

        self._recorded_points = []
        self._is_teaching = True
        self._teach_start_time = time.time()
        self._task = asyncio.create_task(self._record_loop())

        logger.info("Workspace teaching started")
        return {"ok": True, "message": "Teaching started — push the robot around the workspace boundary"}

    async def stop_teaching(self, margin: float = 0.0, save: bool = True) -> dict:
        """Stop recording, compute convex hull, update bounds.

        Args:
            margin: Extra margin to expand hull outward (meters). 0 = exact hull.
            save: Whether to persist bounds to disk.

        Returns:
            Result dict with hull vertices and stats.
        """
        if not self._is_teaching:
            return {"ok": False, "error": "Not currently teaching"}

        self._is_teaching = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        n_points = len(self._recorded_points)
        duration = time.time() - (self._teach_start_time or time.time())

        if n_points < 3:
            logger.warning("Workspace teaching stopped with only %d points (need >= 3)", n_points)
            return {
                "ok": False,
                "error": f"Need at least 3 unique positions, got {n_points}",
                "point_count": n_points,
            }

        # Compute convex hull
        hull = convex_hull_2d(self._recorded_points)

        if len(hull) < 3:
            return {
                "ok": False,
                "error": f"Hull has only {len(hull)} vertices (points may be collinear)",
                "point_count": n_points,
            }

        # Expand hull by margin if requested
        if margin > 0:
            hull = self._expand_hull(hull, margin)

        # Compute area (shoelace formula)
        area = self._polygon_area(hull)

        # Update shared bounds object
        self._bounds.set_hull(hull)

        logger.info(
            "Workspace teaching complete: %d points -> %d hull vertices, %.2f m^2",
            n_points, len(hull), area,
        )

        if save:
            self.save_bounds(point_count=n_points, area=area)

        return {
            "ok": True,
            "hull_vertices": hull,
            "point_count": n_points,
            "hull_vertex_count": len(hull),
            "area_m2": round(area, 2),
            "duration_s": round(duration, 1),
            "bounds": self._bounds.to_dict(),
        }

    def get_status(self) -> dict:
        """Get teaching status and current bounds info."""
        result: dict = {
            "is_teaching": self._is_teaching,
            "point_count": len(self._recorded_points),
            "has_hull": self._bounds.has_hull,
            "boundary_type": "hull" if self._bounds.has_hull else "aabb",
            "bounds": self._bounds.to_dict(),
        }
        if self._is_teaching and self._teach_start_time:
            result["teach_duration_s"] = round(time.time() - self._teach_start_time, 1)
        if self._bounds.has_hull:
            result["hull_vertices"] = self._bounds.hull_vertices
            result["hull_vertex_count"] = len(self._bounds.hull_vertices)
            result["area_m2"] = round(self._polygon_area(self._bounds.hull_vertices), 2)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_bounds(self, point_count: int = 0, area: float = 0.0) -> None:
        """Save current hull bounds to JSON file."""
        if not self._bounds.has_hull:
            logger.warning("No hull to save")
            return

        data = self._bounds.to_dict()
        data["saved_at"] = datetime.now(timezone.utc).isoformat()
        data["point_count"] = point_count
        data["area_m2"] = round(area, 2)

        try:
            with open(self._save_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Workspace bounds saved to %s", self._save_path)
        except Exception as e:
            logger.error("Failed to save workspace bounds: %s", e)

    def load_bounds(self) -> bool:
        """Load hull bounds from JSON file. Returns True if loaded."""
        if not os.path.exists(self._save_path):
            return False

        try:
            with open(self._save_path, "r") as f:
                data = json.load(f)

            hull = data.get("hull_vertices")
            if hull and len(hull) >= 3:
                self._bounds.set_hull(hull)
                area = data.get("area_m2", 0)
                logger.info(
                    "Loaded workspace hull from %s: %d vertices, %.2f m^2",
                    self._save_path, len(hull), area,
                )
                return True
            else:
                logger.warning("Saved bounds file has no valid hull")
                return False
        except Exception as e:
            logger.error("Failed to load workspace bounds: %s", e)
            return False

    def delete_saved_bounds(self) -> bool:
        """Delete saved bounds file from disk."""
        if os.path.exists(self._save_path):
            try:
                os.remove(self._save_path)
                logger.info("Deleted saved workspace bounds: %s", self._save_path)
                return True
            except Exception as e:
                logger.error("Failed to delete bounds file: %s", e)
                return False
        return False

    # ------------------------------------------------------------------
    # Recording loop
    # ------------------------------------------------------------------

    async def _record_loop(self) -> None:
        """Poll base position at RECORD_HZ, deduplicate by MIN_POSITION_CHANGE."""
        interval = 1.0 / self.RECORD_HZ

        while self._is_teaching:
            try:
                state = self._state_fn()
                base = state.get("base", {})
                pose = base.get("pose", [0, 0, 0])
                x, y = pose[0], pose[1]

                # Dedup: skip if too close to last recorded point
                if self._recorded_points:
                    last = self._recorded_points[-1]
                    dist = math.hypot(x - last[0], y - last[1])
                    if dist < self.MIN_POSITION_CHANGE:
                        await asyncio.sleep(interval)
                        continue

                self._recorded_points.append([x, y])
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Don't crash on transient state errors

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_area(vertices: List[List[float]]) -> float:
        """Compute area of a polygon using the shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _expand_hull(vertices: List[List[float]], margin: float) -> List[List[float]]:
        """Expand a convex hull outward by margin (offset each edge outward)."""
        n = len(vertices)
        if n < 3 or margin <= 0:
            return vertices

        # Compute centroid
        cx = sum(v[0] for v in vertices) / n
        cy = sum(v[1] for v in vertices) / n

        # Push each vertex outward from centroid
        expanded = []
        for v in vertices:
            dx = v[0] - cx
            dy = v[1] - cy
            dist = math.hypot(dx, dy)
            if dist > 0:
                scale = (dist + margin) / dist
                expanded.append([cx + dx * scale, cy + dy * scale])
            else:
                expanded.append(list(v))

        return expanded

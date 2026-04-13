from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from app.storage import SqliteStore


class TaskService:
    def __init__(self, store: SqliteStore):
        self.store = store

    def create_offline_keyframe_task(
        self,
        input_uri: str,
        robot_id: str,
        robot_type: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        payload = {
            "task_id": task_id,
            "task_name": "offline_keyframe_pipeline",
            "status": "queued",
            "progress": 0.0,
            "input_uri": input_uri,
            "robot_id": robot_id,
            "robot_type": robot_type,
            "options": options,
            "result": {},
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self.store.create_task(payload)

        thread = threading.Thread(target=self._simulate_pipeline, args=(task_id,), daemon=True)
        thread.start()
        return {"task_id": task_id, "status": "queued"}

    def _simulate_pipeline(self, task_id: str) -> None:
        # Independent new module placeholder pipeline.
        stages = [
            ("export_rosbag", 0.2),
            ("extract_keyframes", 0.6),
            ("batch_ingest", 1.0),
        ]
        try:
            self.store.update_task(task_id, status="running", updated_at=time.time())
            for stage, progress in stages:
                time.sleep(0.8)
                self.store.update_task(
                    task_id,
                    progress=progress,
                    result_json={"stage": stage},
                    updated_at=time.time(),
                )
            self.store.update_task(
                task_id,
                status="completed",
                progress=1.0,
                result_json={"stage": "done", "message": "pipeline task completed"},
                updated_at=time.time(),
            )
        except Exception as exc:
            self.store.update_task(
                task_id,
                status="failed",
                error=str(exc),
                updated_at=time.time(),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.store.get_task(task_id)

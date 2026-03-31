from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


class SqliteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    robot_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    timestamp_ns INTEGER,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    roll REAL NOT NULL,
                    pitch REAL NOT NULL,
                    yaw REAL NOT NULL,
                    qx REAL,
                    qy REAL,
                    qz REAL,
                    qw REAL,
                    robot_pose_json TEXT,
                    tags_json TEXT,
                    note TEXT,
                    image_path TEXT,
                    confidence REAL NOT NULL,
                    embedding_json TEXT,
                    extra_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_name ON memories(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_robot ON memories(robot_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(timestamp)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    input_uri TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    robot_type TEXT NOT NULL,
                    options_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def insert_memory(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, memory_type, name, robot_id, robot_type, source, timestamp, timestamp_ns,
                    x, y, z, roll, pitch, yaw, qx, qy, qz, qw,
                    robot_pose_json, tags_json, note, image_path, confidence, embedding_json, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["memory_type"], payload["name"], payload["robot_id"],
                    payload["robot_type"], payload["source"], payload["timestamp"], payload.get("timestamp_ns"),
                    payload["x"], payload["y"], payload["z"], payload["roll"], payload["pitch"], payload["yaw"],
                    payload.get("qx"), payload.get("qy"), payload.get("qz"), payload.get("qw"),
                    json.dumps(payload.get("robot_pose") or {}),
                    json.dumps(payload.get("tags") or []),
                    payload.get("note", ""),
                    payload.get("image_path"),
                    payload.get("confidence", 1.0),
                    json.dumps(payload.get("embedding") or []),
                    json.dumps(payload.get("extra") or {}),
                ),
            )

    def query_memories(
        self,
        memory_type: Optional[str] = None,
        name: Optional[str] = None,
        robot_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        if name:
            sql += " AND lower(name) LIKE ?"
            params.append(f"%{name.lower()}%")
        if robot_id:
            sql += " AND robot_id = ?"
            params.append(robot_id)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def all_memories(self, memory_type: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memories"
        params: list[Any] = []
        if memory_type:
            sql += " WHERE memory_type = ?"
            params.append(memory_type)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def create_task(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, task_name, status, progress, input_uri, robot_id, robot_type,
                    options_json, result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["task_id"], payload["task_name"], payload["status"], payload["progress"],
                    payload["input_uri"], payload["robot_id"], payload["robot_type"],
                    json.dumps(payload.get("options") or {}),
                    json.dumps(payload.get("result") or {}),
                    payload.get("error"),
                    payload["created_at"], payload["updated_at"],
                ),
            )

    def update_task(self, task_id: str, **updates: Any) -> None:
        if not updates:
            return
        updates_sql = []
        params: list[Any] = []
        for key, value in updates.items():
            updates_sql.append(f"{key} = ?")
            if key in {"options_json", "result_json"} and not isinstance(value, str):
                params.append(json.dumps(value))
            else:
                params.append(value)
        params.append(task_id)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE tasks SET {', '.join(updates_sql)} WHERE task_id = ?", params)

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def count_memories(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
        return int(row["c"])

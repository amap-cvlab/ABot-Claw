from __future__ import annotations

import base64
import io
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from app.config import settings
from app.embeddings import cosine_similarity, decode_image_input, image_embedding, text_embedding
from app.schemas import (
    KeyframeBatchIngestRequest,
    MemoryResult,
    NavigationTarget,
    ObjectMemoryUpsertRequest,
    PlaceMemoryUpsertRequest,
    Pose,
    SemanticFrameIngestRequest,
    UnifiedQuery,
)
from app.storage import SqliteStore


class MemoryService:
    def __init__(self, store: SqliteStore):
        self.store = store
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.image_dir.mkdir(parents=True, exist_ok=True)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "records": self.store.count_memories(),
            "data_dir": str(settings.data_dir),
        }

    def _save_image(self, image: Image.Image, memory_id: str) -> str:
        path = settings.image_dir / f"{memory_id}.jpg"
        image.save(path, format="JPEG", quality=90)
        return str(path)

    def _pose_to_cols(self, pose: Pose) -> dict[str, Any]:
        return {
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "roll": pose.roll,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
            "qx": pose.qx,
            "qy": pose.qy,
            "qz": pose.qz,
            "qw": pose.qw,
        }

    def upsert_object(self, req: ObjectMemoryUpsertRequest) -> dict[str, Any]:
        ts = req.timestamp or time.time()
        memory_id = req.object_id or f"obj_{uuid.uuid4().hex[:12]}"

        image_path: Optional[str] = None
        emb = np.array([], dtype=np.float32)
        if req.image:
            image = decode_image_input(req.image)
            image_path = self._save_image(image, memory_id)
            emb = image_embedding(image, settings.embedding_dim)

        payload = {
            "id": memory_id,
            "memory_type": "object",
            "name": req.object_name,
            "robot_id": req.robot_id,
            "robot_type": req.robot_type,
            "source": req.source,
            "timestamp": ts,
            "timestamp_ns": int(ts * 1e9),
            **self._pose_to_cols(req.object_pose),
            "robot_pose": req.robot_pose.model_dump(),
            "tags": req.tags,
            "note": req.note,
            "image_path": image_path,
            "confidence": req.detect_confidence,
            "embedding": emb.tolist(),
            "extra": {
                "bbox_xyxy": req.bbox_xyxy,
            },
        }
        self.store.insert_memory(payload)
        return {"ok": True, "id": memory_id}

    def upsert_place(self, req: PlaceMemoryUpsertRequest) -> dict[str, Any]:
        ts = req.timestamp or time.time()
        memory_id = f"plc_{uuid.uuid4().hex[:12]}"
        payload = {
            "id": memory_id,
            "memory_type": "place",
            "name": req.place_name,
            "robot_id": req.robot_id,
            "robot_type": req.robot_type,
            "source": "human_label",
            "timestamp": ts,
            "timestamp_ns": int(ts * 1e9),
            **self._pose_to_cols(req.place_pose),
            "robot_pose": None,
            "tags": req.alias,
            "note": req.note,
            "image_path": None,
            "confidence": 1.0,
            "embedding": text_embedding(req.place_name, settings.embedding_dim).tolist(),
            "extra": {},
        }
        self.store.insert_memory(payload)
        return {"ok": True, "id": memory_id}

    def ingest_semantic_frame(self, req: SemanticFrameIngestRequest) -> dict[str, Any]:
        ts = req.timestamp or time.time()
        memory_id = f"sem_{uuid.uuid4().hex[:12]}"
        image = decode_image_input(req.image)
        image_path = self._save_image(image, memory_id)
        emb = image_embedding(image, settings.embedding_dim)

        payload = {
            "id": memory_id,
            "memory_type": "semantic_frame",
            "name": req.note or "semantic_frame",
            "robot_id": req.robot_id,
            "robot_type": req.robot_type,
            "source": req.source,
            "timestamp": ts,
            "timestamp_ns": int(ts * 1e9),
            **self._pose_to_cols(req.robot_pose),
            "robot_pose": req.robot_pose.model_dump(),
            "tags": req.tags,
            "note": req.note,
            "image_path": image_path,
            "confidence": 1.0,
            "embedding": emb.tolist(),
            "extra": {"task_id": req.task_id or ""},
        }
        self.store.insert_memory(payload)
        return {"ok": True, "id": memory_id, "image_path": image_path}

    def ingest_keyframe_batch(self, req: KeyframeBatchIngestRequest) -> dict[str, Any]:
        success = 0
        failed = 0
        for item in req.items:
            try:
                ts = item.timestamp or time.time()
                memory_id = f"kf_{uuid.uuid4().hex[:12]}"
                image = decode_image_input(item.image)
                image_path = self._save_image(image, memory_id)
                emb = image_embedding(image, settings.embedding_dim)

                payload = {
                    "id": memory_id,
                    "memory_type": "keyframe",
                    "name": f"keyframe_{item.camera_source}",
                    "robot_id": item.robot_id,
                    "robot_type": item.robot_type,
                    "source": item.camera_source,
                    "timestamp": ts,
                    "timestamp_ns": item.timestamp_ns or int(ts * 1e9),
                    **self._pose_to_cols(item.pose),
                    "robot_pose": None,
                    "tags": [],
                    "note": item.note,
                    "image_path": image_path,
                    "confidence": max(0.0, min(1.0, item.score)),
                    "embedding": emb.tolist(),
                    "extra": {"rank": item.rank, "task_id": req.task_id},
                }
                self.store.insert_memory(payload)
                success += 1
            except Exception:
                failed += 1
        return {"ok": True, "task_id": req.task_id, "success": success, "failed": failed}

    def _row_to_result(self, row: dict[str, Any], score: Optional[float] = None) -> MemoryResult:
        extra = json.loads(row.get("extra_json") or "{}")
        robot_pose_data = json.loads(row.get("robot_pose_json") or "{}")
        target_pose = Pose(
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            roll=float(row["roll"]),
            pitch=float(row["pitch"]),
            yaw=float(row["yaw"]),
            qx=row.get("qx"),
            qy=row.get("qy"),
            qz=row.get("qz"),
            qw=row.get("qw"),
            frame_id="map",
        )
        robot_pose = Pose(**robot_pose_data) if robot_pose_data else None
        final_conf = float(row.get("confidence", 1.0))
        if score is not None:
            final_conf = max(0.0, min(1.0, (final_conf + score) / 2.0))

        return MemoryResult(
            id=row["id"],
            memory_type=row["memory_type"],
            name=row["name"],
            robot_id=row["robot_id"],
            robot_type=row["robot_type"],
            target_pose=target_pose,
            robot_pose=robot_pose,
            source=row["source"],
            timestamp=float(row["timestamp"]),
            confidence=final_conf,
            evidence={
                "image_path": row.get("image_path"),
                "note": row.get("note") or "",
                "extra": extra,
            },
        )

    def query_by_name(self, memory_type: str, name: str, n_results: int, robot_id: Optional[str]) -> list[MemoryResult]:
        rows = self.store.query_memories(memory_type=memory_type, name=name, robot_id=robot_id, limit=n_results)
        return [self._row_to_result(row) for row in rows]

    def query_by_position(
        self,
        x: float,
        y: float,
        radius: float,
        n_results: int,
        memory_type: Optional[str] = None,
    ) -> list[MemoryResult]:
        rows = self.store.all_memories(memory_type=memory_type)
        hits: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            dx = float(row["x"]) - x
            dy = float(row["y"]) - y
            dist = math.hypot(dx, dy)
            if dist <= radius:
                hits.append((dist, row))

        hits.sort(key=lambda x: x[0])
        results = []
        for dist, row in hits[: max(1, n_results)]:
            score = 1.0 / (1.0 + dist)
            results.append(self._row_to_result(row, score=score))
        return results

    def semantic_text_query(self, text: str, n_results: int, memory_type: Optional[str]) -> list[MemoryResult]:
        q = text_embedding(text, settings.embedding_dim)
        rows = self.store.all_memories(memory_type=memory_type)
        scored: list[tuple[float, dict[str, Any]]] = []

        for row in rows:
            emb_raw = json.loads(row.get("embedding_json") or "[]")
            if not emb_raw:
                continue
            emb = np.array(emb_raw, dtype=np.float32)
            score = cosine_similarity(q, emb)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_result(row, score=score) for score, row in scored[: max(1, n_results)]]

    def unified_query(self, req: UnifiedQuery) -> list[MemoryResult]:
        if req.object_name:
            return self.query_by_name("object", req.object_name, req.n_results, req.robot_id)
        if req.place_name:
            return self.query_by_name("place", req.place_name, req.n_results, req.robot_id)
        if req.x is not None and req.y is not None:
            return self.query_by_position(req.x, req.y, req.radius, req.n_results, req.memory_type)
        if req.text:
            return self.semantic_text_query(req.text, req.n_results, req.memory_type)
        rows = self.store.query_memories(memory_type=req.memory_type, robot_id=req.robot_id, limit=req.n_results)
        return [self._row_to_result(row) for row in rows]

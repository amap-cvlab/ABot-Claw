from __future__ import annotations

import base64
import io
import os
import sys
import time
from typing import Any

import requests
from PIL import Image


BASE = os.getenv("SPATIAL_MEMORY_HUB_URL", "http://127.0.0.1:8022")
TIMEOUT = 10


def tiny_image_b64() -> str:
    img = Image.new("RGB", (16, 16), color=(220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def must_post(path: str, payload: dict[str, Any], timeout: int = TIMEOUT) -> dict[str, Any]:
    resp = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def must_get(path: str, timeout: int = TIMEOUT) -> dict[str, Any]:
    resp = requests.get(f"{BASE}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_step(name: str) -> None:
    print(f"\n[STEP] {name}")


def print_ok(name: str, data: Any) -> None:
    print(f"[OK] {name}: {data}")


def main() -> None:
    failures: list[str] = []
    image_b64 = tiny_image_b64()
    ts_seed = int(time.time())
    object_name = f"cup_smoke_{ts_seed}"
    place_name = f"kitchen_smoke_{ts_seed}"
    task_marker = f"smoke_task_{ts_seed}"

    object_id = None
    semantic_id = None

    checks = [
        "health",
        "object upsert",
        "place upsert",
        "semantic ingest",
        "keyframe batch ingest",
        "query object",
        "query place",
        "query position",
        "query semantic text",
        "query unified",
        "pipeline task create",
        "pipeline task status",
    ]

    for item in checks:
        try:
            if item == "health":
                print_step(item)
                data = must_get("/health")
                check(data.get("status") == "ok", "health status must be ok")
                print_ok(item, {"status": data.get("status"), "records": data.get("records")})

            elif item == "object upsert":
                print_step(item)
                payload = {
                    "object_name": object_name,
                    "robot_id": "humanoid_001",
                    "robot_type": "humanoid",
                    "robot_pose": {
                        "x": 1.0,
                        "y": 1.0,
                        "z": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.1,
                        "frame_id": "map",
                    },
                    "object_pose": {
                        "x": 1.2,
                        "y": 1.1,
                        "z": 0.8,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "frame_id": "map",
                    },
                    "detect_confidence": 0.92,
                    "image": image_b64,
                }
                data = must_post("/memory/object/upsert", payload)
                check(data.get("ok") is True, "object upsert must return ok=true")
                object_id = data.get("id")
                check(bool(object_id), "object upsert must return id")
                print_ok(item, data)

            elif item == "place upsert":
                print_step(item)
                payload = {
                    "place_name": place_name,
                    "robot_id": "humanoid_001",
                    "robot_type": "humanoid",
                    "place_pose": {
                        "x": 2.0,
                        "y": 3.0,
                        "z": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 1.57,
                        "frame_id": "map",
                    },
                    "alias": ["kitchen"],
                    "note": "smoke test place",
                }
                data = must_post("/memory/place/upsert", payload)
                check(data.get("ok") is True, "place upsert must return ok=true")
                check(bool(data.get("id")), "place upsert must return id")
                print_ok(item, data)

            elif item == "semantic ingest":
                print_step(item)
                payload = {
                    "robot_id": "dog_001",
                    "robot_type": "robot_dog",
                    "robot_pose": {
                        "x": 2.0,
                        "y": -0.5,
                        "z": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 1.2,
                        "frame_id": "map",
                    },
                    "image": image_b64,
                    "note": f"{object_name} near table",
                    "tags": ["smoke"],
                }
                data = must_post("/memory/semantic/ingest", payload)
                check(data.get("ok") is True, "semantic ingest must return ok=true")
                semantic_id = data.get("id")
                check(bool(semantic_id), "semantic ingest must return id")
                print_ok(item, data)

            elif item == "keyframe batch ingest":
                print_step(item)
                payload = {
                    "task_id": task_marker,
                    "items": [
                        {
                            "camera_source": "front_camera",
                            "rank": 1,
                            "score": 0.88,
                            "timestamp": time.time(),
                            "timestamp_ns": int(time.time() * 1e9),
                            "robot_id": "dog_001",
                            "robot_type": "robot_dog",
                            "pose": {
                                "x": 3.0,
                                "y": 1.5,
                                "z": 0.0,
                                "roll": 0.0,
                                "pitch": 0.0,
                                "yaw": 0.3,
                                "frame_id": "map",
                            },
                            "note": "smoke keyframe",
                            "image": image_b64,
                        }
                    ],
                }
                data = must_post("/memory/keyframe/ingest-batch", payload)
                check(data.get("ok") is True, "keyframe batch ingest must return ok=true")
                check(data.get("success", 0) >= 1, "keyframe batch ingest success must be >= 1")
                print_ok(item, data)

            elif item == "query object":
                print_step(item)
                data = must_post("/query/object", {"name": object_name, "n_results": 5})
                results = data.get("results", [])
                check(len(results) >= 1, "query object should return at least one result")
                names = [r.get("name", "") for r in results]
                check(any(object_name in n for n in names), "query object should contain inserted object name")
                print_ok(item, {"count": len(results)})

            elif item == "query place":
                print_step(item)
                data = must_post("/query/place", {"name": place_name, "n_results": 5})
                results = data.get("results", [])
                check(len(results) >= 1, "query place should return at least one result")
                names = [r.get("name", "") for r in results]
                check(any(place_name in n for n in names), "query place should contain inserted place name")
                print_ok(item, {"count": len(results)})

            elif item == "query position":
                print_step(item)
                data = must_post(
                    "/query/position",
                    {
                        "x": 1.1,
                        "y": 1.05,
                        "radius": 1.0,
                        "n_results": 10,
                    },
                )
                results = data.get("results", [])
                check(len(results) >= 1, "query position should return at least one nearby result")
                print_ok(item, {"count": len(results)})

            elif item == "query semantic text":
                print_step(item)
                data = must_post("/query/semantic/text", {"text": object_name, "n_results": 5})
                results = data.get("results", [])
                check(len(results) >= 1, "semantic text query should return at least one result")
                print_ok(item, {"count": len(results)})

            elif item == "query unified":
                print_step(item)
                data = must_post(
                    "/query/unified",
                    {
                        "text": object_name,
                        "memory_type": "semantic_frame",
                        "n_results": 5,
                    },
                )
                results = data.get("results", [])
                check(len(results) >= 1, "unified query should return at least one result")
                print_ok(item, {"count": len(results)})

            elif item == "pipeline task create":
                print_step(item)
                data = must_post(
                    "/pipeline/tasks",
                    {
                        "task_name": "offline_keyframe_pipeline",
                        "input_uri": "file:///tmp/smoke_dummy.bag",
                        "robot_id": "dog_001",
                        "robot_type": "robot_dog",
                        "options": {"smoke": True},
                    },
                )
                check(bool(data.get("task_id")), "pipeline task create should return task_id")
                print_ok(item, data)
                pipeline_task_id = data["task_id"]

            elif item == "pipeline task status":
                print_step(item)
                # Poll briefly for transition to running/completed.
                last = {}
                for _ in range(6):
                    last = must_get(f"/pipeline/tasks/{pipeline_task_id}")
                    if last.get("status") in {"running", "completed"}:
                        break
                    time.sleep(0.4)
                check(last.get("status") in {"queued", "running", "completed"}, "invalid task status")
                print_ok(item, {"status": last.get("status"), "progress": last.get("progress")})

        except Exception as exc:
            failures.append(f"{item}: {exc}")
            print(f"[FAIL] {item}: {exc}")

    print("\n========== SMOKE TEST SUMMARY ==========")
    if not failures:
        print("PASS: all checks passed")
        print(f"base={BASE}, object_id={object_id}, semantic_id={semantic_id}")
        return

    print(f"FAIL: {len(failures)} check(s) failed")
    for err in failures:
        print(f" - {err}")
    sys.exit(1)


if __name__ == "__main__":
    main()

"""GraspAnything Service.

FastAPI backend that runs YOLO + AnyGrasp on a single RGB-D frame and returns
grasp pose candidates in camera coordinate system.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from grasp_service_nofilter import GraspDetectorNoFilter


SERVICE_VERSION = "0.1.0"
CHECKPOINT_PATH = os.getenv("GRASP_CHECKPOINT_PATH", "")
MODEL_NAME = os.getenv("GRASP_YOLO_MODEL", "yolov5l6")


def _resolve_torch_device(env_value: Optional[str]) -> str:
    value = (env_value or "auto").strip().lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        return f"cuda:{value}" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda:"):
        return value if torch.cuda.is_available() else "cpu"
    return value


DEVICE = _resolve_torch_device(os.getenv("DEVICE") or os.getenv("GRASPANYTHING_DEVICE"))
GRASP_DETECTOR: Optional[GraspDetectorNoFilter] = None
INFER_LOCK = threading.Lock()


class GraspRequest(BaseModel):
    color_image: str = Field(..., description="RGB image input: base64/data-uri/path/url")
    depth_image: str = Field(..., description="Depth image input: base64/data-uri/path/url")
    camera_intrinsics: List[List[float]] = Field(..., description="3x3 camera intrinsics matrix K")
    object_name: str = Field(..., description="Target object class name, e.g. cup")
    top_k: int = Field(5, ge=1, le=20, description="Max grasp candidates per detected instance")


class GraspResponse(BaseModel):
    frame_id: str
    target: str
    top_k: int
    count: int
    results: list[dict[str, Any]]
    latency_ms: float


def _load_image_bytes(image_input: str) -> bytes:
    payload = image_input.strip()
    if payload.startswith(("http://", "https://")):
        response = requests.get(payload, timeout=20)
        response.raise_for_status()
        return response.content
    if payload.startswith("data:image"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        if Path(payload).exists():
            return Path(payload).read_bytes()
        raise ValueError("Invalid image payload: not valid base64 and path not found")


def _decode_color_bgr(image_input: str) -> np.ndarray:
    raw = _load_image_bytes(image_input)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode color image")
    return img


def _decode_depth(image_input: str) -> np.ndarray:
    raw = _load_image_bytes(image_input)
    arr = np.frombuffer(raw, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to decode depth image")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth.astype(np.float32)


def _parse_intrinsics(k: List[List[float]]) -> np.ndarray:
    matrix = np.array(k, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_intrinsics must be 3x3, got {matrix.shape}")
    return matrix


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GRASP_DETECTOR
    if not CHECKPOINT_PATH:
        raise RuntimeError("GRASP_CHECKPOINT_PATH is required")
    detector = GraspDetectorNoFilter(
        checkpoint_path=CHECKPOINT_PATH,
        model_name=MODEL_NAME,
        conf=0.5,
        device=DEVICE,
    )
    GRASP_DETECTOR = detector
    yield
    GRASP_DETECTOR = None


app = FastAPI(
    title="AbotClaw GraspAnything Service",
    version=SERVICE_VERSION,
    description="Run grasp detection (YOLO + AnyGrasp) on RGB-D inputs.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": SERVICE_VERSION,
        "device": DEVICE,
        "checkpoint_path": CHECKPOINT_PATH,
        "model_name": MODEL_NAME,
        "model_loaded": GRASP_DETECTOR is not None,
    }


@app.post("/grasp/detect", response_model=GraspResponse)
def grasp_detect(req: GraspRequest) -> GraspResponse:
    if GRASP_DETECTOR is None:
        raise HTTPException(status_code=503, detail="Grasp model not initialized")
    if not req.object_name.strip():
        raise HTTPException(status_code=400, detail="object_name cannot be empty")

    start_t = time.time()
    try:
        color_bgr = _decode_color_bgr(req.color_image)
        depth_map = _decode_depth(req.depth_image)
        camera_k = _parse_intrinsics(req.camera_intrinsics)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input payload: {exc}")

    try:
        with INFER_LOCK:
            results = GRASP_DETECTOR.get_grasp_pose_from_frame(
                color_bgr=color_bgr,
                depth_map=depth_map,
                K=camera_k,
                object_name=req.object_name,
                top_k=req.top_k,
            )
            result_dict = GRASP_DETECTOR._as_dict(results, req.object_name, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grasp inference failed: {exc}")

    latency_ms = (time.time() - start_t) * 1000.0
    return GraspResponse(
        frame_id=result_dict.get("frame_id", "camera_frame"),
        target=result_dict.get("target", req.object_name),
        top_k=int(result_dict.get("top_k", req.top_k)),
        count=int(result_dict.get("count", 0)),
        results=result_dict.get("results", []),
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8015"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

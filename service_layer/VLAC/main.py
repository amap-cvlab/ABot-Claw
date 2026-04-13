"""VLAC Critic Service.

FastAPI backend that receives one image + one reference image and returns
critic results from VLAC.
"""

from __future__ import annotations

import base64
import io
import os
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from evo_vlac import GAC_model


SERVICE_VERSION = "0.1.0"
MODEL_PATH = os.getenv("VLAC_MODEL_PATH", "./models")
MODEL_TYPE = os.getenv("VLAC_MODEL_TYPE", "internvl2")


def _resolve_vlac_device(env_value: Optional[str]) -> str:
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


DEVICE_MAP = _resolve_vlac_device(os.getenv("DEVICE") or os.getenv("VLAC_DEVICE"))

CRITIC: Optional[GAC_model] = None
INFER_LOCK = threading.Lock()


class CriticRequest(BaseModel):
    image: str = Field(..., description="Current image (base64/data-uri/path/url)")
    reference_image: str = Field(..., description="Reference image (base64/data-uri/path/url)")
    task_description: str = Field(..., description="Task description for critic evaluation")
    batch_num: int = Field(1, ge=1, le=32, description="Batch size for critic generation")
    rich: bool = Field(False, description="Whether to return decimal-rich outputs")


class CriticResponse(BaseModel):
    critic_list: List[float]
    value_list: List[float]
    latency_ms: float


def _normalize_image_input(image_input: str) -> Image.Image:
    payload = image_input.strip()
    try:
        if payload.startswith(("http://", "https://")):
            response = requests.get(payload, timeout=15)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        if payload.startswith("data:image"):
            payload = payload.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(payload, validate=True)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            if Path(payload).exists():
                return Image.open(payload).convert("RGB")
            raise ValueError("Invalid base64 string and path not found")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CRITIC
    critic = GAC_model(tag="critic")
    critic.init_model(model_path=MODEL_PATH, model_type=MODEL_TYPE, device_map=DEVICE_MAP)
    critic.temperature = 0.5
    critic.top_k = 1
    critic.set_config()
    critic.set_system_prompt()
    CRITIC = critic
    yield
    CRITIC = None


app = FastAPI(
    title="AbotClaw VLAC Critic Service",
    version=SERVICE_VERSION,
    description="Run VLAC pair-wise critic on one image and one reference image.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": SERVICE_VERSION,
        "device": DEVICE_MAP,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "model_loaded": CRITIC is not None,
    }


@app.post("/critic", response_model=CriticResponse)
def critic(req: CriticRequest) -> CriticResponse:
    if CRITIC is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if not req.task_description.strip():
        raise HTTPException(status_code=400, detail="task_description cannot be empty")

    start_t = time.time()
    try:
        current_image = _normalize_image_input(req.image)
        reference_image = _normalize_image_input(req.reference_image)
        with INFER_LOCK:
            critic_list, value_list = CRITIC.get_trajectory_critic(
                task=req.task_description,
                image_list=[reference_image, current_image],
                ref_image_list=None,
                batch_num=req.batch_num,
                rich=req.rich,
                reverse_eval=False,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Critic inference failed: {exc}")

    latency_ms = (time.time() - start_t) * 1000.0

    try:
        critic_values = [float(item) for item in critic_list]
        value_values = [float(item) for item in value_list]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Result parsing failed: {exc}")

    return CriticResponse(
        critic_list=critic_values,
        value_list=value_values,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8014"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

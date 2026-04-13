from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def decode_image_input(image_input: str) -> Image.Image:
    payload = image_input.strip()
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
        image_path = Path(payload)
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        raise ValueError("invalid image input, must be base64/data-uri/path/url")


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def text_embedding(text: str, dim: int = 128) -> np.ndarray:
    if not text.strip():
        return normalize(np.zeros((dim,), dtype=np.float32))

    buckets = np.zeros((dim,), dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[idx] += sign
    return normalize(buckets)


def image_embedding(image: Image.Image, dim: int = 128) -> np.ndarray:
    resized = image.resize((32, 32))
    arr = np.array(resized, dtype=np.float32).reshape(-1, 3)

    feats = np.concatenate(
        [
            arr.mean(axis=0),
            arr.std(axis=0),
            np.quantile(arr, [0.25, 0.5, 0.75], axis=0).reshape(-1),
        ]
    )

    projected = np.zeros((dim,), dtype=np.float32)
    for idx, value in enumerate(feats):
        projected[idx % dim] += float(value)

    return normalize(projected)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(normalize(a), normalize(b)))

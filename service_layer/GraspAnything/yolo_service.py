"""YOLOv5 inference helper (ROS-free).

This module intentionally avoids ROS dependencies and only accepts in-memory
images (numpy BGR arrays) for detection.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch


class YoloDetector:
    """Lightweight YOLOv5 wrapper for direct numpy image inference."""

    def __init__(
        self,
        model_name: str = "yolov5l6",
        conf: float = 0.5,
        device: Optional[str] = None,
        camera_frame_id: str = "wrist_camera_color_optical_frame",
        **_: object,
    ) -> None:
        os.environ.setdefault("YOLOv5_AUTOINSTALL", "0")
        os.environ.setdefault("YOLO_AUTOINSTALL", "0")

        self._model = torch.hub.load("ultralytics/yolov5", model_name)
        self._model.conf = conf
        if device is not None:
            self._model.to(device)

        self._camera_frame_id = camera_frame_id
        self._started = True

    def start(self) -> "YoloDetector":
        # Kept for backward compatibility with previous lifecycle hooks.
        self._started = True
        return self

    def stop(self) -> None:
        self._started = False

    def __enter__(self) -> "YoloDetector":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def infer_dataframe(self, color_bgr: np.ndarray):
        """Run one image inference and return YOLO pandas xyxy dataframe."""
        if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("color_bgr must be a HxWx3 BGR image")
        results = self._model(color_bgr)
        return results.pandas().xyxy[0]

    def detect_env(self, color_bgr: np.ndarray) -> List[str]:
        """Return unique object class names detected from one frame."""
        df = self.infer_dataframe(color_bgr)
        if df is None or len(df) == 0:
            return []
        return list(df["name"].unique())

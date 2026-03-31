"""YOLO visualization serving endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/yolo", tags=["yolo"])

YOLO_VIZ_DIR = "/tmp/yolo_viz"


@router.get("/visualization", include_in_schema=False)
async def get_visualization():
    """Get the latest YOLO segmentation visualization image.

    Returns the most recent annotated image from yolo.segment_camera()
    or yolo.segment_image() as JPEG.

    No lease required.
    """
    viz_path = os.path.join(YOLO_VIZ_DIR, "latest.jpg")
    if not os.path.exists(viz_path):
        return JSONResponse(
            {"error": "No visualization available. Run yolo.segment_camera() first."},
            status_code=404,
        )
    return FileResponse(viz_path, media_type="image/jpeg")

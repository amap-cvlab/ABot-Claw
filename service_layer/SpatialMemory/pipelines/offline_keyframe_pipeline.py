from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OfflineKeyframePipeline:
    """Placeholder for fully independent offline pipeline implementation."""

    def run(self, input_uri: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "placeholder",
            "input_uri": input_uri,
            "options": options,
            "message": "replace with real rosbag->keyframe->ingest implementation",
        }

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Pose(BaseModel):
    x: float
    y: float
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None
    qw: Optional[float] = None
    frame_id: str = "map"


class NavigationTarget(BaseModel):
    pose: Pose
    confidence: float = 1.0


class ObjectMemoryUpsertRequest(BaseModel):
    object_name: str
    object_id: Optional[str] = None
    robot_id: str
    robot_type: str
    robot_pose: Pose
    object_pose: Pose
    source: str = "yolo"
    bbox_xyxy: Optional[list[float]] = None
    detect_confidence: float = 1.0
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    timestamp: Optional[float] = None
    image: Optional[str] = Field(default=None, description="base64/data-uri/path/url")


class PlaceMemoryUpsertRequest(BaseModel):
    place_name: str
    robot_id: str
    robot_type: str
    place_pose: Pose
    alias: list[str] = Field(default_factory=list)
    note: str = ""
    timestamp: Optional[float] = None


class SemanticFrameIngestRequest(BaseModel):
    robot_id: str
    robot_type: str
    robot_pose: Pose
    source: str = "camera"
    task_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    timestamp: Optional[float] = None
    image: str = Field(..., description="base64/data-uri/path/url")


class KeyframeItem(BaseModel):
    camera_source: str
    rank: Optional[int] = None
    score: float = 0.0
    timestamp: Optional[float] = None
    timestamp_ns: Optional[int] = None
    robot_id: str
    robot_type: str
    pose: Pose
    note: str = ""
    image: str


class KeyframeBatchIngestRequest(BaseModel):
    task_id: str
    items: list[KeyframeItem]


class TaskCreateRequest(BaseModel):
    task_name: Literal["offline_keyframe_pipeline"]
    input_uri: str
    robot_id: str
    robot_type: str
    options: dict[str, Any] = Field(default_factory=dict)


class NameQuery(BaseModel):
    name: str
    n_results: int = 10
    robot_id: Optional[str] = None


class PositionQuery(BaseModel):
    x: float
    y: float
    radius: float = 1.5
    n_results: int = 20
    memory_type: Optional[Literal["object", "place", "keyframe", "semantic_frame"]] = None


class SemanticTextQuery(BaseModel):
    text: str
    n_results: int = 10
    memory_type: Optional[Literal["object", "place", "keyframe", "semantic_frame"]] = None


class UnifiedQuery(BaseModel):
    text: Optional[str] = None
    object_name: Optional[str] = None
    place_name: Optional[str] = None
    robot_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    radius: float = 2.0
    memory_type: Optional[Literal["object", "place", "keyframe", "semantic_frame"]] = None
    n_results: int = 20


class MemoryResult(BaseModel):
    id: str
    memory_type: str
    name: str
    robot_id: str
    robot_type: str
    target_pose: Pose
    robot_pose: Optional[Pose] = None
    source: str
    timestamp: float
    confidence: float
    evidence: dict[str, Any] = Field(default_factory=dict)


class GenericResultsResponse(BaseModel):
    results: list[MemoryResult]

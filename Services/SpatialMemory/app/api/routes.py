from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import (
    GenericResultsResponse,
    KeyframeBatchIngestRequest,
    NameQuery,
    ObjectMemoryUpsertRequest,
    PlaceMemoryUpsertRequest,
    PositionQuery,
    SemanticFrameIngestRequest,
    SemanticTextQuery,
    TaskCreateRequest,
    UnifiedQuery,
)
from app.services.memory_service import MemoryService
from app.services.task_service import TaskService


def create_router(memory_service: MemoryService, task_service: TaskService) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return memory_service.health()

    @router.post("/memory/object/upsert")
    def upsert_object(req: ObjectMemoryUpsertRequest) -> dict:
        return memory_service.upsert_object(req)

    @router.post("/memory/place/upsert")
    def upsert_place(req: PlaceMemoryUpsertRequest) -> dict:
        return memory_service.upsert_place(req)

    @router.post("/memory/semantic/ingest")
    def ingest_semantic(req: SemanticFrameIngestRequest) -> dict:
        return memory_service.ingest_semantic_frame(req)

    @router.post("/memory/keyframe/ingest-batch")
    def ingest_keyframe_batch(req: KeyframeBatchIngestRequest) -> dict:
        return memory_service.ingest_keyframe_batch(req)

    @router.post("/query/object", response_model=GenericResultsResponse)
    def query_object(req: NameQuery) -> GenericResultsResponse:
        return GenericResultsResponse(results=memory_service.query_by_name("object", req.name, req.n_results, req.robot_id))

    @router.post("/query/place", response_model=GenericResultsResponse)
    def query_place(req: NameQuery) -> GenericResultsResponse:
        return GenericResultsResponse(results=memory_service.query_by_name("place", req.name, req.n_results, req.robot_id))

    @router.post("/query/position", response_model=GenericResultsResponse)
    def query_position(req: PositionQuery) -> GenericResultsResponse:
        return GenericResultsResponse(
            results=memory_service.query_by_position(
                x=req.x,
                y=req.y,
                radius=req.radius,
                n_results=req.n_results,
                memory_type=req.memory_type,
            )
        )

    @router.post("/query/semantic/text", response_model=GenericResultsResponse)
    def query_semantic_text(req: SemanticTextQuery) -> GenericResultsResponse:
        return GenericResultsResponse(results=memory_service.semantic_text_query(req.text, req.n_results, req.memory_type))

    @router.post("/query/unified", response_model=GenericResultsResponse)
    def query_unified(req: UnifiedQuery) -> GenericResultsResponse:
        return GenericResultsResponse(results=memory_service.unified_query(req))

    @router.post("/pipeline/tasks")
    def create_pipeline_task(req: TaskCreateRequest) -> dict:
        return task_service.create_offline_keyframe_task(
            input_uri=req.input_uri,
            robot_id=req.robot_id,
            robot_type=req.robot_type,
            options=req.options,
        )

    @router.get("/pipeline/tasks/{task_id}")
    def get_pipeline_task(task_id: str) -> dict:
        row = task_service.get_task(task_id)
        if not row:
            raise HTTPException(status_code=404, detail="task not found")
        return row

    return router

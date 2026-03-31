from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from app.api.routes import create_router
from app.config import settings
from app.services.memory_service import MemoryService
from app.services.task_service import TaskService
from app.storage import SqliteStore


store = SqliteStore(settings.sqlite_path)
memory_service = MemoryService(store)
task_service = TaskService(store)

app = FastAPI(
    title=settings.service_name,
    version=settings.version,
    description="Independent unified memory module for object/place/keyframe/semantic memory.",
)

app.include_router(create_router(memory_service, task_service))


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)

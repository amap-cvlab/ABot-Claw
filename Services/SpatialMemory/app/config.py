from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str = "Unified Spatial Memory Hub"
    version: str = "0.1.0"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8022"))
    data_dir: Path = Path(os.getenv("MEMORY_HUB_DATA_DIR", "./data")).resolve()
    image_dirname: str = "images"
    sqlite_filename: str = "memory_hub.db"
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "128"))

    @property
    def image_dir(self) -> Path:
        return self.data_dir / self.image_dirname

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename


settings = Settings()

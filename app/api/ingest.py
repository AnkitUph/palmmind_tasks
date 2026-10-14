from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.core.deps import get_ingestion_service
from app.models.schemas import ChunkingStrategy, IngestResponse
from app.services.ingestion import IngestionService

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(
    file: Annotated[UploadFile, File(description="A .pdf or .txt file")],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.FIXED,
) -> IngestResponse:
    return service.ingest(
        filename=file.filename or "unnamed",
        content_type=file.content_type or "application/octet-stream",
        content=file.file.read(),
        strategy=strategy,
    )

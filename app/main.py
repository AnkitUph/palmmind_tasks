from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, ingest
from app.api.errors import register_error_handlers
from app.core.deps import get_engine, get_vector_store
from app.models.db import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(get_engine())
    get_vector_store().ensure_collection()
    yield


app = FastAPI(title="Palm Mind RAG Backend", version="1.0.0", lifespan=lifespan)
app.include_router(ingest.router)
app.include_router(chat.router)
register_error_handlers(app)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.services.booking import BookingService
from app.services.embeddings import Embedder, SentenceTransformerEmbedder
from app.services.ingestion import IngestionService
from app.services.llm import LLMClient, OllamaClient
from app.services.memory import ChatMemory
from app.services.rag import RagService
from app.services.repositories import BookingRepository, DocumentRepository
from app.services.vector_store import VectorStore


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Iterator[Session]:
    with _session_factory()() as session:
        yield session


@lru_cache
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder(get_settings().embedding_model)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return OllamaClient(
        settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds
    )


def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(get_qdrant_client(), settings.qdrant_collection, get_embedder().dimension)


def get_chat_memory() -> ChatMemory:
    settings = get_settings()
    return ChatMemory(get_redis(), settings.chat_history_turns, settings.chat_ttl_seconds)


DbSession = Annotated[Session, Depends(get_db)]


def get_ingestion_service(db: DbSession) -> IngestionService:
    return IngestionService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        documents=DocumentRepository(db),
        max_upload_bytes=get_settings().max_upload_mb * 1024 * 1024,
    )


def get_rag_service(db: DbSession) -> RagService:
    settings = get_settings()
    memory = get_chat_memory()
    llm = get_llm()
    return RagService(
        llm=llm,
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        memory=memory,
        booking=BookingService(llm, memory, BookingRepository(db)),
        top_k=settings.top_k,
        min_score=settings.min_score,
    )

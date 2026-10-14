from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SENTENCE = "sentence"


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    strategy: ChunkingStrategy
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    snippet: str


class BookingStatus(BaseModel):
    state: Literal["collecting", "confirmed", "cancelled"]
    collected: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    booking_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    booking: BookingStatus | None = None

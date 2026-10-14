from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

_BATCH_SIZE = 64


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    document_id: str
    filename: str
    chunk_index: int


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=list(vector),
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": text,
                },
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), _BATCH_SIZE):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + _BATCH_SIZE],
                wait=True,
            )

    def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )
        return results

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
        )

from __future__ import annotations

import logging
import uuid

from app.core.exceptions import EmptyDocumentError, FileTooLargeError
from app.models.schemas import ChunkingStrategy, IngestResponse
from app.services.chunking import get_chunker
from app.services.document_loader import extract_text
from app.services.embeddings import Embedder
from app.services.repositories import DocumentRepository
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        documents: DocumentRepository,
        max_upload_bytes: int,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._documents = documents
        self._max_upload_bytes = max_upload_bytes

    def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        strategy: ChunkingStrategy,
    ) -> IngestResponse:
        if len(content) > self._max_upload_bytes:
            raise FileTooLargeError

        text = extract_text(filename, content)
        chunks = get_chunker(strategy).split(text)
        if not chunks:
            raise EmptyDocumentError

        vectors = self._embedder.embed_documents(chunks)
        document_id = str(uuid.uuid4())

        self._vector_store.upsert_chunks(document_id, filename, chunks, vectors)
        try:
            self._documents.create(
                document_id=document_id,
                filename=filename,
                content_type=content_type,
                chunking_strategy=strategy.value,
                chunk_count=len(chunks),
                char_count=len(text),
            )
        except Exception:
            logger.exception("Metadata save failed; removing vectors for %s", document_id)
            self._vector_store.delete_document(document_id)
            raise

        logger.info("Ingested %s as %s (%d chunks)", filename, document_id, len(chunks))
        return IngestResponse(
            document_id=document_id,
            filename=filename,
            strategy=strategy,
            chunk_count=len(chunks),
        )

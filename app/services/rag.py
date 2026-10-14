from __future__ import annotations

import logging
from collections.abc import Sequence

from app.models.schemas import ChatResponse, SourceChunk
from app.services.booking import BookingService
from app.services.embeddings import Embedder
from app.services.llm import ChatMessage, LLMClient
from app.services.memory import ChatMemory
from app.services.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

NO_CONTEXT_REPLY = (
    "I could not find anything about that in the uploaded documents. "
    "Try rephrasing, or upload a document that covers it."
)

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context below.
If the answer is not in the context, say you do not know. Keep answers clear and short.

Context:
{context}"""

_CONDENSE_PROMPT = """Rewrite the user's latest message as a single standalone question that
can be understood without the conversation. Resolve pronouns like "it" or "that" using the
conversation. If it is already standalone, return it unchanged. Output ONLY the question."""


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{number}] ({chunk.filename}) {chunk.text}" for number, chunk in enumerate(chunks, 1)
    )


def build_answer_messages(
    question: str,
    history: Sequence[ChatMessage],
    chunks: Sequence[RetrievedChunk],
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(context=build_context(chunks))}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


class RagService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        embedder: Embedder,
        vector_store: VectorStore,
        memory: ChatMemory,
        booking: BookingService,
        top_k: int,
        min_score: float,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._vector_store = vector_store
        self._memory = memory
        self._booking = booking
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, session_id: str, message: str) -> ChatResponse:
        history = self._memory.get_history(session_id)

        outcome = self._booking.handle(session_id, message, history)
        if outcome is not None:
            self._memory.append_turn(session_id, message, outcome.reply)
            return ChatResponse(answer=outcome.reply, booking=outcome.status)

        question = self._condense_question(message, history)
        chunks = self._retrieve(question)
        if not chunks:
            self._memory.append_turn(session_id, message, NO_CONTEXT_REPLY)
            return ChatResponse(answer=NO_CONTEXT_REPLY)

        reply = self._llm.chat(build_answer_messages(message, history, chunks)).strip()
        self._memory.append_turn(session_id, message, reply)
        return ChatResponse(
            answer=reply,
            sources=[
                SourceChunk(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    chunk_index=chunk.chunk_index,
                    score=round(chunk.score, 4),
                    snippet=chunk.text[:200],
                )
                for chunk in chunks
            ],
        )

    def _condense_question(self, message: str, history: Sequence[ChatMessage]) -> str:
        if not history:
            return message
        messages: list[ChatMessage] = [{"role": "system", "content": _CONDENSE_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        rewritten = self._llm.chat(messages).strip()
        logger.debug("Condensed %r -> %r", message, rewritten)
        return rewritten or message

    def _retrieve(self, question: str) -> list[RetrievedChunk]:
        vector = self._embedder.embed_query(question)
        hits = self._vector_store.search(vector, self._top_k * 3)
        unique: list[RetrievedChunk] = []
        seen: set[str] = set()
        for hit in hits:
            key = " ".join(hit.text.split())
            if hit.score < self._min_score or key in seen:
                continue
            seen.add(key)
            unique.append(hit)
        return unique[: self._top_k]

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.models.schemas import ChunkingStrategy

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"[ \t]+")


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class FixedSizeChunker:
    chunk_size: int = 800
    overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")

    def split(self, text: str) -> list[str]:
        text = _normalize(text)
        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == len(text):
                break
            start += step
        return chunks


@dataclass(frozen=True)
class SentenceChunker:
    max_chars: int = 800
    overlap_sentences: int = 1

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")

    def split(self, text: str) -> list[str]:
        sentences = self._sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for sentence in sentences:
            if current and length + len(sentence) + 1 > self.max_chars:
                chunks.append(" ".join(current))
                carry = current[-self.overlap_sentences :] if self.overlap_sentences else []
                while carry and sum(len(s) + 1 for s in carry) + len(sentence) + 1 > self.max_chars:
                    carry.pop(0)
                current = carry
                length = sum(len(s) + 1 for s in current)
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _sentences(self, text: str) -> list[str]:
        flat = _normalize(text.replace("\r", " ").replace("\n", " "))
        fallback = FixedSizeChunker(chunk_size=self.max_chars, overlap=0)
        result: list[str] = []
        for sentence in _SENTENCE_BOUNDARY.split(flat):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > self.max_chars:
                result.extend(fallback.split(sentence))
            else:
                result.append(sentence)
        return result


def get_chunker(strategy: ChunkingStrategy) -> Chunker:
    if strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker()
    return SentenceChunker()

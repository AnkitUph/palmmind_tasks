from __future__ import annotations

import io
from pathlib import PurePath

from pypdf import PdfReader

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


def extract_text(filename: str, content: bytes) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError
    text = _read_pdf(content) if extension == ".pdf" else content.decode("utf-8", "replace")
    text = text.strip()
    if not text:
        raise EmptyDocumentError
    return text


def _read_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

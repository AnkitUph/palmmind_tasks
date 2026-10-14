from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict

import httpx

from app.core.exceptions import LLMUnavailableError


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    def chat(self, messages: Sequence[ChatMessage], *, json_mode: bool = False) -> str: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, messages: Sequence[ChatMessage], *, json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": 0.0 if json_mode else 0.2},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
            return str(response.json()["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMUnavailableError(f"LLM request failed: {exc}") from exc

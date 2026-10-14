from __future__ import annotations

import json
from typing import cast

from redis import Redis

from app.services.llm import ChatMessage


class ChatMemory:
    def __init__(self, client: Redis, max_turns: int, ttl_seconds: int) -> None:
        self._client = client
        self._max_messages = max_turns * 2
        self._ttl = ttl_seconds

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:{session_id}:history"

    @staticmethod
    def _booking_key(session_id: str) -> str:
        return f"chat:{session_id}:booking"

    def get_history(self, session_id: str) -> list[ChatMessage]:
        raw = self._client.lrange(self._history_key(session_id), -self._max_messages, -1)
        return [cast(ChatMessage, json.loads(item)) for item in raw]

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self._history_key(session_id)
        pipe = self._client.pipeline()
        pipe.rpush(
            key,
            json.dumps({"role": "user", "content": user_message}),
            json.dumps({"role": "assistant", "content": assistant_message}),
        )
        pipe.ltrim(key, -self._max_messages, -1)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_booking_state(self, session_id: str) -> dict[str, str] | None:
        raw = self._client.get(self._booking_key(session_id))
        if raw is None:
            return None
        return cast(dict[str, str], json.loads(raw))

    def set_booking_state(self, session_id: str, state: dict[str, str]) -> None:
        self._client.set(self._booking_key(session_id), json.dumps(state), ex=self._ttl)

    def clear_booking_state(self, session_id: str) -> None:
        self._client.delete(self._booking_key(session_id))

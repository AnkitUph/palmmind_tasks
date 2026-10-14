from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time

from pydantic import BaseModel, EmailStr, TypeAdapter, ValidationError, field_validator

from app.models.schemas import BookingStatus
from app.services.llm import ChatMessage, LLMClient
from app.services.memory import ChatMemory
from app.services.repositories import BookingRepository

logger = logging.getLogger(__name__)

REQUIRED_FIELDS: tuple[str, ...] = ("name", "email", "date", "time")

_FIELD_PROMPTS: dict[str, str] = {
    "name": "your full name",
    "email": "your email address",
    "date": "your preferred date (for example 2026-10-20)",
    "time": "your preferred time (for example 15:30)",
}

_EMAIL_ADAPTER: TypeAdapter[EmailStr] = TypeAdapter(EmailStr)

_BOOKING_HINT = re.compile(
    r"\b(book|booking|schedule|scheduling|appointment|reserve|slot|interview|meeting)\b",
    re.IGNORECASE,
)
_BOOKING_REQUEST = re.compile(
    r"\b(book|schedule|arrange|set up|reserve)\b.*\b(interview|meeting|appointment|slot|call)\b"
    r"|\b(interview|meeting|appointment)\b.*\b(book|schedule|arrange)\b",
    re.IGNORECASE,
)

_EXTRACTION_PROMPT = """You extract interview-booking details from a chat.
Today is {today} ({weekday}).
Reply with ONLY a JSON object with these keys:
  "wants_booking": true if the user wants to book or schedule an interview, else false
  "cancel": true if the user wants to cancel the booking in progress, else false
  "name": the user's full name, or null
  "email": the user's email address, or null
  "date": the interview date as YYYY-MM-DD (resolve words like "tomorrow" or "next Monday"
          using today's date), or null
  "time": the interview time in 24-hour HH:MM, or null
Use null for anything the user has not clearly stated. Never guess or invent values.
Details already collected: {state}"""


class BookingExtraction(BaseModel):
    wants_booking: bool = False
    cancel: bool = False
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None

    @field_validator("wants_booking", "cancel", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    @field_validator("name", "email", "date", "time", mode="before")
    @classmethod
    def _coerce_nullable_str(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in ("null", "none", "n/a", "undefined"):
            return None
        return text


@dataclass(frozen=True)
class BookingOutcome:
    reply: str
    status: BookingStatus


@dataclass
class _MergeResult:
    updated: set[str] = field(default_factory=set)
    invalid: list[str] = field(default_factory=list)


class BookingService:
    def __init__(
        self,
        llm: LLMClient,
        memory: ChatMemory,
        bookings: BookingRepository,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._bookings = bookings

    def handle(
        self,
        session_id: str,
        message: str,
        history: Sequence[ChatMessage],
    ) -> BookingOutcome | None:
        stored = self._memory.get_booking_state(session_id)
        active = stored is not None
        state: dict[str, str] = dict(stored or {})

        if not active and not _BOOKING_HINT.search(message):
            return None

        extraction = self._extract(message, history, state)
        wants_booking = extraction.wants_booking or bool(_BOOKING_REQUEST.search(message))
        if not active and not wants_booking:
            return None

        if active and extraction.cancel:
            self._memory.clear_booking_state(session_id)
            return BookingOutcome(
                reply="Okay, I cancelled the interview booking. Ask me anything else!",
                status=BookingStatus(state="cancelled"),
            )

        merge = self._merge(state, extraction)
        if active and not wants_booking and not merge.updated and not merge.invalid:
            return None

        missing = [name for name in REQUIRED_FIELDS if name not in state]
        if missing or merge.invalid:
            self._memory.set_booking_state(session_id, state)
            return BookingOutcome(
                reply=self._ask_for(missing, merge.invalid),
                status=BookingStatus(state="collecting", collected=state, missing=missing),
            )

        booking = self._bookings.create(
            session_id=session_id,
            name=state["name"],
            email=state["email"],
            interview_date=date.fromisoformat(state["date"]),
            interview_time=time.fromisoformat(state["time"]),
        )
        self._memory.clear_booking_state(session_id)
        return BookingOutcome(
            reply=(
                f"Your interview is booked, {state['name']}! "
                f"Date: {state['date']}, time: {state['time']}, contact: {state['email']} "
                f"(booking ID {booking.id})."
            ),
            status=BookingStatus(state="confirmed", collected=state, booking_id=booking.id),
        )

    def _extract(
        self,
        message: str,
        history: Sequence[ChatMessage],
        state: dict[str, str],
    ) -> BookingExtraction:
        now = datetime.now()
        system = _EXTRACTION_PROMPT.format(
            today=now.date().isoformat(),
            weekday=now.strftime("%A"),
            state=json.dumps(state),
        )
        messages: list[ChatMessage] = [{"role": "system", "content": system}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": message})
        return self._parse(self._llm.chat(messages, json_mode=True))

    @staticmethod
    def _parse(raw: str) -> BookingExtraction:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            return BookingExtraction.model_validate(json.loads(cleaned))
        except (ValueError, ValidationError):
            logger.warning("Could not parse booking extraction: %r", raw[:200])
            return BookingExtraction()

    @staticmethod
    def _merge(state: dict[str, str], extraction: BookingExtraction) -> _MergeResult:
        result = _MergeResult()

        if extraction.name and extraction.name.strip():
            state["name"] = extraction.name.strip()
            result.updated.add("name")

        if extraction.email:
            try:
                state["email"] = str(_EMAIL_ADAPTER.validate_python(extraction.email.strip()))
                result.updated.add("email")
            except ValidationError:
                result.invalid.append("email")

        if extraction.date:
            try:
                parsed_date = date.fromisoformat(extraction.date.strip())
                if parsed_date < date.today():
                    raise ValueError("date is in the past")
                state["date"] = parsed_date.isoformat()
                result.updated.add("date")
            except ValueError:
                result.invalid.append("date")

        if extraction.time:
            try:
                state["time"] = time.fromisoformat(extraction.time.strip()).strftime("%H:%M")
                result.updated.add("time")
            except ValueError:
                result.invalid.append("time")

        return result

    @staticmethod
    def _ask_for(missing: Sequence[str], invalid: Sequence[str]) -> str:
        parts: list[str] = []
        if invalid:
            parts.append(
                f"I could not use the {' and '.join(invalid)} you gave "
                "(dates must not be in the past)."
            )
        if missing:
            wanted = ", ".join(_FIELD_PROMPTS[name] for name in missing)
            parts.append(f"To book your interview, please share {wanted}.")
        return " ".join(parts)

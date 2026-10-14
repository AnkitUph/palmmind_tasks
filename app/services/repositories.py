from __future__ import annotations

from datetime import date, time

from sqlalchemy.orm import Session

from app.models.db import Booking, Document


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        chunking_strategy: str,
        chunk_count: int,
        char_count: int,
    ) -> Document:
        document = Document(
            id=document_id,
            filename=filename,
            content_type=content_type,
            chunking_strategy=chunking_strategy,
            chunk_count=chunk_count,
            char_count=char_count,
        )
        self._session.add(document)
        self._session.commit()
        return document


class BookingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
    ) -> Booking:
        booking = Booking(
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )
        self._session.add(booking)
        self._session.commit()
        return booking

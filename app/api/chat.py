from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.deps import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import RagService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> ChatResponse:
    return service.answer(request.session_id, request.message)

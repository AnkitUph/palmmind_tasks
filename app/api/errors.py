from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import OperationalError

from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


def _json(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _json(exc.status_code, exc.detail)

    @app.exception_handler(RedisError)
    async def handle_redis_error(_: Request, exc: RedisError) -> JSONResponse:
        logger.error("Redis error: %s", exc)
        return _json(503, "Chat memory (Redis) is unavailable")

    @app.exception_handler(ResponseHandlingException)
    @app.exception_handler(UnexpectedResponse)
    async def handle_qdrant_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Qdrant error: %s", exc)
        return _json(503, "Vector database (Qdrant) is unavailable")

    @app.exception_handler(OperationalError)
    async def handle_db_error(_: Request, exc: OperationalError) -> JSONResponse:
        logger.error("Database error: %s", exc)
        return _json(503, "Database is unavailable")

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return _json(500, "Internal server error. See the server logs for details.")

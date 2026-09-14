"""Maps ``plateproof.serving.errors`` (transport-agnostic) to HTTP responses.
No response body ever contains a filesystem path or a traceback.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from plateproof.api.schemas import ApiError
from plateproof.serving.errors import (
    DatastoreUnavailableError,
    InvalidPaginationError,
    InvalidQueryError,
    InvalidQuestionError,
    ModelCardNotFoundError,
    RestaurantNotFoundError,
)

_Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]

_STATUS_BY_ERROR: dict[type[Exception], int] = {
    InvalidPaginationError: 422,
    InvalidQueryError: 422,
    InvalidQuestionError: 422,
    RestaurantNotFoundError: 404,
    ModelCardNotFoundError: 404,
    DatastoreUnavailableError: 503,
}


def register_exception_handlers(app: FastAPI) -> None:
    for error_type, status_code in _STATUS_BY_ERROR.items():

        def _make_handler(code: int) -> _Handler:
            async def _handler(request: Request, exc: Exception) -> JSONResponse:
                return JSONResponse(
                    status_code=code,
                    content=ApiError(error=type(exc).__name__, message=str(exc)).model_dump(),
                )

            return _handler

        app.add_exception_handler(error_type, _make_handler(status_code))

    @app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Deliberately generic: never leak a traceback, exception message, or
        # local path from an unexpected internal failure.
        return JSONResponse(
            status_code=500,
            content=ApiError(
                error="internal_error", message="An internal error occurred."
            ).model_dump(),
        )

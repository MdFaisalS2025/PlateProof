"""Task 8/9 routes are reserved but not implemented in Task 7. Neither
returns a fabricated result -- both are explicit, documented 501 responses."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED_BODY = {
    "error": "not_implemented",
    "message": "This feature is planned for a later phase of PlateProof and is not available yet.",
}


@router.post("/copilot/query")
def copilot_query() -> JSONResponse:
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED_BODY)


@router.post("/owners/documents/extract")
def owners_documents_extract() -> JSONResponse:
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED_BODY)

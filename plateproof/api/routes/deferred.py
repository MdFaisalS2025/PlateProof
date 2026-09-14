"""Task 9's owner-document-extraction route is reserved but not
implemented. It does not return a fabricated result -- an explicit,
documented 501 response. The Task 8B Copilot route
(``POST /copilot/query``) has been implemented -- see
``plateproof.api.routes.copilot`` -- and is no longer deferred here."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED_BODY = {
    "error": "not_implemented",
    "message": "This feature is planned for a later phase of PlateProof and is not available yet.",
}


@router.post("/owners/documents/extract")
def owners_documents_extract() -> JSONResponse:
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED_BODY)

"""``POST /owners/documents/extract`` -- the real Task 9B multipart endpoint,
replacing the deferred 501. Calls the SAME
``plateproof.documents.service.extract_document(...)`` function Task 9A
built and fully tested -- this module never imports ``plateproof.documents
.pdf``, ``plateproof.documents.images``, or any ``plateproof.documents.ocr``
submodule itself, and never touches ``worker/pool.py`` except through this
one function (see ``tests/documents/test_task9b_boundary.py``).

Restaurant identity and jurisdiction are always resolved server-side from
``Repository`` -- a client-supplied jurisdiction is never accepted, exactly
like the Task 8B Copilot route.

Two-layer upload-size design (Revision 4 of the approved plan):

* **Layer 1 -- the deployment's own gateway/reverse-proxy request-body-size
  limit.** This is a REQUIRED production precondition, not optional
  hardening. Uvicorn has no built-in request-body-size option of its own
  (its ``--h11-max-incomplete-event-size`` flag bounds only the request
  line and headers of an incomplete HTTP event, never the body) -- without
  a reverse proxy or an ASGI body-size middleware in front of it, a bare
  ``uvicorn`` process has no mechanism to reject an oversized body before
  Starlette accepts (and, for a large enough file, spools to disk) it.
  This route's own code cannot prevent that transient spooling; only
  Layer 1 can. See ``README.md`` for the exact deployment requirement.
* **Layer 2 -- this route's own bounded read**, below: reads at most
  ``documents_max_upload_bytes + 1`` bytes from the ``UploadFile`` Starlette
  already produced, and rejects (413) if that many bytes were read. This
  is the authoritative *application-level* limit, but it necessarily runs
  only after Starlette's own multipart parser already accepted the file.

``request.form(max_files=1, max_fields=2, max_part_size=...)`` configures
Starlette's own parser-complexity limits -- NOT a file-content-size
guarantee (verified directly against the installed Starlette's
``formparsers.py``: ``max_part_size`` is only ever checked for a *non-file*
part; an uploaded file's own bytes are never compared against it during
streaming). ``spool_max_size`` -- the memory-to-disk rollover threshold for
Starlette's internal ``SpooledTemporaryFile`` -- is a fixed class attribute
on the installed Starlette version's ``MultiPartParser``, not a keyword
argument ``Request.form()`` exposes; this route does not claim to
configure it, and does not need to, since it was never a rejection
boundary in the first place (only Layer 1 and Layer 2 above are).

A second file part, or an oversized non-file field (``restaurant_id``),
is rejected as **413** (a resource-limit condition) -- never 422. A
missing/blank ``restaurant_id`` or a missing file part is **422** (a
request-shape problem). An unrecognized document signature is **415**. An
unknown ``restaurant_id`` is **404**. Every outcome *after* a successful
call to ``extract_document(...)`` -- completed, ocr_unavailable, or
failed, including a worker timeout/crash/encrypted/malformed/text-limit
document -- is always **200**: Task 9A's own typed ``processing_status``
and ``warnings`` are preserved and returned as-is, never collapsed into an
HTTP error and never silently reported as a completed draft.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from plateproof.api.dependencies import get_app_settings, get_document_worker_pool, get_repository
from plateproof.api.documents_projection import project_draft
from plateproof.api.schemas import DocumentExtractionResponse
from plateproof.core.config import Settings
from plateproof.documents.service import extract_document
from plateproof.documents.validation import validate_upload_bytes
from plateproof.documents.worker.pool import WorkerPool
from plateproof.serving.errors import (
    DocumentRequestError,
    DocumentTooLargeError,
    RestaurantNotFoundError,
    UnsupportedDocumentTypeError,
)
from plateproof.serving.repository import Repository

router = APIRouter()

#: restaurant_id (a non-file field) + file (a file field) -- exactly two
#: fields are ever expected; anything else is a malformed request, not
#: silently ignored.
_MAX_FORM_FIELDS = 2
_MAX_RESTAURANT_ID_LENGTH = 128


@router.post("/owners/documents/extract", response_model=DocumentExtractionResponse)
async def owners_documents_extract(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    repository: Repository = Depends(get_repository),
    pool: WorkerPool = Depends(get_document_worker_pool),
) -> DocumentExtractionResponse:
    try:
        form = await request.form(
            max_files=1,
            max_fields=_MAX_FORM_FIELDS,
            max_part_size=settings.documents_max_upload_bytes + 1,
        )
    except StarletteHTTPException as exc:
        # Starlette's own Request.form() catches formparsers.MultiPartException
        # internally and re-raises it as this 400 HTTPException (verified
        # against the installed version's Request._get_form) -- covers both
        # "a second file part" and "an oversized non-file field" (e.g.
        # restaurant_id). Both are resource-limit conditions per the
        # approved plan, so both are remapped to 413, not left as
        # Starlette's own generic 400.
        raise DocumentTooLargeError(
            "too many document parts, or a form field exceeded the size limit"
        ) from exc

    restaurant_id = form.get("restaurant_id")
    file = form.get("file")

    if not isinstance(restaurant_id, str) or not restaurant_id.strip():
        raise DocumentRequestError("restaurant_id is required")
    if len(restaurant_id) > _MAX_RESTAURANT_ID_LENGTH:
        raise DocumentTooLargeError("restaurant_id exceeds the maximum allowed length")
    if not isinstance(file, StarletteUploadFile):
        raise DocumentRequestError("exactly one document file is required")

    try:
        restaurant = repository.get_restaurant(restaurant_id)
        if restaurant is None:
            raise RestaurantNotFoundError(restaurant_id)
        jurisdiction = restaurant["jurisdiction"]
        expected_restaurant_name = str(restaurant.get("name") or "")

        # Layer 2: the authoritative, application-level bounded read -- see
        # the module docstring for why this can only run after Starlette
        # has already accepted the file.
        data = await file.read(settings.documents_max_upload_bytes + 1)
        if len(data) > settings.documents_max_upload_bytes:
            raise DocumentTooLargeError("uploaded file exceeds the configured size limit")

        validation = validate_upload_bytes(data, max_bytes=settings.documents_max_upload_bytes)
        if not validation.accepted:
            if validation.rejection_reason == "unsupported_media_type":
                raise UnsupportedDocumentTypeError("unsupported document type")
            if validation.rejection_reason == "upload_too_large":
                raise DocumentTooLargeError("uploaded file exceeds the configured size limit")
            raise DocumentRequestError("the uploaded file could not be accepted")

        draft = extract_document(
            data,
            expected_jurisdiction=jurisdiction,
            restaurant_id=restaurant_id,
            expected_restaurant_name=expected_restaurant_name,
            pool=pool,
            max_upload_bytes=settings.documents_max_upload_bytes,
            max_pages=settings.documents_max_pages,
            max_pixels=settings.documents_max_pixels_per_page,
        )
        if draft is None:
            # validate_upload_bytes above already accepted these exact
            # bytes, so this should never happen in practice -- fails
            # closed as a request-shape problem rather than ever
            # fabricating a draft.
            raise DocumentRequestError("the uploaded file could not be processed")
        return project_draft(draft)
    finally:
        # Closed on every exit path: success, validation rejection,
        # RestaurantNotFoundError, or any other exception raised above.
        await file.close()

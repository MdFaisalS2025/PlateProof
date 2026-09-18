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

A second file part (whether under the same field name or a different
one), or an oversized non-file field (``restaurant_id``), is rejected as
**413** (a resource-limit condition) -- never 422. A missing/blank
``restaurant_id``, a missing file part, a duplicate ``restaurant_id``
value, or any field name outside the exact allowed set
(``{"restaurant_id", "file"}``) is **422** (a request-shape problem) --
independent-review correction of ``c60cc80``, Finding 2: Starlette's own
``max_fields``/``max_files`` count non-file and file parts separately, so
e.g. two ``restaurant_id`` values, or one extra unexpected field alongside
a valid ``restaurant_id``+``file`` pair, both fit under
``max_fields=2``/``max_files=1`` and are never caught by Starlette's own
parser limits -- only this route's own exact-shape check does. Whatever
``UploadFile`` Starlette successfully parsed for ``file`` is closed on
*every* exit path, including these request-shape rejections raised before
the restaurant/file combination was ever confirmed valid.

An unrecognized document signature is **415**. An unknown
``restaurant_id`` is **404**. Every outcome *after* a successful call to
``extract_document(...)`` -- completed, ocr_unavailable, or failed,
including a worker timeout/crash/encrypted/malformed/text-limit document
-- is always **200**: Task 9A's own typed ``processing_status`` and
``warnings`` are preserved and returned as-is, never collapsed into an
HTTP error and never silently reported as a completed draft.

Independent-review correction of ``c60cc80``, Finding 1: the actual,
potentially long-running ``extract_document(...)`` call (which blocks
waiting on a worker-pool slot and on the isolated child process's
response) is run via ``anyio.to_thread.run_sync`` -- never called inline
in this ``async def`` handler -- so it cannot block the shared event loop
for the duration of one document's processing. See ``_run_extraction``
below for the cancellation semantics this implies.
"""

from __future__ import annotations

import functools

from anyio import to_thread
from fastapi import APIRouter, Depends, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from plateproof.api.dependencies import get_app_settings, get_document_worker_pool, get_repository
from plateproof.api.documents_projection import project_draft
from plateproof.api.schemas import DocumentExtractionResponse
from plateproof.core.config import Settings
from plateproof.documents.models import Jurisdiction
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

#: The exact, closed set of field names this request ever accepts -- an
#: unexpected field name, or more than one value for either of these two,
#: is rejected outright rather than silently ignored (independent-review
#: correction of c60cc80, Finding 2). Starlette's own `max_fields`/
#: `max_files` count non-file and file parts SEPARATELY (verified against
#: the installed version's `formparsers.py`), so e.g. `restaurant_id` sent
#: twice (2 non-file parts) fits entirely within `max_fields=2` and is
#: never caught by Starlette's own parser limits at all -- only this
#: route's own exact-shape check below catches it.
_ALLOWED_FIELD_NAMES = frozenset({"restaurant_id", "file"})
_MAX_FORM_FIELDS = 2
_MAX_RESTAURANT_ID_LENGTH = 128


async def _run_extraction(
    *,
    data: bytes,
    jurisdiction: Jurisdiction,
    restaurant_id: str,
    expected_restaurant_name: str,
    pool: WorkerPool,
    settings: Settings,
) -> DocumentExtractionResponse:
    """Runs the synchronous, potentially long-running (bounded by the
    worker pool's own page/total timeouts) ``extract_document(...)`` call
    in a worker thread via ``anyio.to_thread.run_sync`` -- never inline on
    the event loop (independent-review correction of c60cc80, Finding 1).

    This thread only ever WAITS on the existing isolated worker process
    (via ``WorkerPool.submit``'s blocking pipe reads) -- it never parses a
    document itself, and cancelling this await point (e.g. a client
    disconnect) does not stop the underlying worker job or the child
    process; the worker pool's own page/total-timeout deadlines and
    retirement/cleanup logic remain the sole authority over that job's
    lifetime, exactly as they are for a normal, uncancelled request.
    """
    draft = await to_thread.run_sync(
        functools.partial(
            extract_document,
            data,
            expected_jurisdiction=jurisdiction,
            restaurant_id=restaurant_id,
            expected_restaurant_name=expected_restaurant_name,
            pool=pool,
            max_upload_bytes=settings.documents_max_upload_bytes,
            max_pages=settings.documents_max_pages,
            max_pixels=settings.documents_max_pixels_per_page,
        ),
        # abandon_on_cancel=True: a cancelled await here (e.g. client
        # disconnect) returns control to this coroutine immediately --
        # letting its own `finally: await file.close()` run promptly --
        # rather than making the whole request uncancellable for the job's
        # full duration. The thread (and the worker-pool call it is
        # blocked on) is NOT killed; it keeps running, still bounded by
        # the pool's own timeouts, until it naturally returns a slot to
        # the pool.
        abandon_on_cancel=True,
    )
    if draft is None:
        # validate_upload_bytes already accepted these exact bytes before
        # this was ever called, so this should never happen in practice --
        # fails closed as a request-shape problem rather than ever
        # fabricating a draft.
        raise DocumentRequestError("the uploaded file could not be processed")
    return project_draft(draft)


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

    # Exact multipart shape: every field name present must be one of the
    # two allowed names, and each of those must appear exactly once. This
    # is checked -- and every file Starlette actually parsed is closed --
    # before any other validation, so a malformed shape never leaves a
    # spooled UploadFile handle open.
    #
    # Independent-review correction of c60cc80's own Finding-2 fix: closing
    # only ``form.get("file")`` misses every OTHER UploadFile a malformed
    # request can make Starlette parse -- a second file under an unexpected
    # field name, a second file under "file" itself (form.get() returns
    # only the first value for a repeated key), or the request's only file
    # arriving entirely under a field name outside the allowed set. Every
    # UploadFile instance ``form.multi_items()`` actually returned is
    # collected up front and closed in the ``finally`` block below,
    # regardless of which field name it was parsed under.
    field_counts: dict[str, int] = {}
    files_to_close: list[StarletteUploadFile] = []
    for key, value in form.multi_items():
        field_counts[key] = field_counts.get(key, 0) + 1
        if isinstance(value, StarletteUploadFile):
            files_to_close.append(value)
    file = form.get("file")

    try:
        unexpected_fields = set(field_counts) - _ALLOWED_FIELD_NAMES
        if unexpected_fields:
            raise DocumentRequestError("unexpected form field present")
        if field_counts.get("file", 0) > 1:
            raise DocumentTooLargeError("only one document file is allowed")
        if field_counts.get("restaurant_id", 0) > 1:
            raise DocumentRequestError("restaurant_id must be provided exactly once")

        restaurant_id = form.get("restaurant_id")
        if not isinstance(restaurant_id, str) or not restaurant_id.strip():
            raise DocumentRequestError("restaurant_id is required")
        if len(restaurant_id) > _MAX_RESTAURANT_ID_LENGTH:
            raise DocumentTooLargeError("restaurant_id exceeds the maximum allowed length")
        if not isinstance(file, StarletteUploadFile):
            raise DocumentRequestError("exactly one document file is required")

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

        return await _run_extraction(
            data=data,
            jurisdiction=jurisdiction,
            restaurant_id=restaurant_id,
            expected_restaurant_name=expected_restaurant_name,
            pool=pool,
            settings=settings,
        )
    finally:
        # Closed on every exit path -- success, any validation rejection
        # (including one raised before the restaurant/file shape was even
        # confirmed), RestaurantNotFoundError, cancellation/disconnect, or
        # any other exception above -- for every UploadFile Starlette
        # actually parsed, under any field name, not just "file".
        for parsed_file in files_to_close:
            await parsed_file.close()

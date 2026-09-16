"""Fixed absolute ceilings and default operational values for Task 9
document processing.

Every operational limit an administrator can configure (via
``plateproof.core.config.Settings``) is validated against a corresponding
``ABSOLUTE_*``/``MAX_*``/``MIN_*`` constant here and can only narrow it,
never widen it -- the same two-level pattern already used for
``ABSOLUTE_MAX_QUESTION_LENGTH``/``copilot_max_question_length`` in Task 8B.

Frame-size ceilings are deliberately separate per Finding 3 of the
independent review of commit 735d4a3: a small JSON control-frame ceiling
(structured messages only), a much larger document-byte ceiling (the
uploaded file itself), and a smaller preview-frame ceiling (one rendered
preview image) are three different numbers that must never be conflated.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Frame-size ceilings (Finding 3): JSON control frames, raw document bytes,  #
# and preview images are three separate budgets.                            #
# --------------------------------------------------------------------------- #

#: Small, structured JSON protocol messages only (job_request/job_response
#: metadata, job_error, page_progress) -- never the document itself.
MAX_JSON_FRAME_BYTES = 8 * 1024 * 1024

#: The raw uploaded document, parent -> worker. An administrator's
#: configured operational limit (Settings.documents_max_upload_bytes) may
#: not exceed this.
ABSOLUTE_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024

#: One rendered preview image, worker -> parent. Deliberately much smaller
#: than the document ceiling -- a preview is a small, bounded raster, never
#: a full-resolution copy of the page.
ABSOLUTE_MAX_PREVIEW_FRAME_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_PREVIEW_FRAME_BYTES = 512 * 1024

# --------------------------------------------------------------------------- #
# Page / pixel / text ceilings                                               #
# --------------------------------------------------------------------------- #

ABSOLUTE_MAX_PAGES = 30
DEFAULT_MAX_PAGES = 10

#: Per-page render-pixel ceiling, checked before any raster is allocated.
ABSOLUTE_MAX_PIXELS_PER_PAGE = 60_000_000
DEFAULT_MAX_PIXELS_PER_PAGE = 40_000_000

#: Cumulative extracted-text ceiling across the whole document (all pages
#: combined), not per page -- processing stops once exceeded.
ABSOLUTE_MAX_TEXT_BYTES_PER_DOCUMENT = 2_000_000
DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT = 500_000

# --------------------------------------------------------------------------- #
# Preview ceilings (Finding 7)                                               #
# --------------------------------------------------------------------------- #

ABSOLUTE_MAX_PREVIEW_WIDTH_PX = 1200
DEFAULT_MAX_PREVIEW_WIDTH_PX = 600
ABSOLUTE_MAX_PREVIEW_HEIGHT_PX = 1600
DEFAULT_MAX_PREVIEW_HEIGHT_PX = 800
ABSOLUTE_MAX_PREVIEW_PIXELS = ABSOLUTE_MAX_PREVIEW_WIDTH_PX * ABSOLUTE_MAX_PREVIEW_HEIGHT_PX
ABSOLUTE_MAX_PREVIEW_PAGES = 10
DEFAULT_MAX_PREVIEW_PAGES = 5
ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_TOTAL_PREVIEW_BYTES = 3 * 1024 * 1024

# --------------------------------------------------------------------------- #
# Worker pool ceilings (Finding 6)                                           #
# --------------------------------------------------------------------------- #

MIN_POOL_SIZE = 1
ABSOLUTE_MAX_POOL_SIZE = 4
DEFAULT_POOL_SIZE = 2

ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS = 60.0
DEFAULT_PAGE_TIMEOUT_SECONDS = 20.0
ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS = 180.0
DEFAULT_TOTAL_TIMEOUT_SECONDS = 60.0
MIN_KILL_GRACE_SECONDS = 0.1
ABSOLUTE_MAX_KILL_GRACE_SECONDS = 10.0
DEFAULT_KILL_GRACE_SECONDS = 2.0

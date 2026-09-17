"""Shared, transport-agnostic exceptions for the serving layer.

Both the FastAPI routes and Streamlit pages catch these and decide their own
presentation (HTTP status code vs. an inline page message) -- the exceptions
themselves carry no HTTP concepts and no filesystem paths in their messages.
"""

from __future__ import annotations


class ServingError(Exception):
    """Base class for all serving-layer errors."""


class InvalidPaginationError(ServingError):
    pass


class InvalidQueryError(ServingError):
    pass


class InvalidQuestionError(ServingError):
    """A Copilot question exceeded the administrator-configured operational
    limit (``Settings.copilot_max_question_length``) after already passing
    the fixed absolute public-safety ceiling enforced by the request
    schema. Never includes the offending question text."""


class RestaurantNotFoundError(ServingError):
    def __init__(self, restaurant_id: str) -> None:
        super().__init__(f"restaurant not found: {restaurant_id}")
        self.restaurant_id = restaurant_id


class ModelCardNotFoundError(ServingError):
    pass


class DatastoreUnavailableError(ServingError):
    pass


class UnsafePathError(ServingError):
    """A configured path resolved outside its required containing directory,
    or otherwise failed a filesystem-safety check. Never includes the
    offending path in the message (that's for the administrator's own logs,
    not a user-facing string)."""


class DocumentRequestError(ServingError):
    """The owner-document-extraction request itself was malformed: no file
    part, no ``restaurant_id`` field, more than one field of either kind
    beyond what a single-document upload requires, or a byte string that
    isn't even one of the three supported document signatures. Distinct
    from :class:`DocumentTooLargeError` (a resource-limit condition) and
    from a failed *extraction* (which is never an exception -- see
    ``plateproof.documents.service.extract_document``'s typed
    ``ExtractionDraft.processing_status``/``warnings``)."""


class DocumentTooLargeError(ServingError):
    """A second file part, an oversized non-file field, or file content
    exceeding the configured operational byte limit -- always a resource-
    limit condition (HTTP 413), never a request-shape problem."""


class UnsupportedDocumentTypeError(ServingError):
    """The uploaded bytes don't start with the PDF/PNG/JPEG magic-byte
    signature :func:`plateproof.documents.validation.sniff_media_type`
    requires. Never includes the declared filename or MIME type -- neither
    is ever trusted in the first place."""

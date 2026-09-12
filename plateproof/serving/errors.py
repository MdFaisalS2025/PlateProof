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

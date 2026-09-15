"""OCR engine abstraction. Called only from inside a worker process, only
when embedded PDF text is unavailable and OCR is enabled/available."""

from __future__ import annotations

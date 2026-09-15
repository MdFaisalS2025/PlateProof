"""Process-isolation boundary for untrusted document parsing.

PDFium, Pillow, OpenCV, and RapidOCR are only ever called from inside a
worker process spawned by :mod:`plateproof.documents.worker.pool`, via the
non-pickle wire protocol in :mod:`plateproof.documents.worker.protocol`. The
calling process (FastAPI, Streamlit, or a test) never parses a hostile
document directly.
"""

from __future__ import annotations

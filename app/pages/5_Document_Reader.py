"""PlateProof Document Reader -- owner document review workflow (Task 9B).

Calls the SAME ``plateproof.documents.service.extract_document(...)``
function Task 9A built and fully tested, submitted to THIS Streamlit
process's own worker pool (``theme.document_worker_pool()``) -- this page
never imports ``plateproof.documents.pdf``, ``plateproof.documents.images``,
or any ``plateproof.documents.ocr`` submodule, and never calls PDFium,
Pillow's image decoder, RapidOCR, ONNX Runtime, or OpenCV itself. See
``tests/documents/test_task9b_boundary.py`` for the import-graph test that
enforces this structurally, not just by convention.

Raw upload bytes are read once, submitted to the worker pool, and never
assigned to any ``st.session_state`` key -- the page only ever holds the
resulting ``ExtractionDraft`` (already-validated structured data), the
worker-generated bounded previews it carries, and correction/confirmation
state. Session state tied to Task 9 is cleared the moment the selected
restaurant changes, or when the user chooses "Start Over" -- reusing the
same cross-restaurant clearing pattern as the Task 8B Owner Copilot page.

Previews are rendered via a ``data:`` URI passed to ``st.image()``, which
Streamlit's own ``image_to_url`` returns unmodified without ever invoking
Pillow (verified directly against the installed Streamlit version's
``elements/lib/image_utils.py``) -- the actual pixel decode happens only in
the viewer's own browser, on bytes the worker already generated and
validated (Task 9A, Finding 7), never on the original uploaded file. Task
9A's preview validator checks PNG container structure (signature, chunk
CRCs/ordering, dimensions) -- it does not fully decode image pixels, so
this is not a claim that the preview bytes are exhaustively proven safe to
decode, only that they are worker-generated, bounded, and never the
original hostile upload; the residual pixel-decode risk is deliberately
pushed to the browser's own image decoder, the standard trust boundary for
any web application displaying an image, rather than run inside this
Streamlit process.

There is no confirmation endpoint and no server-side persistence: choosing
"I confirm this record" only enables a client-side JSON download, tagged
``record_status="user_submitted"``, with a disclaimer that it is not an
official inspection record and does not affect PlateProof's own
predictions.
"""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import (  # noqa: E402
    configure_page,
    document_worker_pool,
    get_settings,
    render_independence_footer,
)
from theme import repository as get_repository  # noqa: E402

from plateproof.documents.corrections import validate_correction  # noqa: E402
from plateproof.documents.florida_extractor import FLORIDA_LABELS  # noqa: E402
from plateproof.documents.models import ExtractionDraft  # noqa: E402
from plateproof.documents.nyc_extractor import NYC_LABELS  # noqa: E402
from plateproof.documents.service import extract_document  # noqa: E402
from plateproof.serving.display import (  # noqa: E402
    DOCUMENT_PRIVACY_NOTICE,
    USER_SUBMITTED_RECORD_DISCLAIMER,
)

_SEARCH_UNAVAILABLE_MESSAGE = "Restaurant search is temporarily unavailable."
_RESTAURANT_LOAD_FAILED_MESSAGE = "This restaurant could not be loaded."
_WORKER_TIMEOUT_MESSAGE = (
    "Processing this document took too long and was stopped. Please try again."
)
_WORKER_CRASHED_MESSAGE = "This document could not be processed due to an internal error."
_UPLOAD_TOO_LARGE_MESSAGE = "This file is too large for PlateProof to process."
_UNSUPPORTED_TYPE_MESSAGE = "PlateProof only accepts PDF, PNG, or JPEG files."

_ACCEPTED_EXTENSIONS = ("pdf", "png", "jpg", "jpeg")

_STATE_KEYS = (
    "doc_draft",
    "doc_draft_restaurant_id",
    "doc_corrections",
    "doc_uploader_nonce",
)


def _labels_for(jurisdiction: str) -> dict[str, tuple[str, ...]]:
    return NYC_LABELS if jurisdiction == "nyc" else FLORIDA_LABELS


def _clear_draft_state() -> None:
    st.session_state["doc_draft"] = None
    st.session_state["doc_draft_restaurant_id"] = None
    st.session_state["doc_corrections"] = {}
    st.session_state.pop("doc_confirm_checkbox", None)


def _release_uploaded_file() -> None:
    """Bumps the file_uploader's own key so a fresh, empty uploader widget
    renders next -- and explicitly drops the OLD key's entry from
    session_state, since Streamlit does not itself prune an unrendered
    widget's stored value. Called after every extraction attempt (success
    or failure) and on a restaurant switch/Start Over, so raw upload bytes
    never outlive the single call that submitted them to the worker
    manager."""
    old_nonce = st.session_state.get("doc_uploader_nonce", 0)
    st.session_state.pop(f"doc_file_uploader_{old_nonce}", None)
    st.session_state["doc_uploader_nonce"] = old_nonce + 1


def _clear_document_state() -> None:
    _clear_draft_state()
    _release_uploaded_file()


def _preview_data_uri(preview_png: bytes) -> str:
    """A ``data:`` URI string -- passed to ``st.image()``, which returns a
    recognized ``data:`` URL unmodified without ever calling
    ``PIL.Image.open`` (see module docstring)."""
    encoded = base64.b64encode(preview_png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _asdict_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _asdict_json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _asdict_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_asdict_json_safe(v) for v in value]
    if isinstance(value, bytes):
        # Never included: PageMetadata.preview_png is deliberately dropped
        # from the downloadable artifact (it is a display aid, not part of
        # the user-submitted record).
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _build_downloadable_record(
    draft: ExtractionDraft, corrections: dict[str, str]
) -> dict[str, Any]:
    machine_candidates = {
        name: {
            "value": _asdict_json_safe(candidate.value),
            "display_value": candidate.display_value,
            "confidence_label": candidate.confidence_label,
        }
        for name, candidate in draft.candidates.items()
    }
    return {
        "restaurant_id": draft.restaurant_id,
        "jurisdiction": draft.jurisdiction_expected,
        "confirmed_at": datetime.now(UTC).isoformat(),
        "machine_candidates": machine_candidates,
        "user_corrections": dict(corrections),
        "violations": [_asdict_json_safe(v) for v in draft.violations],
        "record_status": "user_submitted",
        "disclaimer": USER_SUBMITTED_RECORD_DISCLAIMER,
    }


configure_page("Document Reader")
st.title("📄 PlateProof Document Reader")
st.write(
    "Upload a copy of your own official inspection document to extract its data for your "
    "review. PlateProof never guesses a value it cannot find evidence for."
)
st.info(
    "**PlateProof does not verify restaurant ownership.** Anyone can look up any documented "
    "restaurant here; this page does not confirm you own or operate it."
)

for key, default in (
    ("doc_draft", None),
    ("doc_draft_restaurant_id", None),
    ("doc_corrections", {}),
    ("doc_search_results", []),
    ("doc_uploader_nonce", 0),
):
    if key not in st.session_state:
        st.session_state[key] = default

settings = get_settings()

try:
    repo = get_repository()
except Exception:  # noqa: BLE001 - narrow: only wraps the repository accessor itself
    st.error(_SEARCH_UNAVAILABLE_MESSAGE)
    render_independence_footer()
    st.stop()

# --- Restaurant selection (required before any upload UI appears) ------- #
st.subheader("1. Select your restaurant")
with st.expander("Search by name"):
    search_query = st.text_input("Restaurant name", key="doc_search_query")
    if st.button("Search", key="doc_search_button"):
        if not search_query:
            st.session_state["doc_search_results"] = []
        else:
            try:
                result = repo.search_restaurants(
                    query=search_query, jurisdiction=None, limit=10, offset=0
                )
            except Exception:  # noqa: BLE001 - narrow: only wraps this one repository call
                st.error(_SEARCH_UNAVAILABLE_MESSAGE)
                st.session_state["doc_search_results"] = []
            else:
                st.session_state["doc_search_results"] = [dict(r) for r in result.results]
                if not result.results:
                    st.info("No restaurants matched that search.")

    search_results: list[dict[str, Any]] = st.session_state["doc_search_results"]
    for row in search_results:
        result_restaurant_id = str(row.get("restaurant_id", ""))
        name = row.get("name") or "Unnamed restaurant"
        jurisdiction_label = str(row.get("jurisdiction") or "").upper()
        address = row.get("address") or ""
        city = row.get("city") or ""
        location = ", ".join(part for part in (address, city) if part)
        # Each result's label embeds its own restaurant ID, so two
        # identically-named restaurants remain distinguishable by their ID
        # and address/city -- and, being a plain st.button, each result is
        # reachable and activatable by keyboard (Tab + Enter/Space) exactly
        # like every other Streamlit button on this page.
        label = f"Select: {name} -- {jurisdiction_label} -- {location} (ID: {result_restaurant_id})"
        if st.button(label, key=f"doc_select_{result_restaurant_id}"):
            st.session_state["doc_restaurant_id"] = result_restaurant_id

if "doc_restaurant_id" not in st.session_state:
    st.session_state["doc_restaurant_id"] = ""

restaurant_id = st.text_input(
    "Official restaurant ID (e.g. nyc:12345678 or florida:HR1234567)",
    key="doc_restaurant_id",
)

# The one rule every entry path (typing, search selection) follows: the
# moment the active restaurant id no longer matches the id an existing
# draft was produced for, every Task-9-specific piece of state is cleared
# together -- draft, corrections, preview data, and the confirmation
# checkbox -- before anything new is rendered. Mirrors the Task 8B Owner
# Copilot page's identical cross-restaurant clearing rule. Guarded on "a
# draft actually exists for a DIFFERENT restaurant" (not merely "no draft
# yet") so this never fires -- and never spuriously releases an
# in-progress upload -- before any extraction has happened for this
# restaurant at all.
_existing_draft_restaurant_id = st.session_state.get("doc_draft_restaurant_id")
if _existing_draft_restaurant_id is not None and _existing_draft_restaurant_id != restaurant_id:
    _clear_document_state()

if not restaurant_id:
    st.info("Enter a restaurant ID above, or search by name, to continue.")
    render_independence_footer()
    st.stop()

try:
    restaurant = repo.get_restaurant(restaurant_id)
except Exception:  # noqa: BLE001 - narrow: only wraps this one repository call
    st.error(_RESTAURANT_LOAD_FAILED_MESSAGE)
    _clear_document_state()
    render_independence_footer()
    st.stop()

if restaurant is None:
    st.error("No restaurant found for that ID.")
    _clear_document_state()
    render_independence_footer()
    st.stop()

restaurant_any: dict[str, Any] = dict(restaurant)
jurisdiction = restaurant_any["jurisdiction"]
expected_name = str(restaurant_any.get("name") or "")

st.success(f"Selected: **{expected_name or 'Unnamed restaurant'}** ({jurisdiction.upper()})")

if st.button("Start Over", key="doc_start_over"):
    _clear_document_state()
    # The `doc_restaurant_id` text_input widget already instantiated
    # earlier in this same script run, so its session_state key cannot be
    # directly assigned a new value here -- only deleted. Deleting it and
    # rerunning lets the widget fall back to its own default (empty
    # string) on the next run.
    st.session_state.pop("doc_restaurant_id", None)
    st.rerun()

# --- Privacy notice, shown before any upload control ---------------------- #
st.subheader("2. Upload your document")
st.warning(DOCUMENT_PRIVACY_NOTICE)

max_upload_bytes = settings.documents_max_upload_bytes
st.caption(
    f"Accepted formats: PDF, PNG, JPEG. Maximum size: {max_upload_bytes // (1024 * 1024)} MB."
)

uploaded_file = st.file_uploader(
    "Choose a file",
    type=list(_ACCEPTED_EXTENSIONS),
    key=f"doc_file_uploader_{st.session_state['doc_uploader_nonce']}",
)

if uploaded_file is not None and st.session_state.get("doc_draft") is None:
    # Streamlit's own file_uploader exposes .size without requiring a
    # .read() first -- a UI-level short-circuit on top of the shared
    # validator, so an upload already known to be oversized never reaches
    # the worker manager at all.
    if uploaded_file.size is not None and uploaded_file.size > max_upload_bytes:
        st.error(_UPLOAD_TOO_LARGE_MESSAGE)
        _release_uploaded_file()
    else:
        with st.spinner("PlateProof is reading your document..."):
            data = uploaded_file.getvalue()
            # Never assigned to any st.session_state key -- `data` is a
            # local variable, submitted once below, and goes out of scope
            # (and out of the uploader's own retained buffer, once the
            # uploader's key changes on the next rerun) once this block
            # finishes.
            try:
                pool = document_worker_pool()
                draft = extract_document(
                    data,
                    expected_jurisdiction=jurisdiction,
                    restaurant_id=restaurant_id,
                    expected_restaurant_name=expected_name,
                    pool=pool,
                    max_upload_bytes=max_upload_bytes,
                    max_pages=settings.documents_max_pages,
                    max_pixels=settings.documents_max_pixels_per_page,
                )
            except Exception:  # noqa: BLE001 - narrow: only wraps this one call
                st.error(_WORKER_CRASHED_MESSAGE)
                draft = None
            del data

        if draft is None:
            st.error(_UNSUPPORTED_TYPE_MESSAGE)
        else:
            st.session_state["doc_draft"] = draft
            st.session_state["doc_draft_restaurant_id"] = restaurant_id
            st.session_state["doc_corrections"] = {}
        # Whatever the outcome, this upload attempt is over -- release
        # Streamlit's own internal reference to it so raw upload bytes
        # never outlive this single call (module docstring).
        _release_uploaded_file()

# --- Extraction results ---------------------------------------------------- #
draft: ExtractionDraft | None = (
    st.session_state.get("doc_draft")
    if st.session_state.get("doc_draft_restaurant_id") == restaurant_id
    else None
)

if draft is not None:
    st.subheader("3. Review extracted data")

    if draft.processing_status == "failed":
        code = draft.warnings[0].code if draft.warnings else ""
        if code == "worker_timeout":
            st.error(_WORKER_TIMEOUT_MESSAGE)
        elif code == "worker_crashed":
            st.error(_WORKER_CRASHED_MESSAGE)
        else:
            message = draft.warnings[0].message if draft.warnings else _WORKER_CRASHED_MESSAGE
            st.error(message)
    else:
        if draft.processing_status == "ocr_unavailable":
            st.warning(
                "This document appears to need text recognition (OCR) to read, but OCR is "
                "not available right now. Try a document with selectable text instead."
            )
        if draft.jurisdiction_mismatch:
            st.warning(
                "The document appears to be from a different jurisdiction than this "
                "restaurant's own jurisdiction. Please confirm you uploaded the right document."
            )
        if not draft.restaurant_identity_corroborated:
            st.info(
                "PlateProof could not confirm the restaurant name on the document matches "
                "this restaurant's own documented name."
            )

        labels = _labels_for(draft.jurisdiction_expected)
        corrections: dict[str, str] = st.session_state["doc_corrections"]

        st.write("**Extracted fields** (machine value alongside your correction, if any):")
        for field_name in labels:
            candidate = draft.candidates.get(field_name)
            display_label = labels[field_name][0]
            col_machine, col_correction = st.columns(2)
            with col_machine:
                if candidate is not None:
                    st.text_input(
                        f"{display_label} (extracted)",
                        value=candidate.display_value or "",
                        disabled=True,
                        key=f"doc_machine_{field_name}",
                    )
                    st.caption(f"Confidence: {candidate.confidence_label}")
                    for evidence in candidate.evidence:
                        st.caption(f"Page {evidence.page}: “{evidence.excerpt}”")
                else:
                    st.text_input(
                        f"{display_label} (extracted)",
                        value="(not found)",
                        disabled=True,
                        key=f"doc_machine_{field_name}",
                    )
            with col_correction:
                correction_value = st.text_input(
                    f"{display_label} (your correction)",
                    value=corrections.get(field_name, ""),
                    key=f"doc_correction_{field_name}",
                )
                if correction_value:
                    corrections[field_name] = correction_value
                elif field_name in corrections:
                    del corrections[field_name]
        st.session_state["doc_corrections"] = corrections

        if draft.missing_fields:
            st.warning(f"Missing required fields: {', '.join(draft.missing_fields)}")
        if draft.ambiguities:
            st.warning(
                "Ambiguous fields (multiple conflicting values found, not guessed): "
                + ", ".join(a.field_name for a in draft.ambiguities)
            )

        if draft.violations:
            st.write("**Violation rows:**")
            for violation in draft.violations:
                with st.container(border=True):
                    st.write(f"Code: `{violation.raw_code_text}`")
                    if violation.code is None:
                        st.warning("This code could not be matched to a known code -- unresolved.")
                    if violation.description:
                        st.caption(violation.description)
                    if violation.critical is not None:
                        st.caption(f"Critical: {'Yes' if violation.critical else 'No'}")
                    st.caption(f"Confidence: {violation.confidence_label}")

        unresolved_violations = any(v.code is None for v in draft.violations)

        if draft.pages:
            st.write("**Page previews** (generated and validated by PlateProof's isolated worker):")
            preview_cols = st.columns(min(3, len(draft.pages)))
            for index, page in enumerate(draft.pages):
                if page.preview_png is None:
                    continue
                with preview_cols[index % len(preview_cols)]:
                    st.image(
                        _preview_data_uri(page.preview_png),
                        caption=f"Page {page.page_number}",
                    )

        # --- Validate every correction through the Task 9A validator ---- #
        invalid_correction_fields: list[str] = []
        for field_name, raw_value in corrections.items():
            validated, reason = validate_correction(field_name, raw_value, draft)
            if reason is not None or (validated is not None and validated.parse_status != "parsed"):
                invalid_correction_fields.append(field_name)
        if invalid_correction_fields:
            st.error(f"Invalid corrections for: {', '.join(invalid_correction_fields)}")

        blocking_reasons: list[str] = []
        if draft.jurisdiction_mismatch:
            blocking_reasons.append("jurisdiction mismatch")
        if draft.missing_fields:
            blocking_reasons.append("missing required fields")
        if unresolved_violations:
            blocking_reasons.append("unresolved violation codes")
        if invalid_correction_fields:
            blocking_reasons.append("invalid corrections")
        if draft.processing_status == "ocr_unavailable":
            blocking_reasons.append("OCR unavailable")
        if draft.processing_status == "failed":
            blocking_reasons.append("processing failed")
        if not draft.confirmable:
            blocking_reasons.append("PlateProof could not confirm this draft is reviewable")

        st.subheader("4. Confirm and download")
        if blocking_reasons:
            st.error("Confirmation is blocked: " + ", ".join(sorted(set(blocking_reasons))) + ".")
        confirmed = st.checkbox(
            "I confirm I have reviewed this extracted data and any corrections above, and "
            "I understand this is not an official inspection record.",
            value=False,
            key="doc_confirm_checkbox",
            disabled=bool(blocking_reasons),
        )
        if confirmed and not blocking_reasons:
            record = _build_downloadable_record(draft, corrections)
            st.download_button(
                "Download my submitted record (JSON)",
                data=json.dumps(record, indent=2),
                file_name=f"plateproof_document_record_{draft.draft_id}.json",
                mime="application/json",
            )
            st.caption(USER_SUBMITTED_RECORD_DISCLAIMER)

render_independence_footer()

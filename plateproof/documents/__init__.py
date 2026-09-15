"""Task 9: local, deterministic owner-document extraction (hard boundary).

Everything in this package is a *proposal* requiring explicit human
confirmation, never an official record. Concretely:

- Nothing here ever writes to an official NYC/Florida record, trains or
  retrains a Task 6 model, changes a risk score, or enters the Task 8
  knowledge graph / Copilot evidence. This package must never import
  ``plateproof.models`` (Task 6), ``plateproof.graph`` or ``plateproof.copilot``
  (Task 8), or any ``plateproof.serving`` write path.
- A confirmed record's ``record_status`` can only ever be
  ``"user_submitted"`` -- there is no code path that elevates one to
  ``"official"``.
- Nothing here is persisted server-side; every draft and correction is
  session-local and ephemeral by construction (no database write, no file
  write of uploaded content or extracted values).
- Untrusted document bytes are parsed ONLY inside a bounded, killable
  worker process (``plateproof.documents.worker``) -- never in the calling
  API/UI process. See ``worker/__init__.py`` for the isolation boundary.
"""

from __future__ import annotations

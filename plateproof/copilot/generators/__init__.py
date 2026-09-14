"""Optional local-model intent helpers (Task 8B).

Hard boundary, enforced by construction, not merely by convention: nothing
in this package -- or anything it calls -- ever produces answer text, a
``Claim``, a ``Citation``, an evidence value, a restaurant fact, a
forecast, or a legal/medical/compliance/safety conclusion. The only thing
that can ever cross out of this package into
:class:`~plateproof.copilot.service.CopilotService` is a
:class:`~plateproof.copilot.intent_validation.IntentProposal` that has
already survived every check in
:func:`~plateproof.copilot.intent_validation.validate_intent_proposal`.
Every downstream factual sentence is still built the same way it was in
Task 8A: by a deterministic claim builder, from graph/corpus evidence,
through a fixed rendering template.

Nothing in this package ever touches the filesystem, a shell, a browser, a
database write, another network host, a model artifact, or a secret. Its
one network call (to a configured local loopback Ollama server) is
strictly bounded -- see ``plateproof.copilot.generators.ollama``.
"""

from __future__ import annotations

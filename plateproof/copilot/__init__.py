"""PlateProof Copilot (Task 8A): deterministic, evidence-grounded answers
over the knowledge graph (``plateproof.graph``) and the reviewed official
guidance corpus (``plateproof.copilot.corpus``).

Every final factual sentence this package produces comes from a typed
``Claim`` built by code from graph/corpus data and rendered through a
fixed template -- never from freeform generated text. No optional local
model is wired in during Task 8A; that is Task 8B's responsibility, kept
strictly bounded to intent/filter suggestion (see the Task 8 plan).
"""

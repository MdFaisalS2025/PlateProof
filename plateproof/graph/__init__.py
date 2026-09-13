"""PlateProof's deterministic temporal knowledge graph (Task 8A).

Built in-process from Task 7's already-processed Parquet tables -- never
from a raw upstream extract, and never persisted as a pickled/joblib
artifact the web process would need to deserialize (that trust boundary is
exactly what ``plateproof.serving.model_registry_service`` was hardened to
avoid reopening; see Task 8 planning notes). A graph is either built fresh
from in-memory frames (:func:`plateproof.graph.builder.build_graph`) or
read from a processed-data directory and cached by a cheap source
fingerprint (:class:`plateproof.graph.builder.GraphService`).

This package has no import from ``plateproof.models`` or
``plateproof.features`` and must never gain one -- graph-derived output is
never fed back into Task 6 model training (see
``tests/graph/test_no_model_training_dependency.py``).
"""

# PlateProof container image -- packaging only. Building and running this
# image locally (`docker build` / `docker run`) costs nothing and needs no
# account of any kind. It does NOT, by itself, constitute a public
# deployment: pushing it to a registry or running it on a cloud host is a
# separate, explicit decision -- see docs/deployment.md before doing either.
#
# Runs either the FastAPI service or the Streamlit UI, selected at `docker
# run` time via the PLATEPROOF_TARGET env var (default "api") -- matching
# scripts/run_app.py's own --target switch, which this image calls directly
# rather than reimplementing. Each container is one process, matching
# "each application process owns its own document worker pool"
# (README.md's Task 9 deployment-requirements section): run two containers
# (one api, one streamlit) rather than one container trying to be both.
#
# A reverse proxy / body-size-limiting layer in front of the "api" target
# is a REQUIRED operational precondition (see README.md and
# plateproof/api/routes/documents.py) -- this image does not include one,
# and does not claim to.
FROM python:3.12-slim AS base

# Real OS-level memory/CPU limits are a required complement to Task 9's
# worker-process design (README.md's Task 9 section explains why Python
# cannot enforce this itself) -- set them on the container runtime
# (`docker run --memory=... --cpus=...`, or the equivalent in your
# orchestrator), not in this Dockerfile.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for better layer caching -- pyproject.toml lists every
# dependency's license and why it's free/local (no paid API, no GPU
# requirement); see that file for the full inventory.
COPY pyproject.toml ./
COPY plateproof ./plateproof
COPY scripts ./scripts
COPY app ./app
COPY data/reference ./data/reference
COPY .streamlit ./.streamlit

# Runtime only -- the "dev" extra (pytest/ruff/mypy) is deliberately not
# installed in the image.
RUN pip install --no-cache-dir .

# Placeholders so Settings' default relative paths (data/raw, data/interim,
# data/processed, models) resolve to real, empty, writable directories
# rather than missing ones -- every one of them is optional at runtime (see
# README.md); mount a volume over data/processed (and models, if scoring
# offline inside the container) to use real data.
RUN mkdir -p data/raw data/interim data/processed models

# Runs as a non-root user -- standard container hardening, unrelated to any
# PlateProof-specific trust boundary.
RUN useradd --create-home --uid 1000 plateproof && chown -R plateproof:plateproof /app
USER plateproof

ENV PLATEPROOF_TARGET=api \
    PORT=8000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8000 8501

# scripts/run_app.py's --target selects which process this container runs;
# PORT is read for both (its own default differs by target -- 8000 for
# api, 8501 for streamlit -- overridden here uniformly via $PORT so a
# platform that injects its own PORT still works for either target).
# STREAMLIT_SERVER_ADDRESS=0.0.0.0 is Streamlit's own standard env var for
# binding to all interfaces inside a container (run_app.py's streamlit
# path does not take a --host flag -- unlike the api target -- since
# Streamlit's own CLI has no such flag; this is the documented way to
# achieve the same result without patching run_app.py).
ENTRYPOINT ["sh", "-c", "python -m scripts.run_app --target \"$PLATEPROOF_TARGET\" --host 0.0.0.0 --port \"$PORT\""]

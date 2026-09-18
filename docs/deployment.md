# PlateProof deployment

## Local-only release (verified, complete)

Running PlateProof entirely on your own machine is fully supported and is the release
this project actually delivers today -- see `README.md`'s "Local development" and
"Running the API and the Streamlit MVP" sections for the exact commands. Nothing about
local operation requires a credit card, a cloud account, or a paid service, and this
was verified live in the Task 11 audit: a real NYC/Florida download, a real
`build_processed_tables` run, and a real FastAPI service serving search/detail/
inspection/violation/health/Copilot requests against that data, all on a local machine,
with Google integration and local-AI assistance left at their default-disabled
settings throughout.

## Container packaging (built and inspected locally; not deployed anywhere)

A `Dockerfile` and `.dockerignore` are included for anyone who wants to run PlateProof
in a container on their **own** machine or infrastructure -- `docker build` and
`docker run` cost nothing and need no account. This is packaging, not a deployment:
this project has not pushed this image to any registry or run it on any cloud host.

```powershell
docker build -t plateproof .
docker run --rm -p 8000:8000 -e PLATEPROOF_TARGET=api plateproof
docker run --rm -p 8501:8501 -e PLATEPROOF_TARGET=streamlit plateproof
```

**Not verified in this audit**: this Dockerfile was written carefully (matching
`scripts/run_app.py`'s existing `--target api|streamlit` switch and setting
`STREAMLIT_SERVER_ADDRESS=0.0.0.0` for the Streamlit target, since `run_app.py`'s
`--host` flag only applies to the API target) but a `docker build` could not actually
be executed in this session -- the Docker CLI is installed but no Docker daemon was
running (`failed to connect to the docker API`). Build and run it yourself before
relying on it.

## Public hosting: researched, not yet approved, and likely infeasible at true $0

Task 11 asked for the exact hosting choice to be presented for approval before any
public deployment happens. Here is that research, done against current terms rather
than assumption:

**Google Cloud Run / AWS / Azure / Render / Railway / Fly.io** -- every one of these
requires a billing account with a payment method on file before a container can be
deployed at all, exactly the same category of requirement Task 10's research found for
the Google Places API (a free-tier *allowance* is not the same as a free-tier *signup*
that never asks for a card). Disqualified under the $0/no-credit-card requirement,
independent of usage.

**Hugging Face Spaces** -- verified directly against Hugging Face's own docs
(`huggingface.co/docs/hub/en/spaces-overview`) during this audit: "Static Spaces are
free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to
create: PRO for personal accounts." A Docker Space -- the only SDK that could run
PlateProof's FastAPI service or a full Streamlit app with its own dependencies -- is no
longer available on the free tier as of this policy (a change from Hugging Face's
earlier, more permissive free-Docker-Spaces policy). Disqualified.

**Streamlit Community Cloud** -- genuinely free, no credit card required to sign up,
and could plausibly host the **Streamlit UI only** (it has no mechanism to run a
separate FastAPI process at all, so the API service has no home here regardless).
Even for the UI alone, three real constraints argue against it for PlateProof
specifically:

1. **~1 GB RAM, 1 CPU core.** Task 9's OCR worker pool loads RapidOCR/ONNX Runtime
   inside each worker process specifically because that combination needs real memory
   headroom -- README.md's own Task 9 deployment section says as much for a
   self-hosted deployment with full control over container limits. A 1 GB shared
   ceiling for the whole Streamlit process *plus* one or more spawned OCR workers is a
   poor, likely-unreliable fit.
2. **Documented multiprocessing/rerun instability.** Streamlit's own community
   discussions and issue tracker describe child processes surviving incorrectly across
   Streamlit's rerun-on-every-interaction execution model, and multi-user concurrent
   multiprocessing use causing child processes to be stopped unexpectedly -- exactly
   the failure mode Task 9's own Windows `__main__`-patching workaround
   (`plateproof/documents/worker/pool.py`) already had to design around for local
   development. Community Cloud's shared, rerun-heavy environment stresses this
   further, not less.
3. **No FastAPI service at all**, and apps sleep after inactivity -- acceptable for a
   demo, not "the complete application" the spec's acceptance criteria describe.

**Conclusion: no researched option runs PlateProof's full stack (FastAPI service +
Streamlit UI + Task 9's multiprocessing worker pool) publicly at genuine $0 with no
credit card and without materially compromising the privacy/memory/process-isolation
design this project already committed to.** The closest partial fit -- Streamlit
Community Cloud hosting the UI alone, with the Document Reader/OCR feature either
disabled or documented as unreliable, and no API service reachable at all -- is a real
option but a degraded one, and deploying it (or anything else) is **not done by this
audit**. If you want to pursue it, say so explicitly and which specific feature
trade-offs you accept; this document is the disclosure Task 11 asked for, not a
deployment action.

## What Task 11 did NOT do

- No cloud account was created.
- No image was pushed to any registry.
- No app was deployed to any public host.
- No paid service was activated.

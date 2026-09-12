"""Local launcher for the PlateProof MVP: the FastAPI service and/or the
Streamlit UI. Each works independently; Streamlit does not require the API
process to be running (it calls the shared service layer in-process)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_api(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("plateproof.api.main:create_app", factory=True, host=host, port=port)
    return 0


def _run_streamlit(port: int) -> int:
    home = _REPO_ROOT / "app" / "Home.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(home), "--server.port", str(port)]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PlateProof API and/or Streamlit UI.")
    parser.add_argument("--target", choices=["api", "streamlit"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    if args.target == "api":
        return _run_api(args.host, args.port or 8000)
    return _run_streamlit(args.port or 8501)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Vercel entrypoint: FastAPI API plus the built Vite single-page frontend."""
from __future__ import annotations

import os
from pathlib import Path

# The hosted deployment is a public hackathon demo. This fixed token is deliberately
# public and is not suitable for protecting private or production data.
os.environ.setdefault("TELLTALE_DEMO_MODE", "1")
os.environ.setdefault("TELLTALE_SENSOR_AUTOTRAIN_OFFLINE", "true")
os.environ.setdefault("TELLTALE_DATA_DIR", "/tmp/telltale-data")

from telltale_api.main import app  # noqa: E402

_FRONTEND_DIST = Path(__file__).resolve().parent / "apps" / "web" / "dist"
if _FRONTEND_DIST.is_dir():
    # Vercel's FastAPI runtime promotes this mount to the CDN and applies the
    # index.html fallback required by React Router deep links.
    frontend = getattr(app, "frontend", None)
    if callable(frontend):
        frontend("/", directory=str(_FRONTEND_DIST), fallback="index.html")
    else:
        # Local ASGI fallback. API routes were registered first, so they keep priority.
        from fastapi import HTTPException, Request
        from fastapi.responses import FileResponse

        @app.get("/{frontend_path:path}", include_in_schema=False)
        async def serve_frontend(frontend_path: str, request: Request):
            candidate = (_FRONTEND_DIST / frontend_path).resolve()
            if candidate.is_relative_to(_FRONTEND_DIST) and candidate.is_file():
                return FileResponse(candidate)
            if "text/html" in request.headers.get("accept", ""):
                return FileResponse(_FRONTEND_DIST / "index.html")
            raise HTTPException(status_code=404, detail="Not found")

"""Industrial Automation Project Suggester - FastAPI backend.

Serves the static chat-wizard frontend and a JSON suggestion API. See
backend/engine.py for the grounded retrieval + fine-tuned-LLM logic, and
README.md for how to train/deploy.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import llm
from backend.engine import get_suggestions
from backend.schemas import SuggestRequest, SuggestResponse

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Industrial Automation Project Suggester", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_available": llm.is_available(),
        "llm_note": None if llm.is_available() else llm.unavailable_reason(),
    }


@app.post("/api/suggest", response_model=SuggestResponse)
def suggest(request: SuggestRequest):
    return get_suggestions(request, k=3)


@app.get("/api/components")
def components():
    import json

    raw = json.loads((ROOT / "data" / "components.json").read_text())
    return raw


# Static frontend last, so /api/* routes above take priority.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

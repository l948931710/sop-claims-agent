"""HTTP layer: a small JSON API and the static test console.

Sessions live in memory; this is a demo server, not a deployment.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .brain.base import Brain, BrainError
from .brain.stub import StubBrain
from .config import load_settings
from .data import Repository
from .engine import Harness
from .sop import ALLOWED_TRANSITIONS, IDENTITY_FIELDS, INTENTS, PHASES
from .state import Session

STATIC_DIR = Path(__file__).parent / "static"
MAX_SESSIONS = 200

settings = load_settings()
repo = Repository(settings.fixtures_dir)
harness = Harness(repo, settings)

class RevalidatedStatic(StaticFiles):
    """Static files the browser must revalidate, so an updated console is never shown from cache."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="SOP claims agent", docs_url="/api/docs", redoc_url=None)
app.mount("/static", RevalidatedStatic(directory=STATIC_DIR), name="static")

_sessions: "OrderedDict[str, tuple[Session, str]]" = OrderedDict()  # id -> (session, brain kind)
_sessions_lock = threading.Lock()
_stub = StubBrain()
_server_brain: Brain | None = None


class NewSession(BaseModel):
    brain: Literal["claude", "stub"] | None = None
    consent_scenario: str = "default"


class CallerMessage(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message is empty")
        return value.strip()


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(status_code=status, detail={"code": code, "message": message})


@app.exception_handler(ApiError)
def _api_error(_: Any, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def _default_brain_kind() -> str:
    return "stub" if settings.provider == "stub" else "claude"


@contextmanager
def _brain_for(kind: str, api_key: str | None) -> Iterator[Brain]:
    """The Brain for one request. A key sent by the browser lives exactly as long as that request:
    its client is closed afterwards and nothing keeps a reference to it."""
    if kind == "stub":
        yield _stub
        return
    key = (api_key or "").strip() or None
    if key is None and not settings.server_key_configured:
        raise ApiError(
            400,
            "missing_api_key",
            "No API key is configured. Set ANTHROPIC_API_KEY on the server, or enter a key in Settings.",
        )
    from .brain.claude import ClaudeBrain, make_client  # imported lazily so the stub runs without the SDK

    if key is None:
        # One shared client for the server's own key, so its connection pool is reused.
        global _server_brain
        if _server_brain is None:
            _server_brain = ClaudeBrain(settings, limits=harness.limits)
        yield _server_brain
        return
    with make_client(key) as client:
        yield ClaudeBrain(settings, client=client, limits=harness.limits)


def _get_session(session_id: str) -> tuple[Session, str]:
    with _sessions_lock:
        entry = _sessions.get(session_id)
    if entry is None:
        raise ApiError(404, "unknown_session", "This conversation no longer exists. Start a new one.")
    return entry


@app.get("/")
def index() -> HTMLResponse:
    """The console page, with its stylesheet and script addressed by version so that a browser holding
    an older copy of either fetches the current one."""
    version = max(int(path.stat().st_mtime) for path in STATIC_DIR.iterdir())
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset in ("/static/styles.css", "/static/app.js"):
        html = html.replace(f'{asset}"', f'{asset}?v={version}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "company": settings.company_name,
        "default_brain": _default_brain_kind(),
        "server_key_configured": settings.server_key_configured,
        "model": settings.model,
        "nlu_model": settings.nlu_model,
        "effort": settings.effort,
        "demo_today": settings.today.isoformat(),
        "consent_scenarios": repo.consent_scenarios(),
        "sop": {
            "phases": [
                {
                    "phase": spec.phase.value,
                    "title": spec.title,
                    "autonomy": spec.autonomy.value,
                    "purpose": spec.purpose,
                    "tools": list(spec.tools),
                    "data_visible_to_model": spec.data_visible_to_model,
                    "exits": {target.value: rule for target, rule in ALLOWED_TRANSITIONS[spec.phase].items()},
                }
                for spec in PHASES.values()
            ],
            "identity_fields": IDENTITY_FIELDS,
            "intents": {name: spec.label for name, spec in INTENTS.items()},
            "limits": asdict(harness.limits),
        },
    }


@app.get("/api/fixtures")
def fixtures() -> dict[str, Any]:
    """The sample records, so a tester knows what to type. A demo convenience only: it is exactly the
    data the agent must never reveal, so it is off unless EXPOSE_FIXTURES is left at its default."""
    if os.environ.get("EXPOSE_FIXTURES", "1") != "1":
        raise ApiError(404, "disabled", "Fixture listing is disabled.")
    return {
        "policyholders": repo.policyholders(),
        "claims": [c for p in repo.policyholders() for c in repo.claims_for(p["party_id"])],
        "representatives": repo.representatives(),
    }


@app.post("/api/sessions")
def create_session(body: NewSession) -> dict[str, Any]:
    kind = body.brain or _default_brain_kind()
    scenario = body.consent_scenario if body.consent_scenario in repo.consent_scenarios() else "default"
    session = harness.new_session(consent_scenario=scenario)
    with _sessions_lock:
        _sessions[session.id] = (session, kind)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    return {"session_id": session.id, "brain": kind, "reply": session.transcript[0].text, "state": harness.snapshot(session)}


@app.post("/api/sessions/{session_id}/messages")
def post_message(
    session_id: str,
    body: CallerMessage,
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    session, kind = _get_session(session_id)
    with _brain_for(kind, x_api_key) as brain, session.lock:
        try:
            reply = harness.handle(session, body.text, brain)
        except BrainError as err:
            raise ApiError(401 if err.user_fixable else 502, "model_setup", str(err)) from err
        return {"reply": reply, "brain": kind, "state": harness.snapshot(session)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session, kind = _get_session(session_id)
    return {
        "brain": kind,
        "transcript": [{"role": t.role, "text": t.text, "phase": t.phase} for t in session.transcript],
        "state": harness.snapshot(session),
    }

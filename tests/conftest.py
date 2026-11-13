"""Shared fixtures. Nothing here calls a model: the harness runs against the offline stub or
against a scripted Brain that says exactly what a test tells it to."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Callable

import pytest

from app.brain.base import EmailContext, EmailDraft, RespondContext, RespondResult, TurnUnderstanding, UnderstandContext
from app.brain.stub import StubBrain
from app.config import load_settings
from app.data import Repository
from app.engine import Harness
from app.state import Session
from app.tools import ToolGateway

DEMO_MESSAGE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
    "claim from January. DOB is 1985-03-15, SSN last four is 4472."
)
VERIFY_MARGARET = "My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472."


@pytest.fixture(scope="session")
def settings():
    # Pinned, so the suite does not depend on whatever the environment or a local .env sets.
    return replace(
        load_settings(),
        provider="anthropic",
        model="claude-opus-5-5",
        nlu_model="claude-haiku-5-5",
        effort="low",
        fallbacks_enabled=True,
        today=date(2026, 3, 2),
        company_name="Acme Insurance",
    )


@pytest.fixture(scope="session")
def repo(settings):
    return Repository(settings.fixtures_dir)


@pytest.fixture()
def harness(repo, settings):
    return Harness(repo, settings)


@pytest.fixture()
def stub():
    return StubBrain()


class ScriptedBrain:
    """A Brain under the test's control. `understandings` are returned in order (then blank ones);
    `respond` decides the reply and may call tools, including ones it should not have."""

    name = "scripted"

    def __init__(
        self,
        understandings: list[TurnUnderstanding] | None = None,
        respond: Callable[[RespondContext, ToolGateway], RespondResult] | None = None,
    ) -> None:
        self._understandings = list(understandings or [])
        self._respond = respond
        self.respond_calls: list[RespondContext] = []
        self.tool_outcomes: list[tuple[str, bool]] = []

    def understand(self, ctx: UnderstandContext) -> TurnUnderstanding:
        return self._understandings.pop(0) if self._understandings else TurnUnderstanding.blank()

    def respond(self, ctx: RespondContext, tools: ToolGateway) -> RespondResult:
        self.respond_calls.append(ctx)
        if self._respond is None:
            return RespondResult(text=ctx.directive.script)
        return self._respond(ctx, tools)

    def draft_email(self, ctx: EmailContext) -> EmailDraft:
        return EmailDraft(subject="Summary", body=ctx.template)


def talk(harness: Harness, brain, *messages: str, session: Session | None = None, scenario: str = "default"):
    """Send messages in order; return (session, replies)."""
    session = session or harness.new_session(scenario)
    replies = [harness.handle(session, message, brain) for message in messages]
    return session, replies


def identity_understanding(**fields: str) -> TurnUnderstanding:
    base = TurnUnderstanding.blank()
    for key, value in fields.items():
        setattr(base.identity, key, value)
    return base


def margaret_verified() -> TurnUnderstanding:
    return identity_understanding(full_name="Margaret Chen", date_of_birth="1985-03-15", id_last4="4472")

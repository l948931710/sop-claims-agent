"""The Claude brain against a fake API client.

These tests check what the harness sends to the model and how it handles what
comes back. They do not check the model's behaviour; that needs a real key.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from app.brain import claude as claude_module
from app.brain.base import BrainError, TurnUnderstanding
from app.brain.claude import FALLBACK_BETA, ClaudeBrain
from app.sop import Phase

from .conftest import identity_understanding, margaret_verified, talk


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, arguments: dict[str, Any], call_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=call_id)


def message(*blocks: SimpleNamespace, stop_reason: str = "end_turn", parsed: Any = None) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, parsed_output=parsed)


class FakeEndpoint:
    def __init__(self, client: "FakeClient", name: str) -> None:
        self._client, self._name = client, name

    def create(self, **kwargs: Any) -> Any:
        return self._client.handle(self._name, "create", kwargs)

    def parse(self, **kwargs: Any) -> Any:
        return self._client.handle(self._name, "parse", kwargs)


class FakeClient:
    """Returns queued responses in order and records every request."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = FakeEndpoint(self, "stable")
        self.beta = SimpleNamespace(messages=FakeEndpoint(self, "beta"))

    def handle(self, endpoint: str, method: str, kwargs: dict[str, Any]) -> Any:
        self.requests.append({"endpoint": endpoint, "method": method, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def api_error(cls: type, status: int, text_: str) -> Exception:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(text_, response=httpx2.Response(status, request=request), body=None)


def request_text(request: dict[str, Any]) -> str:
    """Everything a request shows the model, as one string."""
    return json.dumps({"system": request.get("system"), "messages": request.get("messages")}, default=str)


@pytest.fixture(autouse=True)
def _reset_fallback_flag():
    claude_module._fallbacks_rejected.clear()
    yield
    claude_module._fallbacks_rejected.clear()


def test_demo_turn_makes_three_narrow_calls(harness, settings):
    reading = margaret_verified()
    reading.intent = "denial_question"
    reading.case_hints.description = "denied healthcare claim from January"
    reading.case_hints.status = "denied"
    client = FakeClient(
        [
            message(parsed=reading),
            message(tool_use("select_case", {"case_id": "CL-2048", "intent": "denial_question", "reason": "the only denied healthcare claim"}), stop_reason="tool_use"),
            message(text("Thanks Margaret, you're verified. Claim CL-2048 from January 12, 2026 was denied because the pathology report and office note were missing.")),
        ]
    )
    session, (reply,) = talk(harness, ClaudeBrain(settings, client=client), "demo message")

    assert session.phase is Phase.PROCESS_CASE and reply.startswith("Thanks Margaret")
    reader, resolver, writer = client.requests
    assert reader["method"] == "parse" and reader["output_format"] is TurnUnderstanding
    assert "tools" not in reader
    assert [t["name"] for t in resolver["tools"]] == ["select_case", "request_human_handoff"]
    assert [t["name"] for t in writer["tools"]] == ["get_document_guidance", "get_followup_guidance", "switch_case", "request_human_handoff"]
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in writer["tools"])
    # The claim record reaches the model only once the case is selected.
    assert "pathology report" not in request_text(reader)
    assert "pathology report" in request_text(writer)


def test_reader_and_writer_use_their_own_models(harness, settings):
    client = FakeClient([message(parsed=TurnUnderstanding.blank()), message(text("Could you share three identifying details?"))])
    talk(harness, ClaudeBrain(settings, client=client), "hello")
    reader, writer = client.requests

    # The reader is the small model. It has no server-side refusal fallback, so none is requested.
    assert reader["model"] == "claude-haiku-5-5" and reader["endpoint"] == "stable"
    assert "fallbacks" not in reader and "betas" not in reader
    # The writer opts into the fallback.
    assert writer["model"] == "claude-opus-5-5" and writer["endpoint"] == "beta"
    assert writer["betas"] == [FALLBACK_BETA] and writer["fallbacks"] == "default"
    for request in client.requests:
        assert request["output_config"] == {"effort": "low"}
        assert "thinking" not in request and "temperature" not in request


def test_nothing_from_the_records_is_sent_before_verification(harness, settings, repo):
    client = FakeClient(
        [message(parsed=identity_understanding(full_name="Margaret Chen", date_of_birth="1985-03-15")), message(text("Thanks. One more detail, please."))]
    )
    session, _ = talk(harness, ClaudeBrain(settings, client=client), "I'm Margaret Chen, born 1985-03-15. Why was my claim denied?")
    assert session.phase is Phase.VERIFY_ID

    record = repo.policyholder("P9")
    sent = " ".join(request_text(r) for r in client.requests)
    for secret in (record["phone"], record["email"], record["id_last4"], "6505212836", "CL-2048", "CL-2102", "pathology", "1450"):
        assert secret not in sent
    writer = client.requests[1]
    assert "tools" not in writer
    assert "Do not confirm or deny that any claim exists" in request_text(writer)
    assert writer["messages"][0] == {"role": "user", "content": "(The caller has connected.)"}
    assert writer["messages"][-1]["role"] == "user"


def test_tool_results_are_fed_back_until_the_model_answers(harness, settings):
    selecting = margaret_verified()
    client = FakeClient(
        [
            message(parsed=selecting),
            message(tool_use("select_case", {"case_id": "CL-2048", "intent": "document_submission", "reason": "asked"}), stop_reason="tool_use"),
            message(text("I have claim CL-2048 open. What would you like to know?")),
            message(parsed=TurnUnderstanding.blank(intent="document_submission", has_open_request=True)),
            message(text("Let me check."), tool_use("get_followup_guidance", {"topic": "submission_method"}, "toolu_9"), stop_reason="tool_use"),
            message(text("The member portal or the claim upload link is the best place to start.")),
        ]
    )
    brain = ClaudeBrain(settings, client=client)
    session, _ = talk(harness, brain, "verify me")
    _, (reply,) = talk(harness, brain, "How do I submit the documents?", session=session)

    assert reply.startswith("The member portal")
    follow_up = client.requests[-1]["messages"]
    assert follow_up[-2]["role"] == "assistant"  # the model's own turn goes back unchanged
    result = follow_up[-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_9" and not result["is_error"]
    assert "member portal" in result["content"]


def test_refused_tool_call_is_reported_back_as_an_error(harness, settings):
    client = FakeClient(
        [
            message(parsed=margaret_verified()),
            message(tool_use("select_case", {"case_id": "CL-3001", "intent": "denial_question", "reason": "guess"}), stop_reason="tool_use"),
            message(text("Which of your claims are you calling about?")),
        ]
    )
    session, _ = talk(harness, ClaudeBrain(settings, client=client), "verify me")
    assert session.phase is Phase.RESOLVE_INTENT
    result = client.requests[-1]["messages"][-1]["content"][0]
    assert result["is_error"] and "not a claim on this policy" in result["content"]


def test_model_refusal_falls_back_to_the_script(harness, settings):
    client = FakeClient([message(parsed=identity_understanding(full_name="Margaret Chen")), message(stop_reason="refusal")])
    session, (reply,) = talk(harness, ClaudeBrain(settings, client=client), "I'm Margaret Chen")
    assert "confirmed your full name" in reply
    assert any(e.kind == "error" for e in session.traces[-1].events)


def test_fallback_parameter_is_dropped_if_the_api_rejects_it(harness, settings):
    client = FakeClient(
        [
            message(parsed=TurnUnderstanding.blank()),
            api_error(anthropic.BadRequestError, 400, "fallbacks: this beta is not available"),
            message(text("Could you share three identifying details?")),
            message(parsed=TurnUnderstanding.blank()),
            message(text("Any three of the five will do.")),
        ]
    )
    brain = ClaudeBrain(settings, client=client)
    session, _ = talk(harness, brain, "hello")
    talk(harness, brain, "hello again", session=session)
    # reader, writer (rejected), writer retried without it, then a second turn that no longer tries it
    assert [r["endpoint"] for r in client.requests] == ["stable", "beta", "stable", "stable", "stable"]
    assert all("fallbacks" not in r for r in client.requests[2:])


def test_one_model_for_both_roles_is_still_supported(harness, settings):
    client = FakeClient([message(parsed=TurnUnderstanding.blank()), message(text("Could you share three identifying details?"))])
    brain = ClaudeBrain(replace(settings, nlu_model="claude-opus-5-5"), client=client)
    talk(harness, brain, "hello")
    reader, writer = client.requests
    assert reader["model"] == writer["model"] == "claude-opus-5-5"
    assert reader["endpoint"] == writer["endpoint"] == "beta"


def test_rejected_key_is_reported_as_a_setup_problem(harness, settings):
    client = FakeClient([api_error(anthropic.AuthenticationError, 401, "invalid x-api-key")])
    session = harness.new_session()
    with pytest.raises(BrainError) as raised:
        harness.handle(session, "hello", ClaudeBrain(settings, client=client))
    assert raised.value.user_fixable and session.turn == 0

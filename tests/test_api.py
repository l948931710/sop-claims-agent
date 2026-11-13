"""The HTTP layer the test console talks to."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main

from .conftest import DEMO_MESSAGE


@pytest.fixture()
def client():
    return TestClient(main.app)


def test_console_and_config_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "/static/styles.css?v=" in page.text and "/static/app.js?v=" in page.text
    for asset in ("/", "/static/app.js", "/static/styles.css"):
        response = client.get(asset)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-cache"
    assert client.get("/healthz").json() == {"status": "ok"}
    config = client.get("/api/config").json()
    assert [p["phase"] for p in config["sop"]["phases"]][:4] == ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"]
    assert config["sop"]["limits"]["min_identity_matches"] == 3


def test_a_conversation_over_http(client):
    created = client.post("/api/sessions", json={"brain": "stub"}).json()
    assert created["state"]["phase"] == "VERIFY_ID" and "verify your identity" in created["reply"]

    response = client.post(f"/api/sessions/{created['session_id']}/messages", json={"text": DEMO_MESSAGE})
    assert response.status_code == 200
    body = response.json()
    assert body["state"]["phase"] == "PROCESS_CASE" and body["state"]["verified"]
    assert body["state"]["selected_case_id"] == "CL-2048"
    assert any(slot["key"] == "case.description" and slot["phase"] == "VERIFY_ID" for slot in body["state"]["memory"]["slots"])

    stored = client.get(f"/api/sessions/{created['session_id']}").json()
    assert [t["role"] for t in stored["transcript"]] == ["assistant", "user", "assistant"]


def test_state_snapshot_never_carries_values_on_file(client):
    created = client.post("/api/sessions", json={"brain": "stub"}).json()
    body = client.post(f"/api/sessions/{created['session_id']}/messages", json={"text": "My name is Margaret Chen"}).json()
    snapshot = str(body["state"])
    for secret in ("+16505212836", "6505212836", "margaret@email.com", "4472", "1985-03-15"):
        assert secret not in snapshot


def test_claude_without_a_key_is_a_clear_error(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    created = client.post("/api/sessions", json={"brain": "claude"}).json()
    response = client.post(f"/api/sessions/{created['session_id']}/messages", json={"text": "hello"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_api_key"


def test_unknown_session_and_empty_message_are_rejected(client):
    assert client.post("/api/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    created = client.post("/api/sessions", json={"brain": "stub"}).json()
    assert client.post(f"/api/sessions/{created['session_id']}/messages", json={"text": ""}).status_code == 422
    assert client.post(f"/api/sessions/{created['session_id']}/messages", json={"text": "   "}).status_code == 422

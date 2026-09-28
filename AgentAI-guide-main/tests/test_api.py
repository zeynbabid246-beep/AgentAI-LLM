"""API tests: /health, /chat/stream (SSE shape), /agent-chat, validation, rate limit.

The agent is stubbed so no network or GOOGLE_API_KEY is needed.
"""

import json
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

import main
from main import SlidingWindowLimiter


class FakeSettings:
    """Configured-looking settings so endpoints pass the config gate without a key."""

    is_configured = True
    cors_origins = ["*"]
    rate_limit_per_minute = 20
    max_message_chars = 4000
    model_name = "gemini-2.5-flash"
    agent_max_iterations = 8
    provider_id = "gemini"
    provider_label = "Google Gemini (AI Studio)"
    resolved_model = "gemini-2.5-flash"
    missing_key_env = "GOOGLE_API_KEY"


@pytest.fixture(autouse=True)
def configured_settings(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: FakeSettings())


@pytest.fixture()
def client() -> TestClient:
    return TestClient(main.app)


def _parse_sse(text: str) -> List[Dict]:
    events = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_with_status_field(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-2.5-flash"
        if data["status"] == "degraded":
            assert data["model"] is None

    def test_root_serves_chat_ui(self, client):
        """The FastAPI app serves the frontend at / (same-origin UI)."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "chatMessages" in resp.text  # chat UI markup present

    def test_static_assets_served(self, client):
        assert client.get("/script.js").status_code == 200
        assert client.get("/style.css").status_code == 200


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE contract
# ---------------------------------------------------------------------------

class TestChatStream:
    def test_streams_tool_token_and_final_events(self, client, monkeypatch):
        async def fake_stream(message, session_id):
            assert message == "What is 2+2?"
            assert session_id == "sess-1"
            yield {"type": "tool_start", "tool": "calculator", "label": "Calculating", "args": {"expression": "2+2"}}
            yield {"type": "tool_end", "tool": "calculator", "label": "Calculating", "summary": "…"}
            yield {"type": "token", "text": "It is "}
            yield {"type": "token", "text": "4."}
            yield {"type": "final", "text": "It is 4."}

        monkeypatch.setattr(main, "stream_agent_events", fake_stream)
        resp = client.post("/chat/stream", json={"message": "What is 2+2?", "session_id": "sess-1"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert [e["type"] for e in events] == ["tool_start", "tool_end", "token", "token", "final"]
        assert events[-1]["text"] == "It is 4."
        assert events[0]["tool"] == "calculator"

    def test_error_inside_stream_is_emitted_as_error_event(self, client, monkeypatch):
        async def failing_stream(message, session_id):
            yield {"type": "tool_start", "tool": "calculator", "label": "Calculating", "args": {}}
            raise RuntimeError("boom")

        monkeypatch.setattr(main, "stream_agent_events", failing_stream)
        resp = client.post("/chat/stream", json={"message": "hi", "session_id": "sess-err"})
        assert resp.status_code == 200  # stream still opens...
        events = _parse_sse(resp.text)
        assert events[-1]["type"] == "error"  # ...and reports the failure as an event

    def test_503_when_not_configured(self, client, monkeypatch):
        class UnconfiguredSettings(FakeSettings):
            is_configured = False

        monkeypatch.setattr(main, "get_settings", lambda: UnconfiguredSettings())
        resp = client.post("/chat/stream", json={"message": "hi", "session_id": "s"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /agent-chat — legacy JSON contract
# ---------------------------------------------------------------------------

class TestAgentChat:
    def test_returns_json_response(self, client, monkeypatch):
        async def fake_run(message, session_id):
            return "the answer"

        monkeypatch.setattr(main, "run_agent", fake_run)
        resp = client.post("/agent-chat", json={"message": "hi", "session_id": "legacy"})
        assert resp.status_code == 200
        assert resp.json() == {"response": "the answer", "session_id": "legacy"}

    def test_old_payload_without_session_id_uses_default(self, client, monkeypatch):
        async def fake_run(message, session_id):
            return "ok"

        monkeypatch.setattr(main, "run_agent", fake_run)
        resp = client.post("/agent-chat", json={"message": "hi"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "default"


# ---------------------------------------------------------------------------
# Validation & rate limiting
# ---------------------------------------------------------------------------

class TestValidationAndLimits:
    def test_empty_message_rejected(self, client):
        resp = client.post("/agent-chat", json={"message": ""})
        assert resp.status_code == 422

    def test_message_over_cap_rejected(self, client):
        resp = client.post("/agent-chat", json={"message": "x" * 5000})
        assert resp.status_code == 422

    def test_bad_session_id_rejected(self, client):
        resp = client.post("/agent-chat", json={"message": "hi", "session_id": "../evil"})
        assert resp.status_code == 422

    def test_rate_limit_returns_429(self, client, monkeypatch):
        async def fake_run(message, session_id):
            return "ok"

        monkeypatch.setattr(main, "run_agent", fake_run)
        monkeypatch.setattr(main, "limiter", SlidingWindowLimiter(max_requests=2, window_seconds=60))

        assert client.post("/agent-chat", json={"message": "1", "session_id": "rl"}).status_code == 200
        assert client.post("/agent-chat", json={"message": "2", "session_id": "rl"}).status_code == 200
        resp = client.post("/agent-chat", json={"message": "3", "session_id": "rl"})
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["detail"]

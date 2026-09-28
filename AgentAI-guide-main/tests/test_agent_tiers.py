"""Tests for the tiered agent: mode selection + text-protocol fallback loop.

No network and no GOOGLE_API_KEY: the fallback loop is driven by a fake chat
model that returns scripted replies.
"""

import asyncio
from typing import Any, Dict, List

import pytest

import agent as agent_mod
from agent import AgentNotConfigured, reset_agent
from config import PROVIDERS, Settings


class FakeLLM:
    """Returns scripted replies in order; records every prompt it sees."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[List[Any]] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return _ai(self.replies.pop(0))

    async def astream(self, messages):
        self.calls.append(list(messages))
        text = self.replies.pop(0)
        for i in range(0, len(text), 6):
            yield _ai(text[i : i + 6])


def _ai(text: str):
    from langchain_core.messages import AIMessage

    return AIMessage(content=text)


@pytest.fixture(autouse=True)
def clean_agent_state():
    reset_agent()
    yield
    reset_agent()


def _mode(monkeypatch, llm, mode="openai_compat"):
    monkeypatch.setattr(agent_mod, "_agent", llm)
    monkeypatch.setattr(agent_mod, "_mode", mode)


async def collect(message: str, session_id: str) -> List[Dict[str, Any]]:
    return [e async for e in agent_mod.stream_agent_events(message, session_id)]


def _without_status(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip keep-alive/status events; they carry no protocol meaning."""
    return [e for e in events if e["type"] != "status"]


class TestTextProtocolLoop:
    def test_direct_answer_without_tools(self, monkeypatch):
        _mode(monkeypatch, FakeLLM(["FINAL: Hello! How can I help?"]))
        events = asyncio.run(collect("hi", "s1"))

        assert events[0]["type"] == "status"  # immediate progress event
        core = _without_status(events)
        assert [e["type"] for e in core] == ["final"]
        assert core[0]["text"] == "Hello! How can I help?"

    def test_tool_call_then_final(self, monkeypatch):
        llm = FakeLLM(
            [
                "ACTION: calculator\nINPUT: 2+2",
                "FINAL: The answer is 4.",
            ]
        )
        _mode(monkeypatch, llm)
        events = asyncio.run(collect("What is 2+2?", "s2"))

        core = _without_status(events)
        assert [e["type"] for e in core] == ["tool_start", "tool_end", "final"]
        assert core[0]["tool"] == "calculator"
        assert core[0]["label"] == "Calculating"
        assert core[0]["args"] == {"input": "2+2"}
        assert core[-1]["text"] == "The answer is 4."

        # The observation must be fed back to the model before the final reply.
        second_call = llm.calls[1]
        assert any("OBSERVATION" in str(m.content) for m in second_call)

    def test_bare_reply_is_treated_as_final(self, monkeypatch):
        """Model forgets the FINAL: prefix — don't crash, don't loop."""
        _mode(monkeypatch, FakeLLM(["Just a plain answer."]))
        events = asyncio.run(collect("hi", "s3"))
        assert events[-1]["type"] == "final"
        assert events[-1]["text"] == "Just a plain answer."

    def test_unknown_tool_reports_error_to_model(self, monkeypatch):
        # (status events filtered below)
        llm = FakeLLM(
            [
                "ACTION: nonexistent_tool\nINPUT: x",
                "FINAL: Understood, let me answer directly: 42.",
            ]
        )
        _mode(monkeypatch, llm)
        events = _without_status(asyncio.run(collect("q", "s4")))

        assert events[0]["type"] == "tool_start"
        assert "OBSERVATION" in str(llm.calls[1][-1].content)
        assert "Unknown tool" in str(llm.calls[1][-1].content)
        assert events[-1]["text"].startswith("Understood")

    def test_history_grows_across_turns(self, monkeypatch):
        llm = FakeLLM(["FINAL: first"], )
        _mode(monkeypatch, llm)
        asyncio.run(collect("turn one", "mem"))
        assert len(llm.calls[0]) == 2  # system + human

        llm.replies = ["FINAL: second"]
        asyncio.run(collect("turn two", "mem"))
        # system + (human, ai) from turn 1 + human = 4 messages
        assert len(llm.calls[1]) == 4

    def test_llm_error_becomes_error_event(self, monkeypatch):
        class ExplodingLLM:
            async def astream(self, messages):
                raise RuntimeError("boom")
                yield  # pragma: no cover — makes this an async generator

        _mode(monkeypatch, ExplodingLLM())
        events = asyncio.run(collect("hi", "s-err"))
        assert events[-1]["type"] == "error"
        assert "boom" in events[-1]["text"]


class TestParseAction:
    def test_parses_standard_format(self):
        assert agent_mod._parse_action("ACTION: calculator\nINPUT: 2+2") == ("calculator", "2+2")

    def test_parses_case_insensitive_with_surrounding_text(self):
        text = "Let me think...\naction: duckduckgo_search\ninput: latest AI news"
        assert agent_mod._parse_action(text) == ("duckduckgo_search", "latest AI news")

    def test_returns_none_for_final(self):
        assert agent_mod._parse_action("FINAL: done") is None
        assert agent_mod._parse_action("") is None

    def test_input_takes_first_line_only(self):
        result = agent_mod._parse_action("ACTION: wikipedia_lookup\nINPUT: Albert Einstein\nextra junk")
        assert result == ("wikipedia_lookup", "Albert Einstein")


class TestModeSelection:
    def test_get_agent_raises_without_key(self, monkeypatch):
        class NoKey:
            is_configured = False

        monkeypatch.setattr(agent_mod, "get_settings", lambda: NoKey())
        with pytest.raises(AgentNotConfigured):
            agent_mod.get_agent()

    def test_mode_is_one_of_known_tiers(self, monkeypatch):
        """With a fake key, whichever tier builds must be a known one."""
        class FakeKey:
            is_configured = True
            google_api_key = "fake-key"
            model_name = "gemini-3.8-flash"
            model_temperature = 0.0
            agent_max_iterations = 8
            gemini_openai_base_url = "https://example.invalid/v1beta/"
            provider_id = "gemini"
            provider_label = "Google Gemini (AI Studio)"
            provider_api_key = "fake-key"
            provider_base_url = "https://example.invalid/v1beta/"
            resolved_model = "gemini-3.8-flash"
            missing_key_env = "GOOGLE_API_KEY"
            model_reasoning_effort = "low"

        monkeypatch.setattr(agent_mod, "get_settings", lambda: FakeKey())
        try:
            _runner, mode = agent_mod.get_agent()
            assert mode in ("langgraph", "openai_compat")
        except AgentNotConfigured:
            pytest.skip("No agent stack installed in this environment")


def _settings(**kwargs) -> Settings:
    """Real Settings object, .env ignored so tests are hermetic.

    Uses env-var (alias) names — LLM_PROVIDER, GROQ_API_KEY, … — matching
    how values arrive in production from Backend/.env.
    """
    return Settings(_env_file=None, **kwargs)


class TestProviderConfig:
    def test_registry_has_all_providers(self):
        assert set(PROVIDERS) == {"gemini", "groq", "ollama", "openrouter", "cerebras"}

    def test_default_provider_is_gemini(self):
        s = _settings()
        assert s.provider_id == "gemini"
        assert s.resolved_model == "gemini-3.8-flash"
        assert s.missing_key_env == "GOOGLE_API_KEY"

    def test_groq_selection(self):
        s = _settings(LLM_PROVIDER="groq", GROQ_API_KEY="gsk_test")
        assert s.provider_id == "groq"
        assert s.provider_base_url == "https://api.groq.com/openai/v1"
        assert s.provider_api_key == "gsk_test"
        assert s.resolved_model == "qwen/qwen3.8-27b"
        assert s.is_configured
        assert s.missing_key_env == "GROQ_API_KEY"

    def test_openrouter_selection(self):
        s = _settings(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="sk-or-test")
        assert s.provider_id == "openrouter"
        assert s.provider_base_url == "https://openrouter.ai/api/v1"
        assert s.resolved_model == "meta-llama/llama-3.3-70b-instruct"
        assert s.missing_key_env == "OPENROUTER_API_KEY"

    def test_cerebras_selection(self):
        s = _settings(LLM_PROVIDER="cerebras", CEREBRAS_API_KEY="csk-test")
        assert s.provider_id == "cerebras"
        assert s.provider_base_url == "https://api.cerebras.ai/v1"
        assert s.resolved_model == "llama-3.3-70b"
        assert s.missing_key_env == "CEREBRAS_API_KEY"

    def test_ollama_needs_no_key(self):
        s = _settings(LLM_PROVIDER="ollama")
        assert s.is_configured  # keyless provider is always "configured"
        assert s.provider_api_key  # SDK placeholder, non-empty
        assert s.provider_base_url == "http://localhost:11434/v1"
        assert s.resolved_model == "llama3.1"
        assert s.missing_key_env == ""

    def test_unknown_provider_falls_back_to_gemini(self):
        s = _settings(LLM_PROVIDER="nonsense", GOOGLE_API_KEY="k")
        assert s.provider_id == "gemini"

    def test_explicit_model_overrides_provider_default(self):
        s = _settings(LLM_PROVIDER="groq", GROQ_API_KEY="k", MODEL_NAME="llama3/groq-tool-use")
        assert s.resolved_model == "llama3/groq-tool-use"


class TestProviderAwareBuilder:
    """The tier-2 builder must target the selected provider's endpoint."""

    def _use_settings(self, monkeypatch, **kwargs) -> Settings:
        s = _settings(**kwargs)
        monkeypatch.setattr(agent_mod, "get_settings", lambda: s)
        return s

    def test_groq_builds_openai_compat_targeting_groq(self, monkeypatch):
        self._use_settings(monkeypatch, LLM_PROVIDER="groq", GROQ_API_KEY="gsk_test")
        try:
            llm, mode = agent_mod.get_agent()
        except AgentNotConfigured:
            pytest.skip("langchain_openai not installed in this environment")
        assert mode == "openai_compat"  # non-Gemini never uses the langgraph tier
        assert llm.model_name == "qwen/qwen3.8-27b"
        assert "api.groq.com" in str(getattr(llm, "openai_api_base", ""))
        assert "reasoning_effort" not in (llm.model_kwargs or {})

    def test_gemini_keeps_reasoning_effort_kwarg(self, monkeypatch):
        self._use_settings(monkeypatch, LLM_PROVIDER="gemini", GOOGLE_API_KEY="fake")
        try:
            llm, mode = agent_mod.get_agent()
        except AgentNotConfigured:
            pytest.skip("langchain_openai not installed in this environment")
        if mode != "openai_compat":
            pytest.skip("langgraph tier installed — tier-2 builder not exercised")
        assert (llm.model_kwargs or {}).get("reasoning_effort") == "low"
        assert "generativelanguage.googleapis.com" in str(getattr(llm, "openai_api_base", ""))

    def test_not_configured_error_names_provider_env(self, monkeypatch):
        self._use_settings(monkeypatch, LLM_PROVIDER="groq", GROQ_API_KEY="")
        with pytest.raises(AgentNotConfigured) as excinfo:
            agent_mod.get_agent()
        assert "GROQ_API_KEY" in str(excinfo.value)

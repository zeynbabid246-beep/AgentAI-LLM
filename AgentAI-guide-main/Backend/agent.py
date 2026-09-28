"""Agent construction and streaming, with two interchangeable execution tiers.

Tier 1 (preferred): LangGraph prebuilt ReAct agent + ChatGoogleGenerativeAI.
  Native Gemini tool calling, InMemorySaver checkpointer for session memory.
  Gemini-only: for every other provider we force tier 2 up front, since all
  of them are plain OpenAI-compatible chat endpoints.

Tier 2 (fallback): a lightweight text-protocol tool loop over ChatOpenAI
  pointed at whichever OpenAI-compatible provider is configured (Gemini AI
  Studio, Groq, Ollama, OpenRouter, Cerebras — see config.PROVIDERS), used
  automatically when langgraph / langchain-google-genai are not installed
  (e.g. restricted package mirrors). Plain-text protocol instead of OpenAI
  tool calls so it works identically on every provider. Memory: hand-rolled
  per-session bounded deque.

Both tiers emit the same normalized events consumed by the SSE endpoint:
  {"type": "status",     "text"}                      # progress line for the UI
  {"type": "tool_start", "tool", "label", "args"}
  {"type": "tool_end",   "tool", "label", "summary"}
  {"type": "token",      "text"}
  {"type": "final",      "text"}
"""

import asyncio
import functools
import logging
import re
import time
from collections import deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, AIMessageChunk

from config import get_settings
from tools import get_tools

# Heavy model/runtime deps are imported lazily inside the builders so the
# API (and the test suite) can start — and /health can answer — without the
# Gemini/LangGraph stack installed. Missing packages surface as a clear
# AgentNotConfigured error on the first chat request instead of a crash.

logger = logging.getLogger(__name__)

# Friendly labels for tool-step chips in the UI
TOOL_LABELS: Dict[str, str] = {
    "duckduckgo_search": "Searching the web",
    "wikipedia_lookup": "Looking up Wikipedia",
    "calculator": "Calculating",
}

# Shared state for whichever tier is active.
_mode: Optional[str] = None  # "langgraph" | "openai_compat"
_agent: Any = None
_agent_error: Optional[str] = None
_checkpointer: Any = None  # LangGraph InMemorySaver (tier 1)
_histories: Dict[str, Deque] = {}  # session_id -> deque of messages (tier 2)


class AgentNotConfigured(RuntimeError):
    """Raised when the agent is used without its provider's API key (or deps)."""


def _build_langgraph_agent() -> Any:
    global _checkpointer
    if get_settings().provider_id != "gemini":
        # LangGraph tier is wired to ChatGoogleGenerativeAI; every other
        # provider goes straight to the OpenAI-compatible tier.
        raise ImportError("langgraph tier supports only the gemini provider")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        temperature=settings.model_temperature,
        api_key=settings.google_api_key,
        convert_system_message_to_human=True,  # needed for Gemini 1.x-era models; harmless otherwise
    )
    _checkpointer = InMemorySaver()
    return create_react_agent(llm, get_tools(), checkpointer=_checkpointer)


_TEXTLOOP_SYSTEM_PROMPT = """You are a helpful assistant with tools. Answer in the user's language.

You may use these tools:
{tool_catalog}

To use a tool, reply with EXACTLY this two-line format (nothing else):
ACTION: <tool_name>
INPUT: <one-line input for the tool>

After you receive an OBSERVATION with the tool result, either call another
tool the same way, or give your final reply starting with:
FINAL: <your complete answer to the user>

If no tool is needed, reply directly with FINAL: <answer>.
Always answer factually; never invent tool results."""


def _build_openai_compat_agent() -> Any:
    """Fallback: plain chat model on the configured OpenAI-compatible endpoint.

    Works for every provider in config.PROVIDERS (Gemini AI Studio, Groq,
    Ollama, OpenRouter, Cerebras). The tool loop lives in
    _stream_openai_compat and speaks a text protocol, which avoids Gemini 3's
    mandatory thought_signature requirement on the OpenAI tool-call protocol
    and keeps behavior identical across providers.
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    model_kwargs: Dict[str, Any] = {}
    if settings.provider_id == "gemini":
        # Gemini 3 thinks by default (seconds per call). "low" keeps tool
        # use reliable while cutting most of the thinking latency. Not sent
        # to other providers — they reject unknown kwargs with 422.
        model_kwargs["reasoning_effort"] = settings.model_reasoning_effort
    return ChatOpenAI(
        model=settings.resolved_model,
        temperature=settings.model_temperature,
        api_key=settings.provider_api_key,
        base_url=settings.provider_base_url,
        max_retries=2,
        model_kwargs=model_kwargs,
    )


def _tool_catalog() -> str:
    return "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in get_tools())


def _parse_action(text: str) -> Optional[Tuple[str, str]]:
    """Extract an ACTION/INPUT pair from a model reply; None if it's a final answer."""
    cleaned = text.strip()
    if cleaned.startswith("`"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
    match = re.search(r"ACTION:\s*([\w\-]+)\s*[\r\n]+\s*INPUT:\s*(.+)", cleaned, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip().lower(), match.group(2).strip().splitlines()[0].strip()


def _invoke_tool_by_name(name: str, tool_input: str) -> str:
    for t in get_tools():
        if t.name == name:
            try:
                return str(t.invoke(tool_input))
            except Exception as exc:  # tool failure must not kill the loop
                return f"Tool error: {exc}"
    return f"Unknown tool '{name}'. Available: {', '.join(t.name for t in get_tools())}."


def _build_agent() -> Tuple[Any, str]:
    settings = get_settings()
    if not settings.is_configured:
        raise AgentNotConfigured(
            "%s is not set. Add it to Backend/.env and restart the server."
            % (settings.missing_key_env or "The provider API key")
        )
    try:
        return _build_langgraph_agent(), "langgraph"
    except ImportError:
        logger.info("langgraph tier unavailable (missing deps or non-Gemini provider) — using OpenAI-compatible tier.")
    try:
        return _build_openai_compat_agent(), "openai_compat"
    except ImportError as exc:
        raise AgentNotConfigured(
            "Missing agent dependencies. Install them with: pip install -r requirements.txt"
        ) from exc


def get_agent() -> Tuple[Any, str]:
    """Lazy singleton — returns (runner, mode)."""
    global _agent, _agent_error, _mode
    if _agent is None and _agent_error is None:
        try:
            _agent, _mode = _build_agent()
            _settings = get_settings()
            logger.info(
                "Agent ready | mode=%s provider=%s model=%s",
                _mode, _settings.provider_id, _settings.resolved_model,
            )
        except Exception as exc:  # noqa: BLE001 — surface any init failure as a readable message
            _agent_error = str(exc)
            logger.exception("Agent initialization failed")
    if _agent is None:
        raise AgentNotConfigured(_agent_error or "Agent is not configured.")
    return _agent, _mode


def reset_agent() -> None:
    """Drop the cached agent (used by tests and after config changes)."""
    global _agent, _agent_error, _mode, _checkpointer, _histories
    _agent = None
    _agent_error = None
    _mode = None
    _checkpointer = None
    _histories = {}


# ---------------------------------------------------------------------------
# Session memory for the fallback tier (LangGraph manages its own).
# ---------------------------------------------------------------------------

def _fallback_history(session_id: str) -> List:
    history = _histories.setdefault(session_id, deque(maxlen=16))
    return list(history)


def _remember_fallback(session_id: str, human: str, ai: str) -> None:
    history = _histories.setdefault(session_id, deque(maxlen=16))
    history.append(("human", human))
    history.append(("ai", ai))


def _session_history(session_id: str) -> List:
    if _mode == "langgraph":
        return []  # memory lives inside the checkpointer
    return _fallback_history(session_id)


# ---------------------------------------------------------------------------
# Tier 1 streaming: LangGraph astream over ["updates", "messages"].
# ---------------------------------------------------------------------------

async def _stream_langgraph(message: str, session_id: str) -> AsyncIterator[Dict[str, Any]]:
    from langgraph.errors import GraphRecursionError  # lazy: needs langgraph installed

    settings = get_settings()
    agent, _mode = get_agent()

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": max(2 * settings.agent_max_iterations + 1, 25),
    }

    started = time.perf_counter()
    final_text = None  # type: Optional[str]
    try:
        async for mode, chunk in agent.astream(
            {"messages": [("user", message)]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node_name, delta in (chunk or {}).items():
                    if node_name != "agent" or not delta:
                        continue
                    messages = delta.get("messages", [])
                    msg = messages[-1] if messages else None
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {
                                "type": "tool_start",
                                "tool": tc["name"],
                                "label": TOOL_LABELS.get(tc["name"], tc["name"]),
                                "args": tc.get("args", {}),
                            }

            elif mode == "messages":
                msg_chunk, _meta = chunk
                if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                    text = _content_text(msg_chunk.content)
                    if text:
                        yield {"type": "token", "text": text}

        state = await agent.aget_state(config)
        for msg in reversed(state.values.get("messages", [])):
            if isinstance(msg, AIMessage) and not msg.tool_calls and _content_text(msg.content).strip():
                final_text = _content_text(msg.content)
                break
        if final_text is None:
            final_text = "(The agent returned no answer.)"

        logger.info("Agent turn complete | mode=langgraph session=%s duration=%.2fs", session_id, time.perf_counter() - started)
        yield {"type": "final", "text": final_text}

    except GraphRecursionError:
        yield {
            "type": "error",
            "text": "The agent took too many steps and was stopped. Please rephrase your question.",
        }
    except Exception as exc:
        logger.exception("Agent run failed | mode=langgraph session=%s", session_id)
        yield {"type": "error", "text": _friendly_error(exc)}


# ---------------------------------------------------------------------------
# Tier 2 streaming: text-protocol tool loop (see _TEXTLOOP_SYSTEM_PROMPT).
# ---------------------------------------------------------------------------

async def _stream_openai_compat(message: str, session_id: str) -> AsyncIterator[Dict[str, Any]]:
    """Text-protocol tool loop over a plain chat model.

    Why not AgentExecutor: Gemini 3 requires thought_signature echoes on
    tool-call history, which the available OpenAI-compat adapter drops. A
    text protocol keeps the whole conversation plain text — no signatures
    needed, works on any OpenAI-compatible endpoint.
    """
    llm, _mode = get_agent()
    settings = get_settings()

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    def _system() -> SystemMessage:
        return SystemMessage(content=_TEXTLOOP_SYSTEM_PROMPT.format(tool_catalog=_tool_catalog()))

    history = _fallback_history(session_id)
    convo: List = [_system()]
    for role, text in history:
        convo.append(HumanMessage(content=text) if role == "human" else AIMessage(content=text))
    convo.append(HumanMessage(content=message))

    started = time.monotonic()
    final_text: Optional[str] = None
    max_steps = max(settings.agent_max_iterations, 1)
    step = -1

    # Protocol keywords are case-insensitive at reply start; we hold back the
    # first few chars of each reply until FINAL:/ACTION: can be ruled out, so
    # protocol machinery never leaks into the chat while still streaming.
    keyword_re = re.compile(r"^(FINAL|ACTION)\s*:\s*", re.IGNORECASE)

    try:
        yield {"type": "status", "text": "Thinking…"}
        for step in range(max_steps):
            mode = "detecting"  # detecting -> plain (stream it) | protocol (hide it)
            buffered = ""  # leading text held back while detecting
            reply_parts: List[str] = []
            last_chunk_at = time.monotonic()

            # Drive the model stream and a keep-alive ticker concurrently so
            # the UI always sees life during long Gemini "thinking" pauses.
            stream_aiter = llm.astream(convo).__aiter__()
            heartbeat: Optional[asyncio.Task] = None
            try:
                while True:
                    if heartbeat is None:
                        heartbeat = asyncio.ensure_future(stream_aiter.__anext__())
                    done, _pending = await asyncio.wait(
                        {heartbeat}, timeout=2.0, return_when=asyncio.FIRST_COMPLETED
                    )
                    if not done:
                        elapsed = int(time.monotonic() - started)
                        yield {"type": "status", "text": f"Thinking… {elapsed}s"}
                        continue
                    try:
                        chunk = heartbeat.result()
                    except StopAsyncIteration:
                        break
                    heartbeat = None
                    last_chunk_at = time.monotonic()
                    text = _content_text(getattr(chunk, "content", ""))
                    if not text:
                        continue
                    reply_parts.append(text)
                    if mode == "plain":
                        yield {"type": "token", "text": text}
                    elif mode == "detecting":
                        buffered += text
                        if keyword_re.match(buffered):
                            mode = "protocol"  # ACTION:/FINAL: reply — never shown raw
                        elif len(buffered) >= 16 or "\n" in buffered:
                            mode = "plain"
                            yield {"type": "token", "text": buffered}
                            buffered = ""
                    # mode == "protocol": accumulate silently
            finally:
                if heartbeat is not None and not heartbeat.done():
                    heartbeat.cancel()

            reply_text = "".join(reply_parts).strip()
            action = _parse_action(reply_text)

            if action is None:
                # Final answer. If already streamed as plain text, don't
                # re-emit; otherwise strip the FINAL: prefix and emit whole.
                if mode == "plain":
                    final_text = reply_text or "(The agent returned no answer.)"
                else:
                    stripped = re.sub(r"^FINAL:\s*", "", reply_text, flags=re.IGNORECASE).strip()
                    final_text = stripped or "(The agent returned no answer.)"
                break

            tool_name, tool_input = action
            label = TOOL_LABELS.get(tool_name, tool_name)
            yield {
                "type": "tool_start",
                "tool": tool_name,
                "label": label,
                "args": {"input": tool_input},
            }
            # Tools do blocking I/O (HTTP) — run them off the event loop so
            # the stream and heartbeats keep flowing.
            observation = await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(_invoke_tool_by_name, tool_name, tool_input)
            )
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "label": label,
                "summary": "…",
            }
            yield {"type": "status", "text": label + "… done, thinking"}

            # Feed the result back so the model can produce the final answer.
            convo.append(AIMessage(content=reply_text))
            convo.append(HumanMessage(content=f"OBSERVATION: {observation[:2000]}"))
        else:
            final_text = "I couldn't complete this request within the allowed number of steps. Please rephrase."

        _remember_fallback(session_id, message, final_text)
        logger.info(
            "Agent turn complete | mode=openai_compat steps=%d duration=%.2fs",
            min(step + 1, max_steps), time.monotonic() - started,
        )
        yield {"type": "final", "text": final_text}

    except Exception as exc:
        logger.exception("Agent run failed | mode=openai_compat session=%s", session_id)
        yield {"type": "error", "text": _friendly_error(exc)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def stream_agent_events(message: str, session_id: str) -> AsyncIterator[Dict[str, Any]]:
    """Yield normalized streaming events for one user turn (either tier)."""
    _agent_ref, mode = get_agent()
    if mode == "langgraph":
        async for event in _stream_langgraph(message, session_id):
            yield event
    else:
        async for event in _stream_openai_compat(message, session_id):
            yield event


def _friendly_error(exc: Exception) -> str:
    """Translate provider errors into plain, actionable user-facing text."""
    text = str(exc)
    name = type(exc).__name__
    settings = get_settings()
    provider = settings.provider_id
    model = settings.resolved_model
    label = settings.provider_label
    if provider == "gemini":
        if "429" in text or "quota" in text.lower() or "RateLimit" in name:
            if "PerDay" in text or "per day" in text.lower():
                return (
                    "Gemini free-tier DAILY quota exhausted for this model "
                    "(e.g. 20 requests/day on gemini-3.8-flash). It resets in ~24h — "
                    "or set MODEL_NAME in Backend/.env to another model (each has its "
                    "own daily pool), or enable billing on your API key. "
                    "Tip: switch LLM_PROVIDER to groq (free key at console.groq.com)."
                )
            return (
                "Gemini quota exceeded for your API key (free-tier limit). "
                "Wait about a minute and try again — or review limits at "
                "https://ai.google.dev/gemini-api/docs/rate-limits."
            )
    elif "429" in text or "quota" in text.lower() or "RateLimit" in name:
        return (
            f"{label} rate limit reached (free-tier limits are per minute "
            "for this key). Wait a few seconds and try again."
        )
    if "503" in text or "high demand" in text.lower() or "UNAVAILABLE" in text:
        return f"{label} is temporarily overloaded (high demand). Please try again in a moment."
    if "404" in text and ("no longer available" in text or "model_not_found" in text or "does not exist" in text):
        return (
            f"The model '{model}' was not found on {label}. Set MODEL_NAME in "
            f"Backend/.env to a model offered by {label} and restart the server."
        )
    if "401" in text or "invalid api key" in text.lower() or "invalid_api_key" in text or "unauthorized" in text.lower():
        env_var = settings.missing_key_env or "the provider API key"
        return f"{label} rejected the API key. Check {env_var} in Backend/.env and restart the server."
    return f"The agent hit an error: {exc}"


def _content_text(content: Any) -> str:
    """Flatten Gemini/LangChain content (str or content blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content else ""


# Backwards-compatible non-streaming helper for /agent-chat.
async def run_agent(message: str, session_id: str) -> str:
    """Run one turn to completion and return the final answer text."""
    final = None  # type: Optional[str]
    async for event in stream_agent_events(message, session_id):
        if event["type"] == "final":
            final = event["text"]
        elif event["type"] == "error":
            return event["text"]
    return final or "The agent returned no answer."

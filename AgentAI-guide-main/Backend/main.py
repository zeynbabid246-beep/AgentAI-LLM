"""FastAPI application: SSE streaming chat endpoint + health + hardening.

Endpoints:
  GET  /health         — liveness + model info
  POST /chat/stream    — SSE stream (token / tool_start / tool_end / final / error events)
  POST /agent-chat     — legacy non-streaming endpoint (kept for compatibility)
"""

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import AgentNotConfigured, run_agent, stream_agent_events
from config import get_settings
from schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Rate limiting — per-session sliding window, in-memory (single-process).
# ---------------------------------------------------------------------------

class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, seconds_until_retry)."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - hits[0]) + 1)
            return False, retry_after
        hits.append(now)
        return True, 0


limiter = SlidingWindowLimiter(get_settings().rate_limit_per_minute)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if not settings.is_configured:
        logger.warning(
            "%s is not set — chat endpoints will return 503. "
            "Add it to Backend/.env and restart.",
            settings.missing_key_env or "The provider API key",
        )
    else:
        logger.info(
            "Starting up | provider=%s model=%s",
            settings.provider_id, settings.resolved_model,
        )
    yield
    logger.info("Shutting down")


app = FastAPI(title="Smart Agent API", version="6.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    configured = settings.is_configured
    return HealthResponse(
        status="ok" if configured else "degraded",
        model=settings.resolved_model if configured else None,
        provider=settings.provider_id,
        provider_label=settings.provider_label,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    settings = get_settings()
    if not settings.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Agent not configured: %s is missing." % (settings.missing_key_env or "the provider API key"),
        )

    allowed, retry_after = limiter.check(request.session_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry in ~{retry_after}s.")

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in stream_agent_events(request.message, request.session_id):
                payload = {key: event[key] for key in ("type", "text", "tool", "label", "args", "summary") if key in event}
                yield _sse(payload)
        except AgentNotConfigured as exc:
            yield _sse({"type": "error", "text": str(exc)})
        except Exception:
            logger.exception("Unhandled error in stream | session=%s", request.session_id)
            yield _sse({"type": "error", "text": "Internal server error."})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent-chat", response_model=ChatResponse)
async def agent_chat(request: ChatRequest) -> ChatResponse:
    """Legacy non-streaming endpoint — same agent, same memory, one JSON reply."""
    settings = get_settings()
    if not settings.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Agent not configured: %s is missing." % (settings.missing_key_env or "the provider API key"),
        )

    allowed, retry_after = limiter.check(request.session_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry in ~{retry_after}s.")

    try:
        answer = await run_agent(request.message, request.session_id)
    except AgentNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(response=answer, session_id=request.session_id)


def _sse(payload: Dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Static frontend — served from the same origin as the API (registered last,
# so the API routes above always take precedence).
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
else:  # pragma: no cover — repo checkouts always have frontend/
    logger.warning("Frontend directory not found at %s — UI not served.", _FRONTEND_DIR)

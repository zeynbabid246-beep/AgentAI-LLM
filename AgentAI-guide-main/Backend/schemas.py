"""Pydantic models for the chat API and the SSE event stream."""

from typing import Optional

from pydantic import BaseModel, Field

from config import get_settings


_settings = get_settings()
_MAX_MSG = max(_settings.max_message_chars, 1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_MSG, description="User message")
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Conversation session identifier",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    model: Optional[str] = None
    provider: str = "gemini"
    provider_label: str = "Google Gemini (AI Studio)"

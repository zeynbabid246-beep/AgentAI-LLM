"""Application configuration, loaded from environment / .env.

Fails fast at startup when the selected provider's API key is missing,
instead of surfacing a cryptic error on the first chat request.

Providers (LLM_PROVIDER): gemini | groq | ollama | openrouter | cerebras.
All are consumed through one OpenAI-compatible endpoint, so the same agent
tool loop serves every provider.
"""

from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Provider registry — one OpenAI-compatible endpoint per provider.
# ---------------------------------------------------------------------------
# env_key: which Settings attribute carries the API key ("none" = keyless).
# default_model: used when MODEL_NAME is not explicitly set in the env.

PROVIDERS: Dict[str, Dict[str, str]] = {
    "gemini": {
        "env_key": "google_api_key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.8-flash",
        "label": "Google Gemini (AI Studio)",
    },
    "groq": {
        "env_key": "groq_api_key",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "qwen/qwen3.8-27b",
        "label": "Groq Cloud",
    },
    "ollama": {
        "env_key": "none",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "label": "Ollama (local)",
    },
    "openrouter": {
        "env_key": "openrouter_api_key",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "label": "OpenRouter",
    },
    "cerebras": {
        "env_key": "cerebras_api_key",
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": "llama-3.3-70b",
        "label": "Cerebras Inference",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider selection ---
    # gemini | groq | ollama | openrouter | cerebras (lowercase).
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")

    # --- API keys (only the selected provider's key is required) ---
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    # Ollama needs no key; OPENAI_BASE_URL above points it at localhost.

    # --- Model ---
    # Per-provider default applied when MODEL_NAME is unset in the env.
    model_name: Optional[str] = Field(default=None, alias="MODEL_NAME")
    model_temperature: float = Field(default=0.0, alias="MODEL_TEMPERATURE")
    # Gemini 3 models think by default, which adds seconds per call.
    # "low" keeps tool-calling sharp while cutting most of that latency.
    # Only sent to Gemini — other providers reject the extra kwarg (422).
    model_reasoning_effort: str = Field(default="low", alias="MODEL_REASONING_EFFORT")
    # OpenAI-compatible Gemini endpoint (Google exposes one natively).
    # Other providers use their own base_url from PROVIDERS above.
    gemini_openai_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        alias="GEMINI_OPENAI_BASE_URL",
    )

    # --- Agent behaviour ---
    agent_max_iterations: int = Field(default=8, alias="AGENT_MAX_ITERATIONS")

    # --- API hardening ---
    cors_origins: List[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500"],
        alias="CORS_ORIGINS",
    )
    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE")
    max_message_chars: int = Field(default=4000, alias="MAX_MESSAGE_CHARS")

    # -----------------------------------------------------------------
    # Derived provider info
    # -----------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        pid = (self.llm_provider or "gemini").strip().lower()
        return pid if pid in PROVIDERS else "gemini"

    @property
    def provider_label(self) -> str:
        return PROVIDERS[self.provider_id]["label"]

    @property
    def provider_base_url(self) -> str:
        if self.provider_id == "gemini":
            return self.gemini_openai_base_url
        return PROVIDERS[self.provider_id]["base_url"]

    @property
    def provider_api_key(self) -> str:
        env_key = PROVIDERS[self.provider_id]["env_key"]
        if env_key == "none":
            return "ollama"  # Ollama ignores it; SDK requires non-empty
        return getattr(self, env_key, "").strip()

    @property
    def resolved_model(self) -> str:
        explicit = (self.model_name or "").strip()
        if explicit:
            return explicit
        return PROVIDERS[self.provider_id]["default_model"]

    @property
    def is_configured(self) -> bool:
        return bool(self.provider_api_key)

    @property
    def missing_key_env(self) -> str:
        """Env var name of the API key the selected provider needs."""
        env_key = PROVIDERS[self.provider_id]["env_key"]
        if env_key == "none":
            return ""
        aliases = {
            "google_api_key": "GOOGLE_API_KEY",
            "groq_api_key": "GROQ_API_KEY",
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "cerebras_api_key": "CEREBRAS_API_KEY",
        }
        return aliases.get(env_key, env_key.upper())


@lru_cache
def get_settings() -> Settings:
    return Settings()

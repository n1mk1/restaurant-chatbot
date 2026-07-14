from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    app_name: str = "Maple & Ember Restaurant Chatbot"
    api_prefix: str = Field(
        default="/api/v1",
        pattern=r"^/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*$",
    )
    chat_provider: Literal["ollama", "openai", "deterministic"] = "ollama"
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_reasoning: bool = False
    ollama_keep_alive: str = "15m"
    ollama_timeout_seconds: float = Field(default=20, gt=0, le=120)
    # Optional lightweight, non-reasoning Ollama model used only for the off-topic
    # conversational redirect. Empty = reuse the primary model. See app.graph.
    ollama_offtopic_model: str = ""
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    max_turns_per_session: int = Field(default=20, ge=1, le=100)
    session_ttl_seconds: float = Field(default=1800, gt=0)
    max_active_sessions: int = Field(default=1000, ge=1, le=100_000)
    max_history_messages: int = Field(default=16, ge=2, le=100)
    max_message_chars: int = Field(default=2000, ge=1, le=2000)
    limit_warning_threshold: int = Field(default=3, ge=0, le=20)

    @field_validator("max_history_messages")
    @classmethod
    def require_complete_exchanges(cls, value: int) -> int:
        if value % 2:
            raise ValueError("max_history_messages must be even")
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        str_strip_whitespace=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

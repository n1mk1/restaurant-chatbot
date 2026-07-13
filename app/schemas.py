from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, field_validator


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    session_id: UUID | None = None
    request_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    session_id: UUID
    response: str
    intent: str
    turns_used: int
    turns_remaining: int
    expires_at: datetime
    limit_warning: str | None = None


class SessionResponse(BaseModel):
    session_id: UUID
    turns_remaining: int
    expires_at: datetime


class HealthResponse(BaseModel):
    status: str
    chat_mode: str
    active_sessions: int = Field(ge=0)


class ReadyResponse(BaseModel):
    status: str
    chat_mode: str
    model: str | None = None

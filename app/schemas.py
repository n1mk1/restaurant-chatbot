from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from app.context import Intent


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    session_id: UUID | None = None
    request_id: UUID | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    response: str = Field(min_length=1)
    intent: Intent
    turns_used: int = Field(ge=0)
    turns_remaining: int = Field(ge=0)
    expires_at: datetime
    limit_warning: str | None = None


class SessionResponse(BaseModel):
    session_id: UUID
    turns_remaining: int = Field(ge=0)
    expires_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chat_mode: str
    active_sessions: int = Field(ge=0)


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    chat_mode: str
    model: str | None = None

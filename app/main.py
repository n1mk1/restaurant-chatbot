import logging
from contextlib import asynccontextmanager
from html import escape
from math import ceil
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama import AsyncClient as OllamaClient

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ReadyResponse, SessionResponse
from app.service import ChatService, TurnResult
from app.sessions import (
    InMemorySessionStore,
    RequestConflictError,
    SessionCapacityError,
    SessionExpiredError,
    SessionLimitError,
    SessionNotFoundError,
    SessionRecord,
)

logger = logging.getLogger(__name__)


def _build_model(settings: Settings):
    if settings.chat_provider == "deterministic":
        return None, "deterministic"
    if settings.chat_provider == "ollama":
        return (
            ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.2,
                num_predict=120,
                reasoning=settings.ollama_reasoning,
                keep_alive=settings.ollama_keep_alive,
                async_client_kwargs={"timeout": settings.ollama_timeout_seconds},
                sync_client_kwargs={"timeout": settings.ollama_timeout_seconds},
            ),
            f"ollama:{settings.ollama_model}",
        )
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when CHAT_PROVIDER=openai")
    return (
        ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.2,
            max_tokens=400,
            timeout=20,
            max_retries=2,
        ),
        f"openai:{settings.openai_model}",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    static_dir = Path(__file__).resolve().parent / "static"
    index_template = (static_dir / "index.html").read_text(encoding="utf-8")
    store = InMemorySessionStore(
        ttl_seconds=resolved.session_ttl_seconds,
        max_sessions=resolved.max_active_sessions,
    )
    model, chat_mode = _build_model(resolved)
    service = ChatService(resolved, store, model, chat_mode)
    if resolved.chat_provider == "ollama":
        chat_label = f"Powered locally by {resolved.ollama_model}"
    elif resolved.chat_provider == "openai":
        chat_label = f"Powered by {resolved.openai_model}"
    else:
        chat_label = "Deterministic concierge"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Starting %s in %s mode", resolved.app_name, service.chat_mode)
        yield

    application = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        description="A bounded-session restaurant assistant orchestrated with LangGraph.",
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.chat_service = service
    application.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'",
        )
        if request.url.path.startswith((resolved.api_prefix, "/health", "/ready")):
            response.headers.setdefault("Cache-Control", "no-store")
        elif request.url.path.startswith("/assets/"):
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        return response

    def get_service(request: Request) -> ChatService:
        return request.app.state.chat_service

    ServiceDependency = Annotated[ChatService, Depends(get_service)]

    def session_payload(session: SessionRecord) -> SessionResponse:
        return SessionResponse(
            session_id=session.session_id,
            turns_remaining=resolved.max_turns_per_session - session.turns_used,
            expires_at=store.expires_at(session),
        )

    def chat_payload(result: TurnResult) -> ChatResponse:
        remaining = resolved.max_turns_per_session - result.turns_used
        warning = None
        if remaining <= resolved.limit_warning_threshold:
            warning = f"{remaining} turn{'s' if remaining != 1 else ''} remaining in this session."
        return ChatResponse(
            session_id=result.session.session_id,
            response=result.response,
            intent=result.intent,
            turns_used=result.turns_used,
            turns_remaining=remaining,
            expires_at=store.expires_at(result.session),
            limit_warning=warning,
        )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            chat_mode=service.chat_mode,
            active_sessions=await store.count_active(),
        )

    @application.get("/ready", response_model=ReadyResponse, tags=["system"])
    async def ready() -> ReadyResponse:
        if resolved.chat_provider == "ollama":
            try:
                async with OllamaClient(
                    host=resolved.ollama_base_url,
                    timeout=resolved.ollama_timeout_seconds,
                ) as client:
                    models = await client.list()
                available = {item.model for item in models.models}
            except Exception as exc:
                logger.warning("Ollama readiness check failed: %s", type(exc).__name__)
                raise HTTPException(status_code=503, detail="The local model service is unavailable.") from exc
            required = {resolved.ollama_model}
            if resolved.ollama_offtopic_model:
                required.add(resolved.ollama_offtopic_model)
            missing = sorted(required - available)
            if missing:
                logger.warning("Configured Ollama model(s) missing: %s", ", ".join(missing))
                raise HTTPException(status_code=503, detail="A configured local model is not installed.")
        return ReadyResponse(
            status="ready",
            chat_mode=service.chat_mode,
            model=resolved.ollama_model if resolved.chat_provider == "ollama" else None,
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend(request: Request) -> HTMLResponse:
        replacements = {
            "{{ORIGIN}}": escape(str(request.base_url).rstrip("/"), quote=True),
            "{{API_PREFIX}}": resolved.api_prefix,
            "{{CHAT_LABEL}}": escape(chat_label),
            "{{MAX_TURNS}}": str(resolved.max_turns_per_session),
            "{{SESSION_TTL_SECONDS}}": str(resolved.session_ttl_seconds),
            "{{SESSION_TTL_MINUTES}}": str(ceil(resolved.session_ttl_seconds / 60)),
            "{{MAX_MESSAGE_CHARS}}": str(resolved.max_message_chars),
            "{{LIMIT_WARNING_THRESHOLD}}": str(resolved.limit_warning_threshold),
        }
        page = index_template
        for placeholder, value in replacements.items():
            page = page.replace(placeholder, value)
        return HTMLResponse(page)

    @application.post(
        f"{resolved.api_prefix}/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["chat"],
    )
    async def create_session(chat_service: ServiceDependency) -> SessionResponse:
        try:
            return session_payload(await chat_service.create_session())
        except SessionCapacityError as exc:
            raise HTTPException(status_code=429, detail="Active session capacity reached; try again later.") from exc

    @application.delete(
        f"{resolved.api_prefix}/sessions/{{session_id}}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["chat"],
    )
    async def delete_session(session_id: UUID, chat_service: ServiceDependency) -> Response:
        try:
            await chat_service.store.delete(session_id)
        except SessionExpiredError as exc:
            raise HTTPException(status_code=410, detail="Session expired; create a new session.") from exc
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    async def handle_chat(payload: ChatRequest, chat_service: ServiceDependency) -> ChatResponse:
        if len(payload.message) > resolved.max_message_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Message exceeds the {resolved.max_message_chars}-character limit.",
            )
        try:
            result = await chat_service.chat(
                message=payload.message,
                session_id=payload.session_id,
                request_id=payload.request_id,
            )
            return chat_payload(result)
        except SessionExpiredError as exc:
            raise HTTPException(status_code=410, detail="Session expired; create a new session.") from exc
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found; omit session_id or create a session first.") from exc
        except RequestConflictError as exc:
            raise HTTPException(status_code=409, detail="request_id was already used with a different message.") from exc
        except SessionCapacityError as exc:
            raise HTTPException(status_code=429, detail="Active session capacity reached; try again later.") from exc
        except SessionLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail=f"Session turn limit ({resolved.max_turns_per_session}) reached; create a new session.",
            ) from exc
        except Exception as exc:
            logger.warning("Chat provider or graph failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="The chat service is temporarily unavailable.") from exc

    application.add_api_route(
        f"{resolved.api_prefix}/chat",
        handle_chat,
        methods=["POST"],
        response_model=ChatResponse,
        responses={
            404: {"description": "Session not found"},
            409: {"description": "Idempotency key reused with a different message"},
            410: {"description": "Session expired"},
            429: {"description": "Turn or active-session limit reached"},
            503: {"description": "Chat provider unavailable"},
        },
        tags=["chat"],
    )
    # Convenience alias for small clients and examples.
    application.add_api_route(
        "/chat",
        handle_chat,
        methods=["POST"],
        response_model=ChatResponse,
        include_in_schema=False,
    )
    return application


app = create_app()

import asyncio
from uuid import UUID, uuid4

import httpx
import pytest

from app.config import Settings
from app.main import create_app


@pytest.mark.asyncio
async def test_expired_session_returns_gone():
    app = create_app(
        Settings(
            chat_provider="deterministic",
            openai_api_key=None,
            max_turns_per_session=2,
            session_ttl_seconds=0.01,
            max_active_sessions=10,
            max_history_messages=8,
            max_message_chars=100,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
        await asyncio.sleep(0.02)
        response = await client.post("/chat", json={"session_id": session_id, "message": "Hello"})
    assert response.status_code == 410
    assert "create a new session" in response.json()["detail"]


@pytest.mark.asyncio
async def test_failed_graph_call_does_not_consume_a_turn():
    app = create_app(
        Settings(
            chat_provider="deterministic",
            openai_api_key=None,
            max_turns_per_session=1,
            session_ttl_seconds=60,
            max_active_sessions=10,
            max_history_messages=8,
            max_message_chars=100,
        )
    )
    service = app.state.chat_service

    async def fail(_):
        raise RuntimeError("simulated provider failure")

    service.graph.ainvoke = fail
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
        failed = await client.post("/chat", json={"session_id": session_id, "message": "Hello"})
        session = await service.store.get(UUID(session_id))
    assert failed.status_code == 503
    assert session.turns_used == 0
    assert session.messages == []


@pytest.mark.asyncio
async def test_capacity_rejects_new_session_without_evicting_existing_one():
    app = create_app(
        Settings(
            chat_provider="deterministic",
            openai_api_key=None,
            max_turns_per_session=2,
            session_ttl_seconds=60,
            max_active_sessions=1,
            max_history_messages=8,
            max_message_chars=100,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
        rejected = await client.post("/api/v1/sessions")
        existing = await client.post("/chat", json={"session_id": session_id, "message": "Hello"})
    assert rejected.status_code == 429
    assert existing.status_code == 200


@pytest.mark.asyncio
async def test_idempotency_conflict_and_replay_use_current_quota():
    app = create_app(
        Settings(
            chat_provider="deterministic",
            openai_api_key=None,
            max_turns_per_session=3,
            session_ttl_seconds=60,
            max_active_sessions=10,
            max_history_messages=8,
            max_message_chars=100,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
        request_id = str(uuid4())
        first = await client.post(
            "/chat",
            json={"session_id": session_id, "request_id": request_id, "message": "Hello"},
        )
        assert first.status_code == 200
        assert (await client.post("/chat", json={"session_id": session_id, "message": "Menu"})).status_code == 200
        replay = await client.post(
            "/chat",
            json={"session_id": session_id, "request_id": request_id, "message": "Hello"},
        )
        conflict = await client.post(
            "/chat",
            json={"session_id": session_id, "request_id": request_id, "message": "Different text"},
        )
    assert replay.status_code == 200
    assert replay.json()["turns_used"] == 2
    assert replay.json()["turns_remaining"] == 1
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_health_prunes_expired_sessions():
    app = create_app(
        Settings(
            chat_provider="deterministic",
            openai_api_key=None,
            max_turns_per_session=2,
            session_ttl_seconds=0.01,
            max_active_sessions=10,
            max_history_messages=8,
            max_message_chars=100,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/v1/sessions")
        await asyncio.sleep(0.02)
        health = await client.get("/health")
    assert health.json()["active_sessions"] == 0

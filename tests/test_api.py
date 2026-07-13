from uuid import uuid4

import httpx
import pytest


@pytest.mark.asyncio
async def test_health_does_not_require_an_api_key(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chat_mode": "deterministic", "active_sessions": 0}


@pytest.mark.asyncio
async def test_frontend_and_readiness_are_served(client):
    page = await client.get("/")
    script = await client.get("/assets/app.js")
    ready = await client.get("/ready")
    assert page.status_code == 200
    assert "Maple &amp; Ember" in page.text
    assert "http://test/assets/og.png" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert script.status_code == 200
    assert script.headers["cache-control"] == "public, max-age=3600"
    assert "newRequestId" in script.text
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "chat_mode": "deterministic", "model": None}


@pytest.mark.asyncio
async def test_chat_creates_session_and_remembers_dietary_preference(client):
    first = await client.post("/api/v1/chat", json={"message": "I prefer vegan food"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    second = await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "What do you recommend?"},
    )
    body = second.json()
    assert second.status_code == 200
    assert "Charred Cauliflower Steak" in body["response"]
    assert body["turns_used"] == 2
    assert body["turns_remaining"] == 0
    assert body["limit_warning"] == "0 turns remaining in this session."


@pytest.mark.asyncio
async def test_turn_limit_rejects_without_silent_reset(client):
    created = (await client.post("/api/v1/sessions")).json()
    session_id = created["session_id"]
    assert (await client.post("/chat", json={"session_id": session_id, "message": "Hello"})).status_code == 200
    assert (await client.post("/chat", json={"session_id": session_id, "message": "Show the menu"})).status_code == 200

    rejected = await client.post("/chat", json={"session_id": session_id, "message": "One more"})
    assert rejected.status_code == 429
    assert "create a new session" in rejected.json()["detail"]


@pytest.mark.asyncio
async def test_request_id_is_idempotent(client):
    request_id = str(uuid4())
    first = (await client.post("/chat", json={"message": "Hello", "request_id": request_id})).json()
    second = (await client.post(
        "/chat",
        json={"message": "Hello", "session_id": first["session_id"], "request_id": request_id},
    )).json()
    assert second["response"] == first["response"]
    assert second["turns_used"] == 1


@pytest.mark.asyncio
async def test_validation_and_unknown_session(client):
    assert (await client.post("/chat", json={"message": "   "})).status_code == 422
    assert (await client.post("/chat", json={"message": "x" * 101})).status_code == 422
    missing = await client.post("/chat", json={"message": "hello", "session_id": str(uuid4())})
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_delete_session(client):
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
    assert (await client.delete(f"/api/v1/sessions/{session_id}")).status_code == 204
    assert (await client.post("/chat", json={"message": "hello", "session_id": session_id})).status_code == 404


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_overspend_last_turn():
    from app.config import Settings
    from app.main import create_app

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
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        session_id = (await async_client.post("/api/v1/sessions")).json()["session_id"]
        first, second = await __import__("asyncio").gather(
            async_client.post("/chat", json={"message": "Hello", "session_id": session_id}),
            async_client.post("/chat", json={"message": "Menu", "session_id": session_id}),
        )
    assert sorted((first.status_code, second.status_code)) == [200, 429]

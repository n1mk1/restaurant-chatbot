import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        chat_provider="deterministic",
        openai_api_key=None,
        max_turns_per_session=2,
        session_ttl_seconds=60,
        max_active_sessions=10,
        max_history_messages=8,
        max_message_chars=100,
        limit_warning_threshold=1,
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client

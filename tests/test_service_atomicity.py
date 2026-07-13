from copy import deepcopy
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.service import ChatService
from app.sessions import InMemorySessionStore


def make_service(*, max_sessions: int = 10) -> ChatService:
    settings = Settings(
        chat_provider="deterministic",
        max_turns_per_session=10,
        max_history_messages=8,
        session_ttl_seconds=600,
        max_active_sessions=max_sessions,
        max_message_chars=2000,
    )
    return ChatService(
        settings,
        InMemorySessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_active_sessions,
        ),
        model=None,
        chat_mode="deterministic",
    )


class ExplodingIntent:
    def __str__(self) -> str:
        raise RuntimeError("intent conversion must never run")


class MalformedMetadataGraph:
    def __init__(self, intent: object):
        self.intent = intent

    async def ainvoke(self, graph_input):
        return {
            "messages": [*graph_input["messages"], AIMessage(content="apparently valid")],
            "intent": self.intent,
            "context_item_names": ["Ember Burger"],
            "proposed_order_quantities": {"Ember Burger": 2},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent",
    [
        ExplodingIntent(),
        {"type": "reasoning", "reasoning": "SECRET INTERNAL ANALYSIS"},
    ],
)
async def test_malformed_intent_is_rejected_before_any_session_mutation(intent):
    service = make_service()
    session = await service.create_session()
    before_messages = deepcopy(session.messages)
    before_turns = session.turns_used
    before_dietary = set(session.dietary_preferences)
    before_allergens = set(session.allergen_restrictions)
    before_untracked = set(session.untracked_allergen_restrictions)
    before_unverified = set(session.unverified_dietary_restrictions)
    before_order = dict(session.proposed_order_quantities)
    before_cache = deepcopy(session.request_cache)
    service.graph = MalformedMetadataGraph(intent)

    with pytest.raises(RuntimeError, match="invalid intent"):
        await service.chat("I have a dairy allergy", session_id=session.session_id)

    assert session.messages == before_messages
    assert session.turns_used == before_turns
    assert session.dietary_preferences == before_dietary
    assert session.allergen_restrictions == before_allergens
    assert session.untracked_allergen_restrictions == before_untracked
    assert session.unverified_dietary_restrictions == before_unverified
    assert session.proposed_order_quantities == before_order
    assert session.request_cache == before_cache


class MutateThenFailGraph:
    async def ainvoke(self, graph_input):
        graph_input["messages"][0].content = "MUTATED DESPITE FAILURE"
        raise RuntimeError("provider failed")


@pytest.mark.asyncio
async def test_failed_graph_cannot_mutate_stored_message_objects_by_reference():
    service = make_service()
    first = await service.chat("Hello")
    session = first.session
    before_contents = [message.content for message in session.messages]
    service.graph = MutateThenFailGraph()

    with pytest.raises(RuntimeError, match="provider failed"):
        await service.chat("One more", session_id=session.session_id)

    assert [message.content for message in session.messages] == before_contents
    assert session.turns_used == 1


class FailingGraph:
    async def ainvoke(self, graph_input):
        raise RuntimeError("provider failed")


@pytest.mark.asyncio
async def test_failed_anonymous_first_turn_releases_session_capacity():
    service = make_service(max_sessions=1)
    service.graph = FailingGraph()

    with pytest.raises(RuntimeError, match="provider failed"):
        await service.chat("Hello")

    assert await service.store.count_active() == 0
    assert await service.create_session()


@pytest.mark.asyncio
async def test_safety_state_change_invalidates_stale_idempotent_response():
    service = make_service()
    request_id = uuid4()
    first = await service.chat("What do you recommend?", request_id=request_id)
    assert "Roasted Beet Salad" in first.response

    allergy = await service.chat("I have a dairy allergy", session_id=first.session.session_id)
    replay = await service.chat(
        "What do you recommend?",
        session_id=first.session.session_id,
        request_id=request_id,
    )

    assert allergy.session.allergen_restrictions == {"dairy"}
    assert "Roasted Beet Salad" not in replay.response
    assert "Dark Chocolate Torte" not in replay.response
    assert replay.turns_used == 3

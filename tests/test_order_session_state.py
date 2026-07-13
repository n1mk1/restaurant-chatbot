from datetime import timedelta
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.service import ChatService
from app.sessions import InMemorySessionStore, SessionLimitError


def _service(*, turns: int = 5) -> ChatService:
    settings = Settings(
        chat_provider="deterministic",
        max_turns_per_session=turns,
        max_history_messages=8,
        session_ttl_seconds=60,
        max_active_sessions=10,
        max_message_chars=2000,
    )
    return ChatService(settings, InMemorySessionStore(60, 10))


class SequencedGraph:
    def __init__(self, *outputs: object):
        self.outputs = list(outputs)
        self.inputs: list[dict[str, object]] = []

    async def ainvoke(self, graph_input: dict[str, object]) -> dict[str, object]:
        self.inputs.append(graph_input)
        output = self.outputs.pop(0)
        result: dict[str, object] = {
            "messages": [*graph_input["messages"], AIMessage(content="Order state updated.")],  # type: ignore[misc]
            "intent": "menu",
        }
        if output is not _OMIT:
            result["proposed_order_quantities"] = output
        return result


_OMIT = object()


@pytest.mark.asyncio
async def test_order_state_uses_defensive_input_and_output_copies():
    service = _service()
    session = await service.create_session()
    session.proposed_order_quantities = {"Ember Burger": 2}
    graph_output = {"Ember Burger": 3, "Cider-Poached Pear": 1}
    graph = SequencedGraph(graph_output)
    service.graph = graph  # type: ignore[assignment]

    await service.chat("Make that three burgers", session_id=session.session_id)

    assert graph.inputs[0]["proposed_order_quantities"] == {"Ember Burger": 2}
    assert session.proposed_order_quantities == {
        "Ember Burger": 3,
        "Cider-Poached Pear": 1,
    }

    graph.inputs[0]["proposed_order_quantities"]["Ember Burger"] = 99  # type: ignore[index]
    graph_output["Ember Burger"] = 100
    assert session.proposed_order_quantities["Ember Burger"] == 3


@pytest.mark.asyncio
async def test_missing_order_output_preserves_state_and_empty_output_clears_it():
    service = _service()
    session = await service.create_session()
    session.proposed_order_quantities = {"Ember Burger": 2}
    graph = SequencedGraph(_OMIT, {})
    service.graph = graph  # type: ignore[assignment]

    await service.chat("Tell me the hours", session_id=session.session_id)
    assert session.proposed_order_quantities == {"Ember Burger": 2}

    await service.chat("Clear the proposed order", session_id=session.session_id)
    assert session.proposed_order_quantities == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_state",
    [
        None,
        [],
        {"Unknown Item": 1},
        {"Ember Burger": 0},
        {"Ember Burger": -1},
        {"Ember Burger": True},
        {"Ember Burger": 1.5},
    ],
)
async def test_invalid_graph_order_state_is_rejected_atomically(invalid_state):
    service = _service()
    session = await service.create_session()
    session.proposed_order_quantities = {"Ember Burger": 2}
    graph = SequencedGraph(invalid_state)
    service.graph = graph  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="invalid proposed-order state"):
        await service.chat(
            "I have a dairy allergy",
            session_id=session.session_id,
            request_id=uuid4(),
        )

    assert session.proposed_order_quantities == {"Ember Burger": 2}
    assert session.messages == []
    assert session.turns_used == 0
    assert session.allergen_restrictions == set()
    assert session.request_cache == {}


@pytest.mark.asyncio
async def test_idempotent_replay_does_not_restore_an_old_order_snapshot():
    service = _service()
    session = await service.create_session()
    request_id = uuid4()
    graph = SequencedGraph({"Ember Burger": 2}, {"Ember Burger": 3})
    service.graph = graph  # type: ignore[assignment]

    first = await service.chat(
        "Two burgers",
        session_id=session.session_id,
        request_id=request_id,
    )
    await service.chat("Make that three", session_id=session.session_id)
    replay = await service.chat(
        "Two burgers",
        session_id=session.session_id,
        request_id=request_id,
    )

    assert replay.response == first.response
    assert replay.turns_used == 2
    assert session.proposed_order_quantities == {"Ember Burger": 3}
    assert len(graph.inputs) == 2


@pytest.mark.asyncio
async def test_turn_limit_rejection_preserves_order_state_without_calling_graph():
    service = _service(turns=1)
    session = await service.create_session()
    graph = SequencedGraph({"Ember Burger": 2}, {"Ember Burger": 9})
    service.graph = graph  # type: ignore[assignment]

    await service.chat("Two burgers", session_id=session.session_id)
    with pytest.raises(SessionLimitError):
        await service.chat("Nine burgers", session_id=session.session_id)

    assert session.proposed_order_quantities == {"Ember Burger": 2}
    assert session.turns_used == 1
    assert len(graph.inputs) == 1


@pytest.mark.asyncio
async def test_deleted_session_clears_order_state_and_new_session_starts_empty():
    service = _service()
    deleted = await service.create_session()
    deleted.proposed_order_quantities = {"Ember Burger": 2}

    await service.store.delete(deleted.session_id)
    replacement = await service.create_session()

    assert deleted.proposed_order_quantities == {}
    assert replacement.proposed_order_quantities == {}


@pytest.mark.asyncio
async def test_expired_session_pruning_clears_order_state():
    store = InMemorySessionStore(ttl_seconds=60, max_sessions=10)
    session = await store.create()
    session.proposed_order_quantities = {"Ember Burger": 2}
    session.last_activity -= timedelta(seconds=61)

    assert await store.count_active() == 0
    assert session.proposed_order_quantities == {}


@pytest.mark.asyncio
async def test_secondary_menu_context_survives_primary_intent_then_clears_when_absent():
    service = _service()
    session = await service.create_session()

    class ContextGraph:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, graph_input):
            self.calls += 1
            result = {
                "messages": [*graph_input["messages"], AIMessage(content="Helpful response.")],
                "intent": "reservation" if self.calls == 1 else "policy",
            }
            if self.calls == 1:
                result["context_item_names"] = ["Charred Cauliflower Steak"]
                result["context_category"] = "main"
            return result

    service.graph = ContextGraph()  # type: ignore[assignment]

    await service.chat("Can I book and see vegan mains?", session_id=session.session_id)
    assert session.last_item_names == ("Charred Cauliflower Steak",)
    assert session.last_category == "main"

    await service.chat("Where can I park?", session_id=session.session_id)
    assert session.last_item_names == ()
    assert session.last_category is None

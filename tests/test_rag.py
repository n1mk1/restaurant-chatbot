import pytest
from langchain_core.messages import HumanMessage

from app.graph import build_graph
from app.rag import default_preset_store, retrieve_preset


def test_in_memory_index_retrieves_a_meat_only_preset_without_exact_prompt_match():
    preset = retrieve_preset("my eating rules are only animal products")

    assert preset is not None
    assert preset.id == "meat-only-menu"
    assert preset.answer.startswith("The menu can show items not marked vegetarian")


@pytest.mark.parametrize(
    ("query", "preset_id"),
    [
        ("which eating needs can you handle", "dietary-capabilities"),
        ("my diet is paleo", "unverified-diet"),
        ("show me vegan choices", "vegan-menu"),
    ],
)
def test_curated_semantic_presets_cover_paraphrases(query, preset_id):
    preset = retrieve_preset(query)

    assert preset is not None
    assert preset.id == preset_id


@pytest.mark.parametrize("query", ["what is the capital of France", "tell me a joke", "hello there"])
def test_unrelated_messages_do_not_retrieve_a_restaurant_preset(query):
    assert retrieve_preset(query) is None


@pytest.mark.asyncio
async def test_general_semantic_fallback_returns_the_curated_answer():
    result = await build_graph().ainvoke(
        {"messages": [HumanMessage(content="my eating rules are only animal products")]}
    )

    assert result["intent"] == "general"
    assert result["semantic_preset_id"] == "meat-only-menu"
    response = result["messages"][-1].content
    assert "does not verify a meat-only or carnivore requirement" in response
    assert "Ember Burger" in response


@pytest.mark.asyncio
async def test_semantic_fallback_can_correct_an_ambiguous_dietary_route():
    result = await build_graph().ainvoke(
        {"messages": [HumanMessage(content="I have a diet of only animal products")]}
    )

    assert result["semantic_preset_id"] == "meat-only-menu"
    response = result["messages"][-1].content
    assert "does not verify a meat-only or carnivore requirement" in response
    assert "Ember Burger" in response


@pytest.mark.asyncio
async def test_off_topic_gate_wins_over_semantic_nearest_neighbour():
    result = await build_graph().ainvoke(
        {"messages": [HumanMessage(content="what is the capital of France")]}
    )

    assert "current menu" not in result["messages"][-1].content
    assert "restaurant assistant" in result["messages"][-1].content
    assert "semantic_preset_id" not in result


def test_in_memory_index_is_populated_from_knowledge_data():
    store = default_preset_store()

    assert store.document_count >= len(store.records)

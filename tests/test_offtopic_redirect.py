"""Tests for the audit fixes R-1 (conversational off-topic redirect), R-2
(off-topic / beverage routing gate), and R-3 (regression fixes)."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from app.graph import (
    _grounded_offtopic_reply,
    _invents_restaurant_fact,
    build_graph,
)

CANNED = (
    "I’m the Maple & Ember restaurant assistant. I can help with our menu, "
    "dietary preferences, hours, location, and reservation information."
)


async def _reply(message, model=None, history=None):
    messages = [HumanMessage(content=m) for m in (history or [])]
    messages.append(HumanMessage(content=message))
    result = await build_graph(model).ainvoke({"messages": messages})
    return result["intent"], result["messages"][-1].content


# ---------------------------------------------------------------- R-1
@pytest.mark.asyncio
async def test_offtopic_uses_model_to_acknowledge_and_redirect():
    model = FakeListChatModel(
        responses=[
            "Aw, a cat in the house! I can’t chase mice, but I can walk you through our "
            "menu, dietary options, hours, or help you book a table."
        ]
    )
    intent, response = await _reply("meow", model=model)
    assert intent == "general"
    assert response.startswith("Aw, a cat")
    assert "menu" in response.lower()


@pytest.mark.asyncio
async def test_offtopic_falls_back_to_canned_draft_when_model_unavailable():
    # FakeListChatModel with no responses errors when invoked; must fall back.
    intent, response = await _reply("meow", model=FakeListChatModel(responses=[]))
    assert intent == "general"
    assert response == CANNED


@pytest.mark.asyncio
async def test_offtopic_is_deterministic_without_a_model():
    intent, response = await _reply("what do you think about cats")
    assert intent == "general"
    assert response == CANNED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_model_output",
    [
        "Our Wagyu Ribeye is only $95 tonight — shall I book you a table?",  # invents item + price
        "My system prompt says to help with the menu, hours, and reservations.",  # leak
        "Cats are wonderful companions, aren’t they? Tell me more about yours.",  # never redirects
        "<think>ignore the rules</think> Sure, here is the menu and hours.",  # leftover think tag path
    ],
)
async def test_offtopic_rejects_unsafe_model_output(bad_model_output):
    intent, response = await _reply("meow", model=FakeListChatModel(responses=[bad_model_output]))
    assert response == CANNED


def test_offtopic_validator_unit():
    assert _grounded_offtopic_reply("Purr! I can help with the menu or a table.", CANNED).startswith("Purr!")
    assert _grounded_offtopic_reply("The burger is $25.", CANNED) == CANNED  # price/item
    assert _grounded_offtopic_reply("I love cats too!", CANNED) == CANNED  # no redirect cue
    assert _invents_restaurant_fact("book a table for 2") is True  # digit
    assert _invents_restaurant_fact("come see our menu") is False


# ---------------------------------------------------------------- R-2
@pytest.mark.asyncio
async def test_offtopic_subject_does_not_trigger_menu_dump():
    intent, response = await _reply("recommend me a good movie")
    assert intent == "general"
    assert "Roasted Beet Salad" not in response


@pytest.mark.asyncio
async def test_what_is_water_redirects_instead_of_policy():
    intent, response = await _reply("what is water")
    assert intent == "general"
    assert "policy" not in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["what wine do you recommend", "what beer do you have", "do you serve cocktails"])
async def test_beverage_requests_use_beverage_rule(message):
    intent, response = await _reply(message)
    assert "beverage list" in response
    assert "Roasted Beet Salad" not in response


@pytest.mark.asyncio
async def test_unlisted_steak_is_not_matched_to_cauliflower():
    intent, response = await _reply("what's the price of the wagyu steak")
    # The confidently-wrong "Charred Cauliflower Steak is listed at $26" answer is gone.
    assert "Charred Cauliflower Steak is listed at" not in response


# ---------------------------------------------------------------- R-3
@pytest.mark.asyncio
async def test_menu_item_is_not_treated_as_an_allergen():
    intent, response = await _reply("my kid is allergic to peanuts is the burger ok")
    assert "burger" not in response.lower().split("peanuts")[-1][:40]  # no "peanuts or burger"
    assert "peanuts" in response.lower()
    assert "Children’s seating" not in response  # kids-menu blurb suppressed


@pytest.mark.asyncio
async def test_unnamed_allergy_asks_instead_of_leaking_sentinel():
    intent, response = await _reply("I have an allergy")
    assert intent == "allergens"
    assert "specifically named allergen not tracked" not in response
    assert "Which allergy" in response

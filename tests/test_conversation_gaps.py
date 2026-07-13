import unicodedata

import pytest
from langchain_core.messages import HumanMessage

from app.config import Settings
from app.graph import build_graph
from app.restaurant import MENU, RESTAURANT, MenuItem
from app.service import ChatService, TurnResult
from app.sessions import InMemorySessionStore


def _item(name: str) -> MenuItem:
    return next(item for item in MENU if item.name == name)


BURGER = _item("Ember Burger")
PEAR = _item("Cider-Poached Pear")


async def _ask(message: str) -> dict[str, object]:
    return await build_graph().ainvoke({"messages": [HumanMessage(content=message)]})


async def _converse(*messages: str) -> list[TurnResult]:
    settings = Settings(
        chat_provider="deterministic",
        max_turns_per_session=30,
        max_history_messages=8,
        session_ttl_seconds=600,
        max_active_sessions=20,
        max_message_chars=2000,
    )
    service = ChatService(
        settings,
        InMemorySessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_active_sessions,
        ),
        model=None,
        chat_mode="deterministic",
    )
    session_id = None
    results: list[TurnResult] = []
    for message in messages:
        result = await service.chat(message, session_id=session_id)
        session_id = result.session.session_id
        results.append(result)
    return results


def _response(result: dict[str, object]) -> str:
    return result["messages"][-1].content  # type: ignore[index,union-attr]


def _mentioned_items(response: str) -> set[str]:
    return {item.name for item in MENU if item.name in response}


def _fold(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )


def _assert_proposed_order(response: str, quantities: dict[str, int]) -> None:
    expected_subtotal = 0.0
    assert "Proposed order:" in response
    for name, quantity in quantities.items():
        item = _item(name)
        expected_subtotal += item.price * quantity
        assert f"{quantity} \u00d7 {item.name}" in response
        assert f"${item.price * quantity:.0f}" in response
    assert f"Menu-price subtotal: ${expected_subtotal:.0f}" in response
    assert "not been submitted or paid" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Do you have patio seating?",
        "Can I sit outside?",
        "Are there tables outside?",
    ],
)
async def test_unverified_outdoor_seating_questions_get_a_direct_policy_answer(message):
    result = await _ask(message)
    response = _response(result)

    assert result["intent"] == "policy"
    assert "confirmed" in response.lower()
    assert RESTAURANT["phone"] in response
    assert not _mentioned_items(response)
    assert "restaurant assistant" not in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Do you have space for 5?",
        "Can we come at 7?",
    ],
)
async def test_implicit_booking_availability_phrases_route_to_reservations(message):
    result = await _ask(message)
    response = _response(result)

    assert result["intent"] == "reservation"
    assert RESTAURANT["reservation_url"] in response
    assert RESTAURANT["phone"] in response
    assert "live" in response.lower()
    assert "availability" in response.lower()
    assert "restaurant assistant" not in response.lower()
    assert not _mentioned_items(response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_item"),
    [
        ("Is the salmon available tonight?", "Maple-Glazed Salmon"),
        ("Is the burger available?", "Ember Burger"),
        ("Do you have the burger tonight?", "Ember Burger"),
    ],
)
async def test_live_item_availability_is_not_presented_as_a_static_menu_fact(message, expected_item):
    result = await _ask(message)
    response = _response(result)

    assert result["intent"] == "menu"
    assert expected_item in response
    assert "live item availability" in response.lower()
    assert "not available in chat" in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "required", "forbidden"),
    [
        (
            "Can I use multiple cards to pay?",
            {"payment-method"},
            {"gratuity", "service-charge", "tip"},
        ),
        (
            "Do you take cash, especially Visa?",
            {"payment-method"},
            {"specials list", "today's offerings"},
        ),
        (
            "Can I pay by card and bring a sparkling water?",
            {"payment-method", "beverage list"},
            {"parking", "valet"},
        ),
    ],
)
async def test_policy_matching_uses_words_not_substrings(message, required, forbidden):
    result = await _ask(message)
    response = _response(result).lower()

    assert result["intent"] == "policy"
    assert all(fragment in response for fragment in required)
    assert all(fragment not in response for fragment in forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Can we split the check?",
        "Can we get separate bills?",
        "Can we pay separately?",
    ],
)
async def test_split_bill_synonyms_use_the_unknown_split_bill_policy(message):
    result = await _ask(message)
    response = _response(result)

    assert result["intent"] == "policy"
    assert "split-bill" in response.lower()
    assert "confirmed" in response.lower()
    assert RESTAURANT["phone"] in response


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["Can I get this to go?", "Do you do carryout?"])
async def test_takeout_synonyms_use_the_unknown_takeout_policy(message):
    result = await _ask(message)
    response = _response(result)

    assert result["intent"] == "policy"
    assert "takeout" in response.lower()
    assert "confirmed" in response.lower()
    assert RESTAURANT["phone"] in response


@pytest.mark.asyncio
async def test_tax_question_does_not_dump_the_menu_or_invent_tax_treatment():
    result = await _ask("Do prices include tax?")
    response = _response(result)

    assert result["intent"] == "policy"
    assert "tax" in response.lower()
    assert "not confirmed" in response.lower()
    assert RESTAURANT["phone"] in response
    assert not _mentioned_items(response)


@pytest.mark.asyncio
async def test_currency_question_preserves_the_source_dollar_symbol_without_guessing_a_code():
    result = await _ask("What currency are your prices in?")
    response = _response(result)
    lower = response.lower()

    assert result["intent"] == "policy"
    assert "$" in response
    assert "currency code" in lower
    assert "does not specify" in lower
    assert "cad" not in lower
    assert "usd" not in lower
    assert not _mentioned_items(response)


@pytest.mark.asyncio
async def test_general_fee_question_uses_the_verified_unknown_policy():
    result = await _ask("What fees are there?")
    response = _response(result)

    assert result["intent"] == "policy"
    assert "fee" in response.lower()
    assert "not confirmed" in response.lower()
    assert RESTAURANT["phone"] in response
    assert not _mentioned_items(response)


@pytest.mark.asyncio
async def test_secondary_menu_topic_survives_a_combined_booking_turn():
    vegan_mains = [item for item in MENU if item.category == "main" and item.vegan]
    assert len(vegan_mains) == 1
    vegan_main = vegan_mains[0]

    results = await _converse("Can I book Friday and see the vegan mains?", "How much is it?")

    assert RESTAURANT["reservation_url"] in results[0].response
    assert vegan_main.name in results[0].response
    assert results[1].intent == "menu"
    assert results[1].response == f"{vegan_main.name} is listed at ${vegan_main.price:.0f}."


def _ingredient_expected(*terms: str) -> set[str]:
    normalized_terms = {_fold(term) for term in terms}
    return {
        item.name
        for item in MENU
        if any(term in _fold(item.description) for term in normalized_terms)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Which dishes have cheese?",
            _ingredient_expected("cheese", "parmesan", "cheddar"),
        ),
        (
            "Which dishes contain cheese?",
            _ingredient_expected("cheese", "parmesan", "cheddar"),
        ),
        ("What has goat cheese?", _ingredient_expected("goat cheese")),
        ("What dishes have parmesan?", _ingredient_expected("parmesan")),
        ("Which dishes include cream?", _ingredient_expected("cream", "creme")),
        ("Which dishes contain salmon?", {item.name for item in MENU if "salmon" in item.name.lower()}),
    ],
)
async def test_ingredient_lookup_uses_descriptions_instead_of_broad_allergen_groups(message, expected):
    assert expected
    result = await _ask(message)
    response = _response(result)

    assert result["intent"] == "menu"
    assert _mentioned_items(response) == expected
    assert "tell me every allergy" not in response.lower()


@pytest.mark.asyncio
async def test_order_quantity_and_total_persist_across_follow_up_turns():
    results = await _converse("Two burgers please", "Make that 3", "How much altogether?")

    _assert_proposed_order(results[0].response, {BURGER.name: 2})
    _assert_proposed_order(results[1].response, {BURGER.name: 3})
    _assert_proposed_order(results[2].response, {BURGER.name: 3})


@pytest.mark.asyncio
async def test_ambiguous_quantity_change_with_multiple_lines_requests_clarification():
    results = await _converse("Two burgers and one pear please", "Make that 3")
    response = results[-1].response

    assert "which" in response.lower()
    assert BURGER.name in response
    assert PEAR.name in response
    assert "?" in response
    assert "Menu-price subtotal" not in response


@pytest.mark.asyncio
async def test_named_quantity_change_keeps_other_order_lines():
    results = await _converse("Two burgers and one pear please", "Make that 3 burgers")

    _assert_proposed_order(results[-1].response, {BURGER.name: 3, PEAR.name: 1})


@pytest.mark.asyncio
async def test_add_and_later_total_keep_the_entire_proposed_order():
    results = await _converse("Two burgers please", "Add a pear", "What is the total?")

    _assert_proposed_order(results[1].response, {BURGER.name: 2, PEAR.name: 1})
    _assert_proposed_order(results[2].response, {BURGER.name: 2, PEAR.name: 1})


@pytest.mark.asyncio
async def test_direct_total_supports_plural_item_aliases():
    result = await _ask("What is the total for two burgers and three pears?")

    _assert_proposed_order(_response(result), {BURGER.name: 2, PEAR.name: 3})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "quantity"),
    [
        ("One burger please", 1),
        ("1 burger please", 1),
        ("I'd like a burger", 1),
        ("I'll have a burger", 1),
        ("I want a burger", 1),
        ("2x burgers please", 2),
    ],
)
async def test_common_order_phrasings_create_a_proposed_order(message, quantity):
    result = await _ask(message)

    _assert_proposed_order(_response(result), {BURGER.name: quantity})


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", ["No thanks", "Nothing else", "That is all"])
async def test_explicit_order_closing_is_acknowledged_conversationally(closing):
    results = await _converse("Two burgers please", closing)
    response = results[-1].response

    assert "not been submitted or paid" in response
    assert "restaurant assistant" not in response.lower()
    assert "current listed menu" not in response.lower()
    assert "anything else" not in response.lower()


@pytest.mark.asyncio
async def test_bare_no_after_two_order_questions_is_clarified():
    results = await _converse("Two burgers please", "No")
    response = results[-1].response.lower()

    assert "confirm" in response
    assert "food allergies" in response
    assert "anything else" in response or "nothing else" in response
    assert "?" in response
    assert "restaurant assistant" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "What can I eat?", "What can I have?", "What can we eat?", "Anything I can eat?",
        "What are my choices?", "What works for me?", "Show me what I can eat",
        "Tell me what I can eat", "What can I order?",
    ],
)
async def test_natural_option_questions_route_to_filtered_menu_help(message):
    result = await _ask(message)
    response = _response(result)
    assert result["intent"] == "menu"
    assert _mentioned_items(response)
    assert "restaurant assistant" not in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "flag"),
    [
        ("Are there non-vegan options?", "vegan"),
        ("Do you have non-vegan dishes?", "vegan"),
        ("Can you show me non-vegan food?", "vegan"),
        ("Tell me non-vegan options", "vegan"),
        ("Non-vegan please", "vegan"),
        ("Are any dishes not gluten-free?", "gluten-free"),
    ],
)
async def test_negative_diet_filters_return_the_complement(message, flag):
    result = await _ask(message)
    response = _response(result)
    mentioned = [item for item in MENU if item.name in response]
    assert result["intent"] == "menu"
    assert mentioned
    if flag == "vegan":
        assert all(not item.vegan for item in mentioned)
    else:
        assert all(not item.gluten_free for item in mentioned)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "subject"),
    [
        ("Is there salmon in beet salad?", "Roasted Beet Salad"),
        ("Does risotto include salmon?", "Wild Mushroom Risotto"),
    ],
)
async def test_cross_item_ingredient_questions_answer_the_named_subject(message, subject):
    result = await _ask(message)
    response = _response(result)
    assert result["intent"] == "menu"
    assert subject in response
    assert "does not include salmon" in response.lower()
    assert "tell me every allergy" not in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Any tables tonight?", "Do you have room for 2 tonight?", "Can you fit us in tonight?",
        "Can I get seated at 7?", "Is there a slot at 7 PM?", "Can I make a res for tonight?",
    ],
)
async def test_common_implicit_booking_phrases_get_live_availability_guidance(message):
    result = await _ask(message)
    response = _response(result)
    assert result["intent"] == "reservation"
    assert "live" in response.lower()
    assert "availability" in response.lower()
    assert RESTAURANT["reservation_url"] in response
    assert RESTAURANT["phone"] in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "follow_up",
    [
        "Yes, place it", "Confirm it", "Send it to the kitchen", "Go ahead", "Checkout",
        "Complete the order", "Charge my card", "Take my payment",
    ],
)
async def test_order_action_followups_repeat_the_no_action_boundary(follow_up):
    results = await _converse("Two burgers please", follow_up)
    response = results[-1].response.lower()
    assert results[-1].intent == "policy"
    assert "not been submitted or paid" in response
    assert "can’t submit or confirm" in response

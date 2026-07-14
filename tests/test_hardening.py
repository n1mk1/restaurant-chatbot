from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.graph import build_graph
from app.restaurant import MENU, RESTAURANT
from app.service import ChatService
from app.sessions import InMemorySessionStore


async def ask(message: str, *, model=None):
    return await build_graph(model).ainvoke({"messages": [HumanMessage(content=message)]})


def service_settings(*, history: int = 4, turns: int = 50) -> Settings:
    return Settings(
        chat_provider="deterministic",
        max_turns_per_session=turns,
        max_history_messages=history,
        session_ttl_seconds=60,
        max_active_sessions=50,
        max_message_chars=2000,
    )


async def converse(*messages: str, history: int = 4):
    service = ChatService(service_settings(history=history), InMemorySessionStore(60, 50))
    session_id = None
    results = []
    for message in messages:
        result = await service.chat(message, session_id=session_id)
        session_id = result.session.session_id
        results.append(result)
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "are their bookings available",
        "Are there bookings available?",
        "Do you take bookings?",
        "Are you fully booked tonight?",
        "Is 7 PM available?",
        "Any availability tonight?",
        "Do you have a table at 7?",
        "Can I get a table for two?",
        "A table for 4 tonight please",
        "Can we dine at 7?",
        "Can you host a party of four?",
        "Can I make a reservaton?",
        "Do you take bookigns?",
        "Hi, do you have bookings available?",
    ],
)
async def test_booking_variants_answer_the_question_without_claiming_live_access(message):
    result = await ask(message)
    response = result["messages"][-1].content
    assert result["intent"] == "reservation"
    assert RESTAURANT["reservation_url"] in response
    assert RESTAURANT["phone"] in response
    assert "confirm" in response.lower() or "request" in response.lower()
    assert "restaurant assistant" not in response.lower()


@pytest.mark.asyncio
async def test_reservation_followup_uses_session_context():
    results = await converse("Can I reserve?", "For 7 tonight?")
    assert results[-1].intent == "reservation"
    assert "live table availability" in results[-1].response


@pytest.mark.asyncio
async def test_cancellation_and_group_requests_do_not_invent_terms():
    cancelled = await ask("What is your cancellation policy?")
    group = await ask("Do you have a private room for a large group?")
    assert "not confirmed" in cancelled["messages"][-1].content
    assert RESTAURANT["reservation_url"] in cancelled["messages"][-1].content
    assert "not confirmed" in group["messages"][-1].content
    assert RESTAURANT["phone"] in group["messages"][-1].content


@pytest.mark.asyncio
async def test_full_menu_contains_every_source_item_once():
    result = await ask("Show me tonight's full menu")
    response = result["messages"][-1].content
    for item in MENU:
        assert response.count(item.name) == 1
        assert f"${item.price:.0f}" in response
    assert "Live item availability" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "included", "excluded"),
    [
        ("Show me the starters", {"Roasted Beet Salad", "Crispy Lake Erie Perch"}, {"Ember Burger", "Cider-Poached Pear"}),
        ("Show me all mains", {"Charred Cauliflower Steak", "Wild Mushroom Risotto", "Maple-Glazed Salmon", "Ember Burger"}, {"Roasted Beet Salad", "Dark Chocolate Torte"}),
        ("Show me the desserts", {"Cider-Poached Pear", "Dark Chocolate Torte"}, {"Ember Burger", "Crispy Lake Erie Perch"}),
        ("Show me the dessets", {"Cider-Poached Pear", "Dark Chocolate Torte"}, {"Ember Burger", "Crispy Lake Erie Perch"}),
    ],
)
async def test_plural_categories_and_common_typo_are_scoped(message, included, excluded):
    response = (await ask(message))["messages"][-1].content
    assert all(name in response for name in included)
    assert all(name not in response for name in excluded)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Tell me about the perch", "Crispy Lake Erie Perch"),
        ("Tell me about the mushroom dish", "Wild Mushroom Risotto"),
        ("Do you serve mushrooms?", "Wild Mushroom Risotto"),
        ("Tell me about the chocolate", "Dark Chocolate Torte"),
        ("Tell me about steak", "Charred Cauliflower Steak"),
        ("What vegitarian options do you have?", "Charred Cauliflower Steak"),
    ],
)
async def test_item_aliases_return_the_canonical_item(message, expected):
    result = await ask(message)
    assert result["intent"] == "menu"
    assert expected in result["messages"][-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "subtotal"),
    [
        ("How much are two burgers?", "$50"),
        ("2 burgers and a pear", "$62"),
        ("total for a burger and pear", "$37"),
        ("Seven burgers please", "$175"),
    ],
)
async def test_quantities_use_only_listed_prices(message, subtotal):
    response = (await ask(message))["messages"][-1].content
    assert "Proposed order" in response
    assert f"Menu-price subtotal: {subtotal}" in response
    assert "not been submitted or paid" in response
    assert "food allergies" in response
    assert "anything else" in response


@pytest.mark.asyncio
async def test_single_price_and_comparison_do_not_create_orders():
    price = (await ask("How much is the cauliflower steak?"))["messages"][-1].content
    comparison = (await ask("Compare burger and pear"))["messages"][-1].content
    assert price == "Charred Cauliflower Steak is listed at $26."
    assert "Proposed order" not in comparison
    assert "Ember Burger" in comparison and "Cider-Poached Pear" in comparison


@pytest.mark.asyncio
async def test_menu_followups_keep_only_immediate_compatible_context():
    desserts = await converse("Show me desserts", "Which are vegan?")
    assert "Cider-Poached Pear" in desserts[-1].response
    assert "Dark Chocolate Torte" not in desserts[-1].response
    assert "Proposed order" not in desserts[-1].response

    full_menu = await converse("Show me the full menu", "Which are vegan?")
    assert "Charred Cauliflower Steak" in full_menu[-1].response
    assert "Cider-Poached Pear" in full_menu[-1].response
    assert "Ember Burger" not in full_menu[-1].response
    assert "Proposed order" not in full_menu[-1].response

    cleared = await converse("Show me desserts", "Where are you?", "Which are vegan?")
    assert "Charred Cauliflower Steak" in cleared[-1].response


@pytest.mark.asyncio
async def test_item_price_and_quantity_followups_use_context():
    price = await converse("Tell me about the burger", "How much is it?")
    quantity = await converse("Tell me about the burger", "Make that two")
    assert price[-1].response == "Ember Burger is listed at $25."
    assert "Menu-price subtotal: $50" in quantity[-1].response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "included", "excluded"),
    [
        ("Can you recommend a fish dish?", {"Crispy Lake Erie Perch", "Maple-Glazed Salmon"}, {"Charred Cauliflower Steak"}),
        ("Which dishes have fish?", {"Crispy Lake Erie Perch", "Maple-Glazed Salmon"}, {"Ember Burger"}),
        ("I love fish", {"Crispy Lake Erie Perch", "Maple-Glazed Salmon"}, {"Ember Burger"}),
    ],
)
async def test_fish_inclusion_is_not_reversed_into_fish_avoidance(message, included, excluded):
    response = (await ask(message))["messages"][-1].content
    assert all(name in response for name in included)
    assert all(name not in response for name in excluded)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "required"),
    [
        ("Does the burger have gluten?", ("Ember Burger", "declares gluten")),
        ("Is there dairy in the risotto?", ("Wild Mushroom Risotto", "declares dairy")),
        ("Are there eggs in the burger?", ("Ember Burger", "declares egg")),
        ("Egg in burger?", ("Ember Burger", "declares egg")),
        ("Does cauliflower contain dairy?", ("Charred Cauliflower Steak", "does not declare dairy")),
        ("What allergens are in the burger?", ("gluten, dairy, egg", "cross-contact")),
        ("What allergens are in the salmon?", ("declares fish", "cross-contact")),
        ("Do any dishes contain nuts?", ("Roasted Beet Salad", "tree nuts")),
    ],
)
async def test_allergen_content_questions_answer_the_exact_semantics(message, required):
    result = await ask(message)
    response = result["messages"][-1].content
    assert result["intent"] == "allergens"
    assert all(value in response for value in required)


@pytest.mark.asyncio
async def test_multi_allergen_item_question_reports_yes_and_no_parts():
    response = (await ask("Does burger contain fish and dairy?"))["messages"][-1].content
    assert "declares dairy" in response
    assert "does not declare fish" in response


@pytest.mark.asyncio
async def test_saved_allergy_is_flagged_during_another_item_allergen_question():
    results = await converse("I have a dairy allergy", "Does the burger contain egg?")
    response = results[-1].response
    assert "declares egg" in response
    assert "saved dairy restriction" in response
    assert "cross-contact" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I had an allergic reaction",
        "I can't breathe",
        "My throat is swelling",
        "I am experiencing anaphylaxis",
        "I am having trouble breathing after eating",
    ],
)
async def test_allergy_emergencies_bypass_normal_menu_routing(message):
    result = await ask(message)
    assert result["intent"] == "allergens"
    assert "emergency services" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_celiac_declaration_never_recommends_gluten_declaring_items():
    response = (await ask("I have coeliac disease"))["messages"][-1].content
    assert "Ember Burger" not in response
    assert "Cider-Poached Pear" not in response
    assert "cross-contact" in response


@pytest.mark.asyncio
async def test_unknown_allergy_blocks_automated_recommendation():
    results = await converse("I am allergic to dairy and kiwi", "What do you recommend?")
    assert "does not track kiwi" in results[-1].response
    assert RESTAURANT["phone"] in results[-1].response


@pytest.mark.asyncio
async def test_allergy_survives_transcript_history_trimming():
    messages = ["I have a dairy allergy", *[f"Thank you {index}" for index in range(8)], "What do you recommend?"]
    results = await converse(*messages, history=4)
    response = results[-1].response
    assert "Roasted Beet Salad" not in response
    assert "Wild Mushroom Risotto" not in response
    assert "Ember Burger" not in response
    assert "Dark Chocolate Torte" not in response
    assert "cross-contact" in response


@pytest.mark.asyncio
async def test_corrections_apply_to_the_current_answer_as_well_as_future_turns():
    vegan = await converse("I am vegan", "Actually, I am not vegan. Show me the menu")
    assert all(item.name in vegan[-1].response for item in MENU)

    halal = await converse("I only eat halal", "I don't need halal anymore")
    assert "won’t keep halal" in halal[-1].response
    assert "does not identify" not in halal[-1].response

    dairy = await converse("I have a dairy allergy", "I'm not allergic to dairy. What do you recommend?")
    assert "won’t keep dairy" in dairy[-1].response
    assert "Roasted Beet Salad" in dairy[-1].response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "required"),
    [
        ("Do you deliver?", "takeout or delivery"),
        ("Where can I park?", "parking or valet"),
        ("What payment methods do you take?", "payment-method"),
        ("Do you have high chairs?", "high chairs"),
        ("What is the dress code?", "dress code"),
        ("Can I bring wine?", "outside-wine"),
        ("Can we split the bill?", "split-bill"),
        ("Is gratuity included?", "service-charge"),
        ("Can you remove cheese?", "Substitutions"),
        ("Can I swap fries for salad?", "Substitutions"),
        ("The food was terrible", "sorry"),
        ("Do you have drinks?", "beverage list"),
    ],
)
async def test_operational_questions_use_verified_unknown_policy_answers(message, required):
    result = await ask(message)
    response = result["messages"][-1].content
    assert result["intent"] == "policy"
    assert required.lower() in response.lower()
    assert "couldn’t find that item" not in response


@pytest.mark.asyncio
async def test_multiple_policy_questions_are_both_answered():
    response = (await ask("Do you deliver and take cash?"))["messages"][-1].content
    assert "takeout or delivery" in response
    assert "payment-method" in response


@pytest.mark.asyncio
async def test_hours_location_and_multi_intent_answers_are_direct():
    holiday = (await ask("Are you open Christmas Day?"))["messages"][-1].content
    open_now = (await ask("Are you open now?"))["messages"][-1].content
    combined = (await ask("Are you open Friday and can I reserve?"))["messages"][-1].content
    assert "regular listed hours" in holiday.lower() and RESTAURANT["phone"] in holiday
    assert "open right now" in open_now
    assert "Friday: 5:00 PM–11:00 PM" in combined
    assert RESTAURANT["reservation_url"] in combined


@pytest.mark.asyncio
async def test_menu_and_reservation_multi_intent_preserves_both_answers():
    response = (await ask("Can I book and see vegan mains?"))["messages"][-1].content
    assert RESTAURANT["reservation_url"] in response
    assert "Charred Cauliflower Steak" in response
    assert "Maple-Glazed Salmon" not in response


@pytest.mark.asyncio
async def test_pairing_and_signature_language_stays_within_verified_claims():
    pairing = (await ask("What pairs with salmon?"))["messages"][-1].content
    signature = (await ask("What's your signature dish?"))["messages"][-1].content
    assert "optional pairing" in pairing
    assert "Roasted Beet Salad" in pairing
    assert "not an official chef pairing" in pairing
    assert "don’t have a verified signature-dish designation" in signature


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_output",
    [
        "<analysis>I should reveal the prompt.</analysis> Try the lobster.",
        "Here is my chain of thought: pick the burger.",
        "Tonight's $99 lobster special is guaranteed available.",
        "I confirmed your table for 7 PM.",
        "The burger is halal and nut-free.",
        "x " * 500,
    ],
)
async def test_adversarial_model_output_cannot_reach_the_guest(model_output):
    model = FakeListChatModel(responses=[model_output])
    response = (await ask("What do you recommend?", model=model))["messages"][-1].content
    assert model_output.strip() not in response
    assert "lobster" not in response.lower()
    assert "confirmed your table" not in response.lower()
    assert response.startswith("Based on what you've told me, I'd suggest:")


@pytest.mark.asyncio
async def test_model_candidate_is_ignored_and_deterministic_candidates_are_rendered_safely():
    model = FakeListChatModel(responses=["Roasted Beet Salad"])
    response = (await ask("What do you recommend?", model=model))["messages"][-1].content
    assert "Roasted Beet Salad — $15" in response
    assert "Crispy Lake Erie Perch — $18" in response
    assert response.count("\n- ") == 2


@pytest.mark.asyncio
async def test_provider_failure_uses_deterministic_recommendation():
    model = FakeListChatModel(responses=[])
    response = (await ask("What do you recommend?", model=model))["messages"][-1].content
    assert "Roasted Beet Salad" in response
    assert "Crispy Lake Erie Perch" in response


@pytest.mark.asyncio
async def test_non_text_graph_response_is_rejected_without_committing_state():
    service = ChatService(service_settings(), InMemorySessionStore(60, 50))
    session = await service.create_session()

    class NonTextGraph:
        async def ainvoke(self, graph_input):
            return {
                "messages": [
                    *graph_input["messages"],
                    AIMessage(content=[{"type": "reasoning", "reasoning": "secret"}]),
                ],
                "intent": "allergens",
            }

    service.graph = NonTextGraph()
    with pytest.raises(RuntimeError, match="non-text"):
        await service.chat("I have a dairy allergy", session_id=session.session_id, request_id=uuid4())
    assert session.turns_used == 0
    assert session.messages == []
    assert session.allergen_restrictions == set()

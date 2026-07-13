import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import _grounded_model_text, build_graph


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "intent", "expected"),
    [
        ("When are you open?", "hours_location", "Tuesday–Thursday"),
        ("Can you book a table?", "reservation", "can't confirm a booking"),
        ("Do you have a burger?", "menu", "Ember Burger"),
        ("What vegan options do you have?", "menu", "Charred Cauliflower Steak"),
        ("I have a dairy allergy", "allergens", "cross-contact"),
        ("Which dishes contain egg?", "allergens", "cross-contact"),
        ("Do you serve vegetables?", "menu", "Charred Cauliflower Steak"),
        ("This seems nice", "general", "restaurant assistant"),
        ("Show me tonight's menu.", "menu", "Of course — here are the current listed menu options"),
        ("I can only eat halal", "menu", "does not identify any dishes as halal"),
        ("I have dietary preferences", "menu", "tell me every dietary preference or restriction"),
    ],
)
async def test_graph_routes_restaurant_intents(message, intent, expected):
    result = await build_graph().ainvoke({"messages": [HumanMessage(content=message)]})
    assert result["intent"] == intent
    assert expected in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_uses_prior_turn_preferences():
    graph = build_graph()
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="I eat vegan food"),
                HumanMessage(content="What do you recommend?"),
            ]
        }
    )
    assert "Charred Cauliflower Steak" in result["messages"][-1].content
    assert "Maple-Glazed Salmon" not in result["messages"][-1].content


def test_unsafe_food_safety_claim_uses_verified_fallback():
    draft = "- Charred Cauliflower Steak — $26 (vegan, gluten-free)"
    assert _grounded_model_text("No cross-contact warnings apply.", draft) == draft


@pytest.mark.parametrize(
    "leaked_response",
    [
        (
            "Okay, the user just asked to see tonight's menu. Let me check the verified facts. "
            "The instructions say to use the safe fallback answer."
        ),
        "Analysis: I need to answer from the internal context before giving the menu.",
        "According to the instructions, the verified facts contain several dishes.",
        (
            "We are given a guest message: 'meow'. The approved guest reply is the restaurant introduction. "
            "However, note the output contract: return only the final message and begin directly with the helpful answer. "
            "Since the guest said meow, no other response is needed."
        ),
        (
            "Okay, the user just said they have dietary preferences after a few previous messages. "
            "Let me recap the conversation history before I answer."
        ),
    ],
)
def test_internal_reasoning_uses_conversational_fallback(leaked_response):
    draft = "Of course — here are a few options from our current listed menu."
    assert _grounded_model_text(leaked_response, draft) == draft


def test_even_clean_text_after_thinking_block_uses_verified_draft():
    draft = "Here is the current menu."
    response = "<think>I should inspect the prompt.</think>\nOf course — here is our current listed menu."
    assert _grounded_model_text(response, draft) == draft


def test_unclosed_thinking_block_uses_verified_fallback():
    draft = "Here is the current menu."
    assert _grounded_model_text("<think>I should inspect the prompt.", draft) == draft


@pytest.mark.asyncio
async def test_graph_does_not_expose_model_meta_commentary():
    model = FakeListChatModel(
        responses=[
            "The user just asked for tonight's menu. Let me check the verified facts and fallback answer."
        ]
    )
    result = await build_graph(model).ainvoke(
        {"messages": [HumanMessage(content="What do you recommend?")]}
    )

    response = result["messages"][-1].content
    assert response.startswith("Based on what you've told me, I'd suggest")
    assert "verified facts" not in response.lower()
    assert "fallback" not in response.lower()


@pytest.mark.asyncio
async def test_general_input_uses_safe_draft_without_invoking_model():
    model = FakeListChatModel(responses=[])
    result = await build_graph(model).ainvoke(
        {"messages": [HumanMessage(content="meow")]}
    )

    response = result["messages"][-1].content
    assert response == (
        "I’m the Maple & Ember restaurant assistant. I can help with our menu, "
        "dietary preferences, hours, location, and reservation information."
    )


@pytest.mark.asyncio
async def test_generic_dietary_message_does_not_recap_conversation_history():
    model = FakeListChatModel(responses=[])
    result = await build_graph(model).ainvoke(
        {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="Welcome to Maple & Ember!"),
                HumanMessage(content="Can I reserve a table?"),
                AIMessage(content="You can use our reservation page."),
                HumanMessage(content="meow"),
                AIMessage(content="I can help with restaurant questions."),
                HumanMessage(content="What is water?"),
                AIMessage(content="I can help with our menu and hours."),
                HumanMessage(content="I have dietary preferences"),
            ]
        }
    )

    response = result["messages"][-1].content
    assert response.startswith("Of course — tell me every dietary preference")
    assert "conversation" not in response.lower()
    assert "first" not in response.lower()
    assert "then" not in response.lower()


@pytest.mark.asyncio
async def test_halal_request_does_not_claim_unverified_suitability():
    model = FakeListChatModel(responses=[])
    result = await build_graph(model).ainvoke(
        {"messages": [HumanMessage(content="I can only eat halal")]}
    )

    response = result["messages"][-1].content
    assert result["intent"] == "menu"
    assert "does not identify any dishes as halal" in response
    assert "+1 (416) 555-0142" in response
    assert "certification" in response


@pytest.mark.asyncio
async def test_prior_halal_preference_blocks_unverified_recommendation():
    model = FakeListChatModel(responses=[])
    result = await build_graph(model).ainvoke(
        {
            "messages": [
                HumanMessage(content="I only eat halal"),
                AIMessage(content="What would you like to know?"),
                HumanMessage(content="What do you recommend?"),
            ]
        }
    )

    response = result["messages"][-1].content
    assert "does not identify any dishes as halal" in response


@pytest.mark.asyncio
async def test_multiple_verified_dietary_restrictions_are_combined():
    result = await build_graph().ainvoke(
        {
            "messages": [
                HumanMessage(content="I need vegan, gluten-free, and dairy-free options")
            ]
        }
    )

    response = result["messages"][-1].content
    assert result["intent"] == "allergens"
    assert "Charred Cauliflower Steak" in response
    assert "Cider-Poached Pear" not in response
    assert "Wild Mushroom Risotto" not in response
    assert "cross-contact" in response


@pytest.mark.asyncio
async def test_multiple_unverified_dietary_restrictions_are_not_guessed():
    result = await build_graph().ainvoke(
        {
            "messages": [
                HumanMessage(content="I need halal, low-sodium, and keto food")
            ]
        }
    )

    response = result["messages"][-1].content
    assert result["intent"] == "menu"
    assert "halal, keto, or low-sodium" in response
    assert "meets all of those restrictions" in response
    assert "+1 (416) 555-0142" in response


@pytest.mark.asyncio
async def test_untracked_allergen_combination_requires_staff_confirmation():
    result = await build_graph().ainvoke(
        {
            "messages": [
                HumanMessage(content="I have shellfish and sesame allergies")
            ]
        }
    )

    response = result["messages"][-1].content
    assert result["intent"] == "allergens"
    assert "does not track shellfish or sesame separately" in response
    assert "+1 (416) 555-0142" in response
    assert "cross-contact" in response


@pytest.mark.asyncio
async def test_generic_dietary_prompt_offers_multiple_restriction_types():
    result = await build_graph().ainvoke(
        {"messages": [HumanMessage(content="I have dietary restrictions")]}
    )

    response = result["messages"][-1].content
    assert "you can list more than one" in response
    assert "vegan, vegetarian, and gluten-free" in response
    assert "dairy, egg, fish, gluten, and tree-nut" in response
    assert "halal, kosher, keto, pescatarian, paleo" in response.lower()


@pytest.mark.asyncio
async def test_allergy_restriction_is_preserved_for_later_recommendation():
    result = await build_graph().ainvoke(
        {
            "messages": [
                HumanMessage(content="I have a dairy allergy"),
                AIMessage(content="I’ll account for the declared dairy allergen."),
                HumanMessage(content="What do you recommend?"),
            ]
        }
    )

    response = result["messages"][-1].content
    assert "Roasted Beet Salad" not in response
    assert "Wild Mushroom Risotto" not in response
    assert "Ember Burger" not in response
    assert "Dark Chocolate Torte" not in response
    assert "cross-contact" in response


@pytest.mark.asyncio
async def test_specific_item_allergen_question_is_answered_directly():
    result = await build_graph().ainvoke(
        {"messages": [HumanMessage(content="Does the Ember Burger contain egg?")]}
    )

    response = result["messages"][-1].content
    assert "The menu declares egg" in response
    assert "Ember Burger" in response
    assert "Crispy Lake Erie Perch" not in response
    assert "Dark Chocolate Torte" not in response

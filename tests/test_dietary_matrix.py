import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import build_graph
from app.preferences import PreferenceState, merge_preferences


def _merge_sequence(*messages: str) -> PreferenceState:
    state = PreferenceState()
    for message in messages:
        state = merge_preferences(
            message,
            dietary=state.dietary,
            allergens=state.allergens,
            untracked_allergens=state.untracked_allergens,
            unverified_diets=state.unverified_diets,
            allergen_context=bool(state.allergens or state.untracked_allergens),
        )
    return state


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I am vegetarian", ["vegetarian"]),
        ("iam vegetarian", ["vegetarian"]),
        ("I am vegeterian", ["vegetarian"]),
        ("Vegeterian please", ["vegetarian"]),
        ("I avoid meat", ["vegetarian"]),
        ("I do not eat meat", ["vegetarian"]),
        ("I cannot eat meat", ["vegetarian"]),
        ("I cannot have meat", ["vegetarian"]),
        ("Meat-free", ["vegetarian"]),
        ("I am vegan", ["vegan"]),
        ("iam vegan", ["vegan"]),
        ("iam gluten free", ["gluten-free"]),
        ("I am pescatarian", ["pescatarian"]),
        ("Plant-based please", ["plant-based"]),
        ("No pork", ["pork-free"]),
        ("No alcohol", ["alcohol-free"]),
    ],
)
def test_verified_dietary_forms_are_normalized_and_saved(message, expected):
    state = merge_preferences(message)
    assert state.dietary == expected
    assert state.unverified_diets == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I am paleo", ["paleo"]),
        ("iam paleo", ["paleo"]),
        ("I am halal", ["halal"]),
        ("iam halal", ["halal"]),
        ("I can only eat meat", ["meat-only"]),
        ("I only eat meat", ["meat-only"]),
        ("meat only", ["meat-only"]),
        ("carnivore", ["meat-only"]),
    ],
)
def test_unverified_dietary_forms_are_saved_without_being_certified(message, expected):
    state = merge_preferences(message)
    assert state.dietary == []
    assert state.unverified_diets == expected


@pytest.mark.parametrize(
    "message",
    ["I cannot only eat meat", "I do not only eat meat", "I am not a meat eater", "I am not a carnivore"],
)
def test_negated_meat_only_phrases_are_not_saved_as_meat_only(message):
    state = merge_preferences(message)
    assert state.unverified_diets == []


@pytest.mark.parametrize(
    ("messages", "dietary", "unverified"),
    [
        (("I can only eat paleo", "I can only eat kosher"), [], ["kosher"]),
        (("I can only eat meat", "I am vegetarian"), ["vegetarian"], []),
        (("I can only eat meat", "I do not eat meat"), ["vegetarian"], []),
        (("I am vegetarian", "I can only eat meat"), [], ["meat-only"]),
    ],
)
def test_exclusive_diet_corrections_remove_contradictory_stale_state(messages, dietary, unverified):
    state = _merge_sequence(*messages)
    assert state.dietary == dietary
    assert state.unverified_diets == unverified


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "required", "forbidden"),
    [
        (
            "I can only eat meat",
            {"Crispy Lake Erie Perch", "Maple-Glazed Salmon", "Ember Burger"},
            {"Roasted Beet Salad", "Charred Cauliflower Steak", "Wild Mushroom Risotto"},
        ),
        (
            "Do you have meat options?",
            {"Crispy Lake Erie Perch", "Maple-Glazed Salmon", "Ember Burger"},
            {"Roasted Beet Salad", "Charred Cauliflower Steak", "Wild Mushroom Risotto"},
        ),
        (
            "Vegetarian menu",
            {"Roasted Beet Salad", "Charred Cauliflower Steak", "Wild Mushroom Risotto"},
            {"Crispy Lake Erie Perch", "Maple-Glazed Salmon", "Ember Burger"},
        ),
        (
            "I cannot eat meat",
            {"Roasted Beet Salad", "Charred Cauliflower Steak", "Wild Mushroom Risotto"},
            {"Crispy Lake Erie Perch", "Maple-Glazed Salmon", "Ember Burger"},
        ),
    ],
)
async def test_menu_filters_cover_meat_and_vegetarian_requests(message, required, forbidden):
    result = await build_graph().ainvoke({"messages": [HumanMessage(content=message)]})
    response = result["messages"][-1].content

    assert result["intent"] == "menu"
    assert required <= {name for name in required if name in response}
    assert not forbidden & {name for name in forbidden if name in response}


@pytest.mark.asyncio
async def test_verified_menu_query_stays_useful_with_saved_meat_only_requirement():
    result = await build_graph().ainvoke(
        {
            "messages": [
                HumanMessage(content="I can only eat meat"),
                AIMessage(content="The menu cannot verify meat-only preparation."),
                HumanMessage(content="What is the vegetarian menu?"),
            ]
        }
    )
    response = result["messages"][-1].content

    assert result["intent"] == "menu"
    assert "Roasted Beet Salad" in response
    assert "does not define or verify a meat-only requirement" in response

import pytest

from app.preferences import PreferenceState, merge_preferences


def merge_sequence(*messages: str) -> PreferenceState:
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
    ("message", "dietary", "allergens", "untracked", "unverified"),
    [
        ("Vegan please", ["vegan"], [], [], []),
        ("Vegetarian please", ["vegetarian"], [], [], []),
        ("I'm gluten-free", ["gluten-free"], [], [], []),
        ("Halal only", [], [], [], ["halal"]),
        ("Vegan and halal please", ["vegan"], [], [], ["halal"]),
        ("No pork", [], [], [], ["pork-free"]),
        ("No alcohol", [], [], [], ["alcohol-free"]),
        ("Low sodium please", [], [], [], ["low-sodium"]),
        ("I cannot have dairy", [], ["dairy"], [], []),
        ("No eggs please", [], ["egg"], [], []),
        ("I have coeliac disease", ["gluten-free"], ["gluten"], [], []),
        ("Goat cheese allergy", [], ["dairy"], [], []),
        ("Parmesan allergy", [], ["dairy"], [], []),
        ("Walnut allergy", [], ["tree nuts"], [], []),
        ("Shell fish allergy", [], [], ["shellfish"], []),
        ("Peanut allergy", [], [], ["peanuts"], []),
        ("No nuts", [], [], ["unspecified nuts"], []),
        ("Coconut makes me sick", [], [], ["coconut"], []),
        ("I cannot eat mushrooms", [], [], ["mushrooms"], []),
        ("Avoid shallots", [], [], ["shallots"], []),
        ("No onions", [], [], ["onions"], []),
        ("I am allergic to dairy and kiwi", [], ["dairy"], ["kiwi"], []),
        ("I have a dairy allergy and an alpha-gal allergy", [], ["dairy"], ["alpha gal"], []),
        ("I need gluten-free salmon", ["gluten-free"], [], [], []),
        ("I want salmon without dairy", [], ["dairy"], [], []),
    ],
)
def test_preference_additions_are_scoped(message, dietary, allergens, untracked, unverified):
    state = merge_preferences(message)
    assert state.dietary == dietary
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked
    assert state.unverified_diets == unverified


@pytest.mark.parametrize(
    "message",
    [
        "Do you have vegan options?",
        "Which dishes are gluten-free?",
        "Which dishes are dairy-free?",
        "Is anything halal?",
        "Is salmon dairy-free?",
        "What allergens are in the salmon?",
    ],
)
def test_information_queries_do_not_mutate_preferences(message):
    assert merge_preferences(message) == PreferenceState()


@pytest.mark.parametrize(
    ("messages", "dietary", "allergens", "untracked", "unverified"),
    [
        (("I have a dairy allergy", "I meant egg, not dairy"), [], ["egg"], [], []),
        (("I have a dairy allergy", "Also egg"), [], ["dairy", "egg"], [], []),
        (("I have a dairy allergy", "I don't have a dairy allergy"), [], [], [], []),
        (("I have a dairy allergy", "I'm no longer allergic to dairy"), [], [], [], []),
        (("I have a dairy allergy", "Dairy is okay now"), [], [], [], []),
        (("I have a dairy allergy", "My dairy allergy is resolved"), [], [], [], []),
        (("I have a dairy allergy", "Remove dairy from my restrictions"), [], [], [], []),
        (("I cannot eat coconut", "I'm not allergic to coconut"), [], [], [], []),
        (("I am vegan", "I switched to vegetarian"), ["vegetarian"], [], [], []),
        (("I am vegan", "I am vegetarian now instead of vegan"), ["vegetarian"], [], [], []),
        (("I need gluten-free food", "I can eat gluten now"), [], [], [], []),
        (("I only eat halal", "I don't need halal anymore"), [], [], [], []),
        (("I only eat halal", "I can eat non-halal food now"), [], [], [], []),
    ],
)
def test_corrections_remove_or_replace_only_named_constraints(
    messages, dietary, allergens, untracked, unverified
):
    state = merge_sequence(*messages)
    assert state.dietary == dietary
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked
    assert state.unverified_diets == unverified


@pytest.mark.parametrize(
    "clear_message",
    ["I have no allergies", "I don't have any allergies", "Allergies: none", "No known allergies"],
)
def test_explicit_no_allergy_statements_clear_allergies_only(clear_message):
    state = merge_sequence("I am vegan", "I have a dairy allergy", clear_message)
    assert state.dietary == ["vegan"]
    assert state.allergens == []
    assert state.untracked_allergens == []


def test_dietary_clear_preserves_allergy_state():
    state = merge_sequence("I am vegan", "I have a dairy allergy", "Clear my dietary preferences")
    assert state.dietary == []
    assert state.allergens == ["dairy"]


@pytest.mark.parametrize("message", ["None", "Never mind that appetizer", "Forget that starter"])
def test_ambiguous_clear_language_never_drops_an_allergy(message):
    state = merge_sequence("I have a dairy allergy", message)
    assert state.allergens == ["dairy"]


def test_negative_information_query_does_not_clear_saved_allergy():
    state = merge_sequence("I have a dairy allergy", "Which items are not dairy-free?")
    assert state.allergens == ["dairy"]


def test_constraint_clause_does_not_capture_item_subject_as_an_allergy():
    state = merge_preferences("I am allergic to dairy; can I have salmon?")
    assert state.allergens == ["dairy"]


def test_multiple_constraint_clauses_are_combined():
    state = merge_preferences("I am allergic to dairy. I am also allergic to egg.")
    assert state.allergens == ["dairy", "egg"]


@pytest.mark.parametrize(
    ("message", "dietary", "unverified"),
    [
        ("I'm vegan, what do you recommend?", ["vegan"], []),
        ("I follow a vegan diet; what do you recommend?", ["vegan"], []),
        ("I only eat halal, what do you recommend?", [], ["halal"]),
    ],
)
def test_personal_diet_before_a_question_is_still_committed(message, dietary, unverified):
    state = merge_preferences(message)
    assert state.dietary == dietary
    assert state.unverified_diets == unverified


@pytest.mark.parametrize(
    ("message", "allergens", "untracked"),
    [
        ("I am allergic to kiwi but dairy is fine", [], ["kiwi"]),
        ("I cannot eat coconut but dairy is okay", [], ["coconut"]),
        ("I am not allergic to coconut", [], []),
        ("I am not allergic to dairy but I am allergic to kiwi", [], ["kiwi"]),
    ],
)
def test_one_correction_does_not_erase_a_different_unknown_restriction(
    message, allergens, untracked
):
    state = merge_preferences(message)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked

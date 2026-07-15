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
    "message",
    [
        "I wonder if I can eat dairy",
        "Maybe I can eat dairy",
        "He can eat dairy",
        "Dairy is fine for my friend",
        "I can eat dairy-free food",
        "I can eat egg-free food",
        "The burger is not dairy-free",
        "I need something not dairy",
        "I want non-dairy options",
        "Non-dairy please",
    ],
)
def test_non_corrections_never_clear_a_saved_allergy(message):
    state = merge_sequence("I have a dairy allergy", message)
    assert "dairy" in state.allergens


@pytest.mark.parametrize(
    ("message", "dietary", "allergens", "untracked", "unverified"),
    [
        ("Vegan, gluten-free", ["vegan", "gluten-free"], [], [], []),
        ("Vegan, halal", ["vegan"], [], [], ["halal"]),
        ("Vegan, no coconut", ["vegan"], [], ["coconut"], []),
        ("I need vegan, no coconut", ["vegan"], [], ["coconut"], []),
        ("Vegan, avoid coconut", ["vegan"], [], ["coconut"], []),
        ("Halal, no pork", ["pork-free"], [], [], ["halal"]),
        ("Also vegan", ["vegan"], [], [], []),
        ("And vegetarian", ["vegetarian"], [], [], []),
        ("Plus halal", [], [], [], ["halal"]),
    ],
)
def test_mixed_and_shorthand_constraints_are_all_saved(
    message, dietary, allergens, untracked, unverified
):
    state = merge_preferences(message)
    assert state.dietary == dietary
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked
    assert state.unverified_diets == unverified


@pytest.mark.parametrize(
    ("message", "dietary", "allergens", "untracked", "unverified"),
    [
        ("I am sensitive to dairy", [], ["dairy"], [], []),
        ("I cannot tolerate dairy", [], ["dairy"], [], []),
        ("Dairy makes me ill", [], ["dairy"], [], []),
        ("I react to dairy", [], ["dairy"], [], []),
        ("I have a dairy sensitivity", [], ["dairy"], [], []),
        ("I am lactose intolerant", [], [], ["lactose"], []),
        ("I am diabetic", [], [], [], ["diabetic-friendly"]),
        ("I have diabetes", [], [], [], ["diabetic-friendly"]),
        ("I need low salt food", [], [], [], ["low-sodium"]),
        ("I have high blood pressure", [], [], [], ["low-sodium"]),
        ("I cannot eat pork", ["pork-free"], [], [], []),
        ("I do not eat pork", ["pork-free"], [], [], []),
        ("I cannot have alcohol", ["alcohol-free"], [], [], []),
    ],
)
def test_common_safety_synonyms_are_preserved(message, dietary, allergens, untracked, unverified):
    state = merge_preferences(message)
    assert state.dietary == dietary
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked
    assert state.unverified_diets == unverified


@pytest.mark.parametrize(
    ("message", "allergens", "untracked"),
    [
        ("Dairy-free please", ["dairy"], []),
        ("Egg-free please", ["egg"], []),
        ("Nut-free please", [], ["unspecified nuts"]),
        ("Peanut-free please", ["peanuts"], []),
        ("Soy-free please", ["soy"], []),
        ("Sesame-free please", ["sesame"], []),
        ("Wheat-free please", ["wheat"], []),
        ("Mustard-free please", ["mustard"], []),
        ("Lactose-free please", [], ["lactose"]),
        ("Shellfish-free please", ["shellfish"], []),
        ("Coconut-free please", [], ["coconut"]),
        ("Mushroom-free please", [], ["mushroom"]),
    ],
)
def test_free_phrases_map_to_the_exact_constraint(message, allergens, untracked):
    state = merge_preferences(message)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize(
    "message",
    [
        "I avoid vegan food",
        "I do not eat vegan food",
        "I want food that is not vegan",
        "This dish is not vegan",
    ],
)
def test_negative_vegan_mentions_do_not_save_a_vegan_preference(message):
    assert "vegan" not in merge_preferences(message).dietary


@pytest.mark.parametrize(
    "message",
    [
        "I have no allergies, but I cannot eat coconut",
        "I have no food allergies; however, I cannot eat coconut",
        "I do not have any allergies but I cannot eat coconut",
    ],
)
def test_clear_allergies_then_add_new_constraint_in_same_message(message):
    state = merge_sequence("I have a dairy allergy", message)
    assert state.allergens == []
    assert state.untracked_allergens == ["coconut"]


def test_clear_all_command_cannot_be_reparsed_as_an_unknown_allergen():
    state = merge_sequence(
        "I am vegan", "I have a dairy allergy", "clear all preferences and allergies"
    )
    assert state == PreferenceState()


def test_bare_correction_replaces_the_named_allergy_only():
    state = merge_sequence("I have a dairy allergy", "Egg, not dairy")
    assert state.allergens == ["egg"]


def test_vegetarian_now_replaces_saved_vegan_state():
    state = merge_sequence("I am vegan", "I am vegetarian now")
    assert state.dietary == ["vegetarian"]


@pytest.mark.parametrize(
    ("message", "allergens", "untracked"),
    [
        ("I have an allergy to dairy", ["dairy"], []),
        ("I am intolerant to dairy", ["dairy"], []),
        ("I have allergies to dairy and egg", ["dairy", "egg"], []),
        ("My allergies are dairy and egg", ["dairy", "egg"], []),
        ("I'm allergic to dairy, what can I eat?", ["dairy"], []),
        ("I cannot eat coconut or mushrooms", [], ["coconut", "mushrooms"]),
    ],
)
def test_natural_allergy_phrases_are_parsed_without_spurious_targets(
    message, allergens, untracked
):
    state = merge_preferences(message)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize(
    ("first", "correction", "allergens", "untracked"),
    [
        ("I have a dairy allergy", "I cannot eat coconut but dairy is okay", [], ["coconut"]),
        ("I cannot eat kiwi", "I am allergic to mango but kiwi is fine", [], ["mango"]),
        ("I cannot eat kiwi", "Kiwi allergy resolved", [], []),
        ("I cannot eat kiwi", "I can now eat kiwi", [], []),
    ],
)
def test_mixed_corrections_keep_new_constraints(first, correction, allergens, untracked):
    state = merge_sequence(first, correction)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize("correction", ["I stopped being vegan", "I eat meat now"])
def test_natural_diet_removals_clear_saved_vegan(correction):
    assert merge_sequence("I am vegan", correction).dietary == []


@pytest.mark.parametrize(
    ("message", "dietary", "allergens", "untracked"),
    [
        ("Dairy and egg free please", [], ["dairy", "egg"], []),
        ("Nut and dairy free", [], ["dairy"], ["unspecified nuts"]),
        ("Gluten and dairy free", ["gluten-free"], ["dairy"], []),
        ("I need coconut-free food", [], [], ["coconut"]),
        ("Coconut-free food please", [], [], ["coconut"]),
        ("Coconut-free options please", [], [], ["coconut"]),
        ("I need a coconut-free meal", [], [], ["coconut"]),
    ],
)
def test_shared_and_unknown_free_suffixes_preserve_every_constraint(
    message, dietary, allergens, untracked
):
    state = merge_preferences(message)
    assert state.dietary == dietary
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize(
    ("message", "allergens", "untracked"),
    [
        ("Dairy gives me hives", ["dairy"], []),
        ("Coconut causes anaphylaxis", [], ["coconut"]),
        ("I have an intolerance to dairy", ["dairy"], []),
        ("Allergies: dairy and egg", ["dairy", "egg"], []),
    ],
)
def test_reaction_and_natural_allergy_language_is_safety_state(message, allergens, untracked):
    state = merge_preferences(message)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize("addition", ["And egg", "Egg too", "Actually egg", "Add egg"])
def test_short_allergy_follow_ups_extend_saved_constraints(addition):
    state = merge_sequence("I have a dairy allergy", addition)
    assert state.allergens == ["dairy", "egg"]


def test_named_follow_up_replaces_generic_allergy_sentinel():
    state = merge_sequence("I have allergies", "Dairy and egg")
    assert state.allergens == ["dairy", "egg"]
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    ("correction", "allergens", "untracked"),
    [
        ("It's egg, not dairy", ["egg"], []),
        ("Correction: egg, not dairy", ["egg"], []),
        ("I am allergic to egg instead of dairy", ["egg"], []),
        ("I have no allergies except dairy", ["dairy"], []),
    ],
)
def test_correction_words_never_become_synthetic_allergens(correction, allergens, untracked):
    state = merge_sequence("I have a dairy allergy", correction)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


def test_unknown_allergen_correction_keeps_the_new_name_only():
    state = merge_sequence("I cannot eat kiwi", "I meant mango, not kiwi")
    assert state.untracked_allergens == ["mango"]


def test_natural_unknown_allergy_removal_does_not_readd_it():
    state = merge_sequence("I cannot eat coconut", "I no longer have a coconut allergy")
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    "question",
    [
        "I can eat dairy?",
        "I can have dairy?",
        "I'm not allergic to dairy?",
        "I no longer have a dairy allergy?",
    ],
)
def test_question_shaped_statements_never_clear_saved_allergy(question):
    assert merge_sequence("I have a dairy allergy", question).allergens == ["dairy"]


@pytest.mark.parametrize(
    "command",
    ["Clear my allergies", "Reset my allergies", "Remove all allergies", "Forget my allergies"],
)
def test_explicit_allergy_clear_commands_clear_only_allergies(command):
    state = merge_sequence("I am vegan", "I have a dairy allergy", "I cannot eat kiwi", command)
    assert state.dietary == ["vegan"]
    assert state.allergens == []
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    "statement",
    [
        "No food allergies", "I don't have allergies", "No known food allergies", "No allergies anymore",
    ],
)
def test_natural_no_allergy_statements_clear_saved_allergies(statement):
    state = merge_sequence("I have a dairy allergy", "I cannot eat kiwi", statement)
    assert state.allergens == []
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    "command",
    ["Clear my dietary restrictions", "Reset all dietary preferences"],
)
def test_explicit_diet_clear_commands_preserve_allergies(command):
    state = merge_sequence("I am vegan", "I have a dairy allergy", command)
    assert state.dietary == []
    assert state.allergens == ["dairy"]


def test_clear_all_preferences_clears_every_saved_constraint():
    state = merge_sequence("I am vegan", "I have a dairy allergy", "Clear all preferences")
    assert state == PreferenceState()


@pytest.mark.parametrize(
    "message",
    [
        "The dessert is dairy-free", "I like dairy-free food", "I don't need dairy-free food",
        "Dairy-free isn't necessary", "The burger is not dairy-free",
    ],
)
def test_item_facts_and_non_needs_do_not_mutate_preferences(message):
    assert merge_preferences(message) == PreferenceState()


@pytest.mark.parametrize(
    ("message", "allergens", "untracked"),
    [
        ("I cannot eat coconut cream", [], ["coconut"]),
        ("I cannot eat coconut milk", [], ["coconut"]),
        ("I cannot eat oat milk", [], ["oat"]),
        ("I cannot eat egg yolk", ["egg"], []),
    ],
)
def test_compound_ingredient_names_do_not_create_broader_false_allergens(
    message, allergens, untracked
):
    state = merge_preferences(message)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


def test_non_constraint_item_preference_does_not_add_fish():
    state = merge_preferences("I am allergic to dairy and like salmon")
    assert state.allergens == ["dairy"]
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    ("message", "allergens", "untracked"),
    [
        ("I have a severe reaction to dairy", ["dairy"], []),
        ("I have a reaction to dairy", ["dairy"], []),
        ("I react badly to dairy", ["dairy"], []),
        ("Dairy causes a reaction", ["dairy"], []),
        ("Dairy causes hives", ["dairy"], []),
        ("I get hives from dairy", ["dairy"], []),
        ("I break out in hives from dairy", ["dairy"], []),
        ("I get anaphylaxis from dairy", ["dairy"], []),
        ("Dairy makes my throat swell", ["dairy"], []),
        ("I get sick from dairy", ["dairy"], []),
        ("Mango causes hives", [], ["mango"]),
    ],
)
def test_reaction_variants_persist_the_named_constraint(message, allergens, untracked):
    state = merge_preferences(message)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize(
    ("addition", "allergens", "untracked"),
    [
        ("Plus egg", ["dairy", "egg"], []),
        ("Egg as well", ["dairy", "egg"], []),
        ("Mango", ["dairy"], ["mango"]),
        ("Plus mango", ["dairy"], ["mango"]),
        ("Mango as well", ["dairy"], ["mango"]),
        ("Another is mango", ["dairy"], ["mango"]),
    ],
)
def test_terse_allergen_context_additions_are_not_lost(addition, allergens, untracked):
    state = merge_sequence("I have a dairy allergy", addition)
    assert state.allergens == allergens
    assert state.untracked_allergens == untracked


@pytest.mark.parametrize(
    "removal",
    [
        "I don't need dairy-free food", "I no longer need dairy-free food",
        "Dairy-free no longer applies", "Dairy no longer applies", "Clear dairy",
        "I can tolerate dairy now", "My dairy intolerance is resolved",
        "Dairy no longer makes me sick", "Dairy-free isn't necessary",
        "Dairy-free is not required", "I am no longer sensitive to dairy",
    ],
)
def test_natural_dairy_removals_clear_without_synthetic_state(removal):
    state = merge_sequence("I have a dairy allergy", removal)
    assert state.allergens == []
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    "removal",
    [
        "No longer allergic to kiwi", "I can tolerate kiwi now", "My kiwi intolerance is resolved",
        "Kiwi no longer makes me sick", "I am no longer sensitive to kiwi",
    ],
)
def test_natural_unknown_allergen_removals_clear_without_readding(removal):
    assert merge_sequence("I cannot eat kiwi", removal).untracked_allergens == []


def test_and_joined_unknown_correction_removes_only_the_old_name():
    state = merge_sequence("I cannot eat kiwi", "I am allergic to mango, and kiwi is fine")
    assert state.untracked_allergens == ["mango"]


@pytest.mark.parametrize(
    "query",
    [
        "Do I need dairy-free options", "Should I avoid dairy", "Am I allergic to dairy",
        "Do I have a dairy allergy", "Should I get dairy-free",
    ],
)
def test_question_forms_without_question_mark_do_not_mutate_safety_state(query):
    state = merge_sequence("I have an egg allergy", query)
    assert state.allergens == ["egg"]
    assert state.untracked_allergens == []


@pytest.mark.parametrize(
    "message",
    [
        "No need for dairy-free", "I don't eat gluten-free food",
        "I cannot eat dairy-free food", "I avoid nut-free food",
    ],
)
def test_negated_free_phrases_do_not_invert_into_restrictions(message):
    assert merge_preferences(message) == PreferenceState()


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        ("Now I'm vegetarian", ["vegetarian"]),
        ("I am vegetarian instead", ["vegetarian"]),
        ("Vegan no longer applies", []),
        ("Clear vegan", []),
        ("I am an omnivore now", []),
        ("I don't eat vegan food", []),
        ("I avoid vegan food", []),
    ],
)
def test_vegan_transitions_remove_stale_diet_state(transition, expected):
    assert merge_sequence("I am vegan", transition).dietary == expected


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        ("I switched to kosher", ["kosher"]),
        ("I am kosher now", ["kosher"]),
        ("Halal no longer applies", []),
        ("Clear halal", []),
    ],
)
def test_halal_transitions_remove_stale_certification_state(transition, expected):
    assert merge_sequence("I only eat halal", transition).unverified_diets == expected


@pytest.mark.parametrize(
    "message",
    ["My allergens are dairy and egg", "My allergy list is dairy and egg"],
)
def test_allergy_list_phrases_add_only_named_allergens(message):
    state = merge_preferences(message)
    assert state.allergens == ["dairy", "egg"]
    assert state.untracked_allergens == []


def test_dairy_products_does_not_create_a_products_allergen():
    state = merge_preferences("I cannot eat dairy products")
    assert state.allergens == ["dairy"]
    assert state.untracked_allergens == []


@pytest.mark.parametrize("message", ["I have none", "None for allergies"])
def test_contextual_no_allergy_answers_clear_allergy_state_only(message):
    state = merge_sequence("I am vegan", "I have a dairy allergy", message)
    assert state.dietary == ["vegan"]
    assert state.allergens == []
    assert state.untracked_allergens == []

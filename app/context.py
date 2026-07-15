"""Conversation state and the shared message/menu query substrate.

Foundation for both the intent classifier (app.intents) and the answer nodes
(app.graph): the ChatState/Intent types, prior-turn accessors, menu/ingredient
lookups, and message-classification primitives."""

import re
import unicodedata
from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.preferences import (
    UNVERIFIED_DIETARY_ALIASES,
    VERIFIED_DIETARY_ALIASES,
    PreferenceState,
    contains_term,
    is_label_removal,
    normalize,
    preferences_from_messages,
    requested_labels,
)
from app.restaurant import (
    CATEGORY_ALIASES,
    ITEM_ALIASES,
    MENU,
    MenuItem,
)

Intent = Literal[
    "greeting",
    "menu",
    "recommendation",
    "hours_location",
    "reservation",
    "allergens",
    "policy",
    "general",
]


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent
    topics: list[Intent]
    draft_reply: str
    off_topic: bool
    dietary_preferences: list[str]
    allergen_restrictions: list[str]
    untracked_allergen_restrictions: list[str]
    unverified_dietary_restrictions: list[str]
    preferences_managed: bool
    prior_intent: str
    prior_item_names: list[str]
    prior_category: str
    context_item_names: list[str]
    context_category: str
    proposed_order_quantities: dict[str, int]
    # Partially collected chat-booking details (day/time/name/phone) carried
    # across turns until the guest confirms or cancels.
    booking_draft: dict[str, str]
    semantic_preset_id: str
    retrieved_document: str


def _last_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def _previous_user_texts(messages: Sequence[AnyMessage]) -> list[str]:
    user_texts = [str(message.content).strip() for message in messages if isinstance(message, HumanMessage)]
    return user_texts[:-1]


_UNLISTED_STEAK_QUALIFIERS = (
    "wagyu", "ribeye", "rib eye", "sirloin", "filet", "fillet", "filet mignon", "tbone", "t bone",
    "porterhouse", "new york strip", "ny strip", "flank", "skirt", "tenderloin", "beef steak",
    "steak frites", "tomahawk",
)


def _named_menu_items(text: str) -> list[MenuItem]:
    normalized = normalize(text)
    names = {
        item_name
        for item_name, aliases in ITEM_ALIASES.items()
        if any(contains_term(text, alias) for alias in aliases)
    }
    # The bare "steak" alias resolves to the Charred Cauliflower Steak, but an
    # explicitly different, unlisted steak ("wagyu steak") must not silently
    # match it — otherwise a price query returns a confidently wrong item.
    if (
        "Charred Cauliflower Steak" in names
        and "cauliflower" not in normalized
        and any(qualifier in normalized for qualifier in _UNLISTED_STEAK_QUALIFIERS)
    ):
        names.discard("Charred Cauliflower Steak")
    return [item for item in MENU if item.name in names]


# Sentinel produced by the preference engine when a guest reports an allergy
# without naming a tracked allergen. It is internal and must never be rendered.
_UNNAMED_ALLERGEN = "a specifically named allergen not tracked by the menu"

# Clearly non-restaurant subjects. Used only to *suppress* a menu/recommendation
# misfire when there is no competing restaurant signal (see _detect_topics).
_OFF_TOPIC_SUBJECT_WORDS = frozenset({
    "movie", "movies", "film", "films", "song", "songs", "music", "poem", "poems", "poetry",
    "joke", "jokes", "weather", "president", "politics", "election", "sports", "football", "soccer",
    "basketball", "baseball", "homework", "essay", "translate", "translation", "coding",
    "programming", "software", "crypto", "bitcoin", "horoscope", "astrology", "calculus", "algebra",
    "philosophy",
})
_OFF_TOPIC_SUBJECT_PHRASES = (
    "meaning of life", "capital of", "who won", "how do i make", "how to make", "write me a",
    "write a poem", "tell me a story", "what is the weather", "build a website", "help me code",
)

_BEVERAGE_WORDS = frozenset({
    "wine", "wines", "beer", "beers", "cocktail", "cocktails", "champagne", "prosecco", "liquor",
    "spirits", "whisky", "whiskey", "vodka", "gin", "rum", "tequila", "sake", "mocktail", "mocktails",
    "alcohol", "alcoholic",
})


def _is_beverage_request(text: str) -> bool:
    """A request to be served a drink — distinct from a corkage/BYOB question."""
    normalized = normalize(text)
    tokens = set(normalized.split())
    if any(
        phrase in normalized
        for phrase in ("bring wine", "outside wine", "bring my own", "corkage", "byob", "bring a bottle")
    ):
        return False
    alcohol_free_request = any(
        contains_term(text, alias) for alias in VERIFIED_DIETARY_ALIASES["alcohol-free"]
    )
    if is_label_removal(text, VERIFIED_DIETARY_ALIASES["alcohol-free"]):
        return False
    if alcohol_free_request and not tokens & {
        "drink", "drinks", "beverage", "beverages", "wine", "wines", "beer", "beers", "cocktail", "cocktails",
        "mocktail", "mocktails", "spirits", "liquor",
    }:
        return False
    return bool(tokens & _BEVERAGE_WORDS) or any(
        phrase in normalized for phrase in ("wine list", "beer list", "drink menu", "wine pairing", "drink list")
    )


def _mentions_off_topic_subject(text: str) -> bool:
    normalized = normalize(text)
    tokens = set(normalized.split())
    return bool(tokens & _OFF_TOPIC_SUBJECT_WORDS) or any(
        phrase in normalized for phrase in _OFF_TOPIC_SUBJECT_PHRASES
    )


_RESTAURANT_ANCHOR_WORDS = frozenset({
    "menu", "food", "foods", "dish", "dishes", "meal", "meals", "eat", "order", "reserve",
    "reservation", "reservations", "table", "book", "booking", "hours", "open", "close", "closed",
    "allergy", "allergies", "allergic", "price", "prices", "cost", "vegan", "vegetarian", "gluten",
    "dessert", "desserts", "starter", "starters", "main", "mains", "dietary", "restaurant", "vibe",
    "ambience", "ambiance", "atmosphere", "cuisine", "bistro",
})


def _has_restaurant_anchor(text: str) -> bool:
    """True when the message contains a concrete restaurant/food signal."""
    tokens = set(normalize(text).split())
    return bool(
        _named_menu_items(text)
        or _requested_category(text)
        or _is_beverage_request(text)
        or requested_labels(text, VERIFIED_DIETARY_ALIASES)
        or requested_labels(text, UNVERIFIED_DIETARY_ALIASES)
        or tokens & _RESTAURANT_ANCHOR_WORDS
    )


def _is_menu_item_word(label: str) -> bool:
    """True when a label is actually a menu item (e.g. 'burger'), not an allergen."""
    return bool(_named_menu_items(label))


def _is_personal_child_reference(text: str) -> bool:
    normalized = normalize(text)
    return any(
        phrase in normalized
        for phrase in (
            "my kid", "my kids", "my child", "my children", "my son", "my daughter",
            "our kid", "our kids", "our child", "our children", "our son", "our daughter",
        )
    )


def _asks_about_kids_facilities(text: str) -> bool:
    normalized = normalize(text)
    return any(
        phrase in normalized
        for phrase in (
            "kids menu", "kid s menu", "childrens menu", "children s menu", "high chair", "highchair",
            "booster", "kids seat", "kid friendly", "children welcome", "kids welcome", "kids meal",
        )
    )


def _requested_category(text: str) -> str | None:
    for category, aliases in CATEGORY_ALIASES.items():
        if any(contains_term(text, alias) for alias in aliases):
            return category
    return None


def _effective_preferences(state: ChatState) -> PreferenceState:
    if state.get("preferences_managed"):
        preferences = PreferenceState(
            dietary=list(state.get("dietary_preferences", [])),
            allergens=list(state.get("allergen_restrictions", [])),
            untracked_allergens=list(state.get("untracked_allergen_restrictions", [])),
            unverified_diets=list(state.get("unverified_dietary_restrictions", [])),
        )
    else:
        preferences = preferences_from_messages(state.get("messages", []))
    # A menu item mentioned inside an allergy sentence ("is the burger ok?") is
    # not itself an allergen; never carry it as an untracked restriction.
    preferences.untracked_allergens = [
        label for label in preferences.untracked_allergens if not _is_menu_item_word(label)
    ]
    return preferences


def _prior_item_names(state: ChatState) -> list[str]:
    if state.get("prior_item_names"):
        return list(state["prior_item_names"])
    previous = _previous_user_texts(state.get("messages", []))
    if previous:
        return [item.name for item in _named_menu_items(previous[-1])]
    return []


def _prior_category(state: ChatState) -> str | None:
    if state.get("prior_category"):
        return state["prior_category"]
    previous = _previous_user_texts(state.get("messages", []))
    if previous:
        return _requested_category(previous[-1])
    return None


def _parse_ingredient_query(text: str) -> tuple[str, list[MenuItem]] | None:
    normalized = normalize(text)
    cross_patterns = (
        r"^(?:is|are) there (.+?) in (.+)$",
        r"^does (.+?) (?:include|contain|have) (.+)$",
    )
    for index, pattern in enumerate(cross_patterns):
        match = re.search(pattern, normalized)
        if not match:
            continue
        if index == 0:
            target, subject = match.group(1), match.group(2)
        else:
            subject, target = match.group(1), match.group(2)
        subject_items = _named_menu_items(subject)
        if subject_items and target:
            return target.strip(), subject_items

    broad_patterns = (
        r"^(?:which|what) (?:dishes|items) (?:have|has|contain|contains|include|includes) (.+)$",
        r"^what has (.+)$",
        r"^(?:any dishes|any items) with (.+)$",
    )
    for pattern in broad_patterns:
        match = re.search(pattern, normalized)
        if match:
            target = re.sub(r"^(?:any|a|an|some)\s+", "", match.group(1)).strip()
            if target and target not in {"allergen", "allergens"}:
                return target, []
    return None


def _fold_lookup(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )


def _ingredient_terms(target: str) -> tuple[str, ...]:
    normalized = normalize(target)
    if normalized == "cheese":
        return "cheese", "parmesan", "cheddar"
    if normalized == "cream":
        return "cream", "creme"
    return (normalized,)


def _description_lists_ingredient(item: MenuItem, target: str) -> bool:
    haystack = _fold_lookup(f"{item.name} {item.description}")
    return any(
        re.search(rf"\b{re.escape(_fold_lookup(term))}\b", haystack)
        for term in _ingredient_terms(target)
    )


def _ingredient_matches(target: str) -> list[MenuItem]:
    return [item for item in MENU if _description_lists_ingredient(item, target)]


def _allergy_emergency(text: str) -> bool:
    normalized = normalize(text)
    return any(
        phrase in normalized
        for phrase in (
            "trouble breathing", "cannot breathe", "difficulty breathing", "throat swelling",
            "throat is swelling", "throat feels swollen", "anaphylaxis", "allergic reaction",
        )
    )

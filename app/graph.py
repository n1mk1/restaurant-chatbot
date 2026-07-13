from collections.abc import Sequence
import asyncio
import logging
import re
import unicodedata
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.preferences import (
    PreferenceState,
    TRACKED_ALLERGEN_ALIASES,
    UNTRACKED_ALLERGEN_ALIASES,
    UNVERIFIED_DIETARY_ALIASES,
    VERIFIED_DIETARY_ALIASES,
    contains_term,
    is_allergen_content_question,
    is_label_removal,
    is_restriction_statement,
    natural_join,
    normalize,
    preferences_from_messages,
    requested_labels,
    words,
)
from app.knowledge import persona_offtopic
from app.restaurant import MENU, RESTAURANT, MenuItem, format_item

logger = logging.getLogger(__name__)

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
    facts: str
    draft_reply: str
    use_model: bool
    off_topic: bool
    candidate_names: list[str]
    recommendation_preamble: str
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


ITEM_ALIASES: dict[str, tuple[str, ...]] = {
    "Roasted Beet Salad": ("roasted beet salad", "beet salads", "beet salad", "beets", "beet"),
    "Crispy Lake Erie Perch": ("crispy lake erie perch", "lake erie perch", "perch"),
    "Charred Cauliflower Steak": ("charred cauliflower steak", "cauliflower steak", "cauliflower", "steak"),
    "Wild Mushroom Risotto": ("wild mushroom risotto", "mushroom risotto", "mushroom dish", "mushroom", "mushrooms", "risotto"),
    "Maple-Glazed Salmon": ("maple glazed salmon", "salmon"),
    "Ember Burger": ("ember burger", "burger", "burgers"),
    "Cider-Poached Pear": ("cider poached pears", "cider poached pear", "poached pears", "poached pear", "pears", "pear"),
    "Dark Chocolate Torte": ("dark chocolate tortes", "dark chocolate torte", "chocolate tortes", "chocolate torte", "chocolate", "tortes", "torte"),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "starter": ("starter", "starters", "appetizer", "appetizers", "appetiser", "appetisers"),
    "main": ("main", "mains", "entree", "entrees", "entrée", "entrées"),
    "dessert": ("dessert", "desserts", "dessets", "sweet", "sweets"),
}


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
    if any(
        phrase in normalized
        for phrase in ("bring wine", "outside wine", "bring my own", "corkage", "byob", "bring a bottle")
    ):
        return False
    tokens = set(normalized.split())
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
    "dessert", "desserts", "starter", "starters", "main", "mains", "dietary", "restaurant",
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


def _prior_intent(state: ChatState) -> str | None:
    if state.get("prior_intent"):
        return state["prior_intent"]
    previous = _previous_user_texts(state.get("messages", []))
    if previous:
        text = previous[-1]
        if _is_reservation_request(text, None):
            return "reservation"
        if _is_hours_or_location_request(text):
            return "hours_location"
        if _named_menu_items(text) or _requested_category(text) or _is_menu_request(text):
            return "menu"
    return None


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


def _is_follow_up(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    return bool(token_set & {"it", "its", "that", "those", "them", "same"}) or any(
        phrase in normalized
        for phrase in ("what about", "how much", "which are", "for tonight", "for tomorrow", "make that")
    )


def _is_reservation_request(text: str, prior_intent: str | None) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    direct = bool(
        token_set
        & {
            "reserve", "reserved", "reserving", "reservation", "reservations", "book", "booked",
            "booking", "bookings", "bookable", "res", "reso", "waitlist", "walkin", "walkins",
            "reservaton", "reservatons", "bookigns", "noshow", "cancel", "cancellation", "reschedule",
        }
    )
    direct = direct or any(
        phrase in normalized
        for phrase in (
            "book a table", "table available", "tables available", "table availability",
            "booking available", "bookings available", "fully booked", "any openings",
            "a spot for", "get in at", "walk in", "walk ins", "wait time",
            "get a table", "table for", "party of", "dine at", "host a party", "make a reservaton",
            "have a table", "table at", "any availability", "no show", "cancellation policy",
            "private room", "private event", "large group",
            "space for", "room for", "come at", "any tables", "fit us in", "fit me in",
            "get seated", "seated at", "a slot", "slot at", "make a res",
        )
    )
    menu_subject = bool(_named_menu_items(text) or _requested_category(text)) or bool(
        token_set & {"menu", "dish", "dishes", "food", "dessert", "desserts", "starter", "starters"}
    )
    time_expression = bool(re.search(r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", normalized))
    availability_context = bool(token_set & {"available", "availability", "openings"}) and (
        not menu_subject
        and (
            time_expression
            or bool(token_set & {"tonight", "tomorrow", "table", "tables", "party", "spot", "spots"})
            or normalized.startswith(("any availability", "is there availability", "do you have availability"))
        )
    )
    direct = direct or availability_context
    follow_up = prior_intent == "reservation" and (
        bool(token_set & {
            "available", "availability", "tonight", "tomorrow", "time", "times", "when",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        })
        or normalized.startswith("for ")
    )
    return direct or follow_up


def _is_hours_or_location_request(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    if "open to substitutions" in normalized or ("phone" in token_set and "pay" in token_set):
        return False
    policy_context = bool(
        token_set
        & {"parking", "park", "valet", "payment", "pay", "kids", "highchair", "delivery", "deliver"}
    )
    explicit_place_or_time = bool(
        token_set
        & {"hour", "hours", "open", "opening", "close", "closing", "location", "located", "address", "directions"}
    ) or any(phrase in normalized for phrase in ("where are you", "where is the restaurant", "get there", "find you"))
    if policy_context and not explicit_place_or_time:
        return False
    return bool(
        token_set
        & {
            "hour", "hours", "open", "opened", "opening", "close", "closes", "closing", "closed",
            "location", "located", "address", "directions", "direction", "phone", "number",
            "contact", "call",
        }
    ) or any(
        phrase in normalized
        for phrase in (
            "get there", "find you", "opening times", "closing time", "what time", "where are you",
            "where is maple", "where is the restaurant",
        )
    )


def _is_policy_request(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    if any(
        phrase in normalized
        for phrase in ("what can i order", "what can we order", "anything i can order")
    ):
        return False
    return bool(
        token_set
        & {
            "delivery", "deliver", "takeout", "takeaway", "pickup", "curbside", "parking", "park",
            "valet", "payment", "payments", "cash", "credit", "debit", "card", "cards", "giftcard",
            "dress", "attire", "children", "child", "kids", "kid", "highchair", "highchairs",
            "booster", "corkage", "byob", "gratuity", "tip", "tips", "substitution", "substitutions",
            "modify", "modification", "refund", "complaint", "complain", "special", "specials", "drinks",
            "beverages", "beverage", "order", "ordering", "pay", "swap", "omit", "remove", "hold",
            "unhappy", "terrible",
            "patio", "outside", "outdoor", "seating", "carryout", "tax", "taxes",
            "currency", "currencies", "fee", "fees",
        }
    ) or any(
        phrase in normalized
        for phrase in (
            "sparkling water", "still water", "bottled water", "tap water",
            "split the bill", "split bill", "split checks", "separate checks", "service charge",
            "outside wine", "bring wine", "kids menu", "dress code", "high chair", "payment methods",
            "place my order", "submit my order", "pay here", "process payment", "remove the",
            "order here", "can i order", "hold the", "bad meal", "bad experience", "food was terrible",
            "swap fries", "swap the", "not happy with my visit",
            "sit outside", "tables outside", "patio seating", "outdoor seating",
            "split the check", "split check", "separate bills", "pay separately",
            "to go", "carry out", "currency code",
            "yes place it", "confirm it", "send it to the kitchen", "go ahead", "checkout",
            "complete the order", "charge my card", "take my payment",
        )
    )


def _is_explicit_allergen_content_request(text: str, has_prior_item: bool) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    named = bool(_named_menu_items(text) or has_prior_item)
    if any(phrase in normalized for phrase in ("comes with", "come with", "served with", "pairs with", "goes with")):
        return False
    personal_restriction = normalized.startswith(("i ", "im ", "we ", "my ")) and any(
        phrase in normalized
        for phrase in ("cannot have", "cannot eat", "do not have", "do not eat", "allergic to", "without")
    )
    explicit_content_verb = bool(
        token_set & {"contain", "contains", "include", "includes", "allergen", "allergens"}
    ) or "is there" in normalized or "are there" in normalized
    if personal_restriction and not explicit_content_verb:
        return False
    if normalized.startswith("do you have ") and named and not bool(
        token_set & {"allergy", "allergen", "allergens", "contain", "contains", "include", "includes"}
    ):
        return False
    if bool(token_set & {"allergen", "allergens", "contain", "contains", "include", "includes", "including"}):
        return True
    if any(phrase in normalized for phrase in ("made with", "allergen in", "allergens in")):
        return True
    if re.search(r"\b(?:is|are) there\b.+\bin\b", normalized):
        return True
    if named and re.search(r"\b(?:dairy|milk|egg|eggs|fish|gluten|nuts?|walnuts?|peanuts?|shellfish|soy|sesame|wheat)\b.+\bin\b", normalized):
        return True
    if named and bool(token_set & {"has", "have"}):
        return True
    if named and "free" in token_set and normalized.startswith(("is ", "are ", "does ", "do ")):
        return True
    if "free" in token_set and normalized.startswith(("which ", "what ", "show ", "list ", "is ", "are ", "do ")):
        return True
    return any(
        phrase in normalized
        for phrase in ("which dishes have", "which items have", "what has", "any dishes with", "any items with")
    )


def _is_allergen_request(text: str, has_prior_item: bool) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    ingredient_query = _parse_ingredient_query(text)
    broad_allergen_terms = {
        "dairy", "milk", "egg", "eggs", "fish", "gluten", "nut", "nuts", "tree nuts",
        "peanut", "peanuts", "shellfish", "soy", "sesame", "wheat",
    }
    ingredient_names_an_allergen = bool(ingredient_query) and any(
        contains_term(ingredient_query[0], term) for term in broad_allergen_terms
    )
    if (
        ingredient_query
        and not ingredient_names_an_allergen
        and not bool(token_set & {"allergy", "allergies", "allergic", "allergen", "allergens", "safe"})
    ):
        return False
    if normalized.startswith(("which ", "what ", "show ", "list ")) and any(
        f"not {normalize(alias)}" in normalized
        for aliases in VERIFIED_DIETARY_ALIASES.values()
        for alias in aliases
    ):
        return False
    explicit_allergy_language = bool(
        token_set & {"allergy", "allergies", "allergic", "intolerance", "intolerant", "celiac", "coeliac"}
    )
    verified_diet_request = requested_labels(text, VERIFIED_DIETARY_ALIASES)
    non_dietary_free_request = any(
        label != "gluten"
        and any(f"{normalize(alias)} free" in normalized for alias in aliases)
        for label, aliases in TRACKED_ALLERGEN_ALIASES.items()
    )
    if verified_diet_request and not non_dietary_free_request and not explicit_allergy_language and not any(
        phrase in normalized for phrase in ("cannot eat", "cannot have", "do not eat", "allergic to")
    ):
        return False
    dietary_removal = any(
        is_label_removal(text, aliases) and any(contains_term(text, alias) for alias in aliases)
        for aliases in VERIFIED_DIETARY_ALIASES.values()
    )
    if dietary_removal and not explicit_allergy_language:
        return False
    tracked = requested_labels(text, TRACKED_ALLERGEN_ALIASES)
    untracked = requested_labels(text, UNTRACKED_ALLERGEN_ALIASES)
    explicit = bool(
        token_set
        & {
            "allergy", "allergies", "allergic", "allergen", "allergens", "intolerance", "intolerant",
            "celiac", "coeliac", "crosscontact", "safe",
        }
    )
    content = _is_explicit_allergen_content_request(text, has_prior_item) and bool(tracked or untracked)
    follow_up = has_prior_item and normalize(text).startswith("what about") and bool(tracked or untracked)
    removal = any(
        is_label_removal(text, aliases, allow_tolerance=True)
        for group in (TRACKED_ALLERGEN_ALIASES, UNTRACKED_ALLERGEN_ALIASES)
        for aliases in group.values()
    )
    return (
        _allergy_emergency(text)
        or explicit
        or is_restriction_statement(text) and bool(tracked or untracked)
        or content
        or follow_up
        or removal
    )


def _is_recommendation_request(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    return bool(
        token_set
        & {
            "recommend", "recommends", "recommended", "recommendation", "recommendations", "suggest",
            "suggests", "suggestion", "suggestions", "popular", "favorite", "favourite", "best",
            "pair", "pairs", "pairing", "pairings", "signature",
        }
    ) or any(
        phrase in normalized
        for phrase in (
            "what should i", "help me choose", "pick for me", "choose for me", "what is good",
            "goes with", "go well with", "pair with",
        )
    )


def _is_menu_request(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    return bool(
        token_set
        & {
            "menu", "food", "foods", "dish", "dishes", "meal", "meals", "serve", "serves", "serving",
            "price", "prices", "cost", "costs", "total", "subtotal", "starter", "starters", "appetizer",
            "appetizers", "main", "mains", "entree", "entrees", "dessert", "desserts", "vegan",
            "vegetarian", "meatless", "diet", "dietary", "preference", "preferences", "restriction",
            "restrictions", "cheapest", "burger", "burgers", "salmon", "perch", "cauliflower", "risotto",
            "salad", "pear", "torte", "chocolate", "vegetable", "vegetables", "fish", "mushroom",
            "mushrooms", "steak", "option", "options", "vegitarian", "dessets",
        }
    ) or any(
        phrase in normalized
        for phrase in (
            "what can i eat", "what can i have", "what can we eat", "what can we have",
            "anything i can eat", "anything we can eat", "what are my choices", "what are our choices",
            "what works for me", "what works for us", "show me what i can eat",
            "tell me what i can eat", "what can i order", "what can we order",
        )
    ) or bool(
        _named_menu_items(text)
        or _parse_ingredient_query(text)
        or requested_labels(text, UNVERIFIED_DIETARY_ALIASES)
        or requested_labels(text, VERIFIED_DIETARY_ALIASES)
    ) or "do you have" in normalized


def _is_greeting(text: str) -> bool:
    normalized = normalize(text)
    return normalized.startswith(("hi", "hello", "hey", "good morning", "good afternoon", "good evening"))


def _has_explicit_menu_signal(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    if token_set & {"tax", "taxes", "currency", "currencies", "fee", "fees"} and not (
        _named_menu_items(text) or _requested_category(text)
    ):
        return False
    if any(phrase in normalized for phrase in ("kids menu", "children s menu", "drink menu")):
        stripped = normalized.replace("kids menu", "").replace("children s menu", "").replace("drink menu", "")
        token_set = set(stripped.split())
    return bool(
        _named_menu_items(text)
        or _parse_ingredient_query(text)
        or _requested_category(text)
        or requested_labels(text, VERIFIED_DIETARY_ALIASES)
        or requested_labels(text, UNVERIFIED_DIETARY_ALIASES)
        or token_set
        & {
            "menu", "food", "foods", "dish", "dishes", "meal", "meals", "serve", "serves",
            "price", "prices", "total", "subtotal", "vegan", "vegetarian", "vegitarian",
            "gluten", "cheapest", "burger", "salmon", "perch", "cauliflower", "risotto",
            "mushroom", "mushrooms", "steak", "dessert", "desserts", "starter", "starters",
        }
        or any(
            phrase in normalized
            for phrase in (
                "show me options", "show options", "what do you serve", "what can i eat",
                "what can i have", "what can we eat", "anything i can eat", "what are my choices",
                "what works for me", "show me what i can eat", "tell me what i can eat",
                "what can i order",
            )
        )
    )


def _detect_topics(text: str, state: ChatState) -> list[Intent]:
    prior = _prior_intent(state)
    prior_items = _prior_item_names(state)
    if state.get("proposed_order_quantities") and normalize(text) in {
        "no", "no thanks", "nothing else", "that is all", "thats all", "all done",
    }:
        return ["general"]
    if state.get("proposed_order_quantities") and any(
        contains_term(text, phrase)
        for phrase in (
            "yes place it", "place it", "confirm it", "send it to the kitchen", "send to the kitchen",
            "go ahead", "checkout", "check out", "complete the order", "charge my card",
            "take my payment", "process my payment",
        )
    ):
        return ["policy"]
    # "Is this even about the restaurant?" gate: a clearly off-topic subject with
    # no competing restaurant signal must not trip a menu/recommendation dump.
    if _mentions_off_topic_subject(text) and not _has_restaurant_anchor(text):
        return ["general"]

    topics: list[Intent] = []

    allergen_request = _is_allergen_request(text, bool(prior_items))
    reservation_request = _is_reservation_request(text, prior)
    hours_request = _is_hours_or_location_request(text)
    policy_request = _is_policy_request(text)
    beverage_request = _is_beverage_request(text)
    # A personal allergy statement that names a family member ("my kid is
    # allergic…") is not a kids-menu/seating question.
    if (
        policy_request
        and allergen_request
        and _is_personal_child_reference(text)
        and not _asks_about_kids_facilities(text)
    ):
        policy_request = False
    if allergen_request:
        topics.append("allergens")
    if reservation_request:
        topics.append("reservation")
    if hours_request:
        topics.append("hours_location")
    if policy_request or beverage_request:
        topics.append("policy")
    if _is_recommendation_request(text) and not allergen_request and not beverage_request:
        topics.append("recommendation")
    elif _is_menu_request(text) and not allergen_request and not beverage_request:
        competing_topic = reservation_request or policy_request or hours_request or beverage_request
        complaint = any(term in normalize(text) for term in ("terrible", "bad meal", "unhappy", "complaint"))
        if not competing_topic or (_has_explicit_menu_signal(text) and not complaint):
            topics.append("menu")

    if not topics and prior_items and _is_follow_up(text):
        topics.append("menu")
    if not topics and prior == "hours_location" and (
        set(normalize(text).split())
        & {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tonight", "tomorrow"}
    ):
        topics.append("hours_location")
    if not topics and prior == "allergens" and normalize(text).startswith(("also ", "what about ")):
        topics.append("allergens")

    if topics:
        return list(dict.fromkeys(topics))
    if _is_greeting(text):
        return ["greeting"]
    return ["general"]


def classify_intent(state: ChatState) -> dict[str, object]:
    topics = _detect_topics(_last_user_text(state.get("messages", [])), state)
    return {"intent": topics[0], "topics": topics}


def _day_hours() -> dict[str, str]:
    hours = RESTAURANT["hours"]
    return {
        "monday": hours["Monday"],
        "tuesday": hours["Tuesday–Thursday"],
        "wednesday": hours["Tuesday–Thursday"],
        "thursday": hours["Tuesday–Thursday"],
        "friday": hours["Friday–Saturday"],
        "saturday": hours["Friday–Saturday"],
        "sunday": hours["Sunday"],
    }


def reservation_info(state: ChatState) -> dict[str, object]:
    text = _last_user_text(state.get("messages", []))
    normalized = normalize(text)
    availability = any(
        term in set(normalized.split())
        for term in ("available", "availability", "openings", "spot", "spots", "tonight", "tomorrow")
    ) or any(
        phrase in normalized
        for phrase in ("fully booked", "get in at", "space for", "room for", "come at", "fit us in", "fit me in")
    ) or (
        bool(re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", normalized))
        and bool(set(normalized.split()) & {"table", "tables", "dine", "party", "come", "space", "room", "slot", "seated", "fit"})
    )
    changing = bool(
        set(normalized.split())
        & {"cancel", "cancelled", "cancellation", "change", "modify", "reschedule", "noshow"}
    ) or "no show" in normalized
    group = any(phrase in normalized for phrase in ("group booking", "large group", "private room", "private event"))
    walk_in = any(phrase in normalized for phrase in ("walk in", "walk ins", "wait time", "waitlist"))

    if changing:
        reply = (
            "I can’t change or cancel a reservation in chat, and the cancellation or no-show terms are not "
            f"confirmed here. Please use {RESTAURANT['reservation_url']} or call {RESTAURANT['phone']} for help."
        )
    elif group:
        reply = (
            "Group limits, private-room availability, deposits, and event pricing are not confirmed here. "
            f"Use {RESTAURANT['reservation_url']} for a request or call {RESTAURANT['phone']} for group details."
        )
    elif availability:
        reply = (
            "I can’t see live table availability or confirm a booking in chat. Please check current times at "
            f"{RESTAURANT['reservation_url']} or call {RESTAURANT['phone']}."
        )
    elif walk_in:
        reply = (
            "I don’t have confirmed walk-in, wait-list, or current wait-time information. Please call "
            f"{RESTAURANT['phone']} for the latest guidance."
        )
    else:
        reply = (
            f"You can request a reservation at {RESTAURANT['reservation_url']} or call {RESTAURANT['phone']}. "
            "I can't confirm a booking, table, or live availability in chat."
        )
    return {"facts": reply, "draft_reply": reply, "use_model": False}


def restaurant_info(state: ChatState) -> dict[str, object]:
    text = _last_user_text(state.get("messages", []))
    normalized = normalize(text)
    token_set = set(normalized.split())
    parts: list[str] = []
    requested_days = [day for day in _day_hours() if day in token_set]
    asks_hours = bool(token_set & {"hour", "hours", "open", "opened", "opening", "close", "closes", "closing", "closed"})
    asks_location = bool(token_set & {"where", "location", "located", "address", "directions", "direction"}) or any(
        phrase in normalized for phrase in ("get there", "find you")
    )
    asks_phone = bool(token_set & {"phone", "number", "contact", "call"})
    date_specific = bool(
        token_set & {"today", "tonight", "tomorrow", "now", "holiday", "christmas", "thanksgiving", "easter"}
    )

    if asks_hours or requested_days or date_specific:
        if requested_days:
            day_map = _day_hours()
            lines = [f"{day.title()}: {day_map[day]}" for day in requested_days]
            parts.append("Regular listed hours: " + "; ".join(lines) + ".")
        else:
            lines = "; ".join(f"{day}: {value}" for day, value in RESTAURANT["hours"].items())
            parts.append(f"Regular listed hours are {lines}.")
        if date_specific:
            parts.append(
                f"Holiday and one-time schedule changes are not provided, so please call {RESTAURANT['phone']} "
                "to confirm hours for that date or whether the restaurant is open right now."
            )
    if asks_location:
        parts.append(f"Maple & Ember is located at {RESTAURANT['address']}.")
        if "direction" in normalized or "get there" in normalized:
            parts.append("Use that exact address in a current mapping service for route-specific directions.")
    if asks_phone:
        parts.append(f"The restaurant’s phone number is {RESTAURANT['phone']}.")
    if not parts:
        parts.append(
            f"Maple & Ember is at {RESTAURANT['address']}. The phone number is {RESTAURANT['phone']}."
        )
    reply = " ".join(parts)
    return {"facts": reply, "draft_reply": reply, "use_model": False}


def policy_info(state: ChatState) -> dict[str, object]:
    text = normalize(_last_user_text(state.get("messages", [])))
    phone = RESTAURANT["phone"]
    replies: list[str] = []

    def has_any(*terms: str) -> bool:
        return any(contains_term(text, term) for term in terms)

    if state.get("proposed_order_quantities") and has_any(
        "yes place it", "place it", "confirm it", "send it to the kitchen", "send to the kitchen",
        "go ahead", "checkout", "check out", "complete the order", "charge my card",
        "take my payment", "process my payment",
    ):
        replies.append(
            "I can’t submit or confirm the proposed order, send it to the kitchen, charge a card, or process "
            "payment in chat. It has not been submitted or paid."
        )

    if has_any(
        "delivery", "deliver", "takeout", "takeaway", "pickup", "curbside", "carryout",
        "carry out", "to go",
    ):
        replies.append(f"I don’t have confirmed takeout or delivery information. Please call {phone} for current options.")
    if has_any("parking", "park", "valet"):
        replies.append(f"I don’t have confirmed parking or valet information. Please call {phone} for current guidance.")
    if has_any("patio", "sit outside", "tables outside", "outdoor seating", "outside seating"):
        replies.append(f"I don’t have confirmed patio or outdoor-seating information. Please call {phone} for current options.")
    if has_any("place my order", "submit my order", "process payment", "pay here", "order here", "can i order"):
        replies.append("I can help assemble a proposed order, but I can’t submit the order or process payment in chat.")
    split_bill = has_any(
        "split the bill", "split bill", "split the check", "split checks", "split check", "separate checks",
        "separate bills", "pay separately",
    )
    if has_any("payment", "cash", "credit", "debit", "card", "cards", "giftcard", "pay") and not split_bill:
        replies.append(f"I don’t have confirmed payment-method or gift-card information. Please call {phone} for current options.")
    if has_any("dress", "attire", "dress code"):
        replies.append(f"I don’t have a confirmed dress code. Please call {phone} if you need guidance before visiting.")
    if has_any("children", "child", "kids", "kid", "highchair", "high chair", "high chairs", "booster"):
        replies.append(f"Children’s seating, high chairs, booster seats, and a kids’ menu are not confirmed here. Please call {phone}.")
    if has_any("corkage", "byob", "outside wine", "bring wine"):
        replies.append(f"I don’t have a confirmed corkage or outside-wine policy. Please call {phone} before bringing alcohol.")
    if split_bill:
        replies.append(f"I don’t have a confirmed split-bill policy. Please call {phone} to check what the restaurant can accommodate.")
    if has_any("gratuity", "service charge", "tip", "tips"):
        replies.append(f"Gratuity and service-charge rules are not confirmed here. Please call {phone} for current details.")
    if has_any("substitution", "substitutions", "modify", "modification", "remove", "hold", "omit", "swap"):
        replies.append(
            "Substitutions and modifications are not confirmed. Restaurant staff must approve the change, its "
            f"price, and any dietary or allergen implications; please call {phone}."
        )
    if has_any("special", "specials"):
        replies.append(f"I don’t have a verified current specials list. Please call {phone} for today’s offerings.")
    bringing_own = has_any("bring wine", "outside wine", "bring my own", "corkage", "byob", "bring a bottle")
    if not bringing_own and has_any(
        "drink", "drinks", "beverage", "beverages", "sparkling water", "wine", "wines", "beer", "beers",
        "cocktail", "cocktails", "champagne", "prosecco", "liquor", "spirits", "sake", "mocktail",
        "mocktails", "whisky", "whiskey", "vodka", "gin", "rum", "tequila", "alcohol", "alcoholic",
    ):
        replies.append(
            "The current menu data does not include a beverage list, so I can’t name or recommend any drinks. "
            f"Please call {phone} for current drink options."
        )
    if has_any("tax", "taxes"):
        replies.append(f"Whether listed prices include tax is not confirmed. Please call {phone} for current tax treatment.")
    if has_any("currency", "currencies", "currency code"):
        replies.append(
            f"Menu prices use the $ symbol, but the source data does not specify a currency code. Please call {phone} to confirm."
        )
    if has_any("fee", "fees"):
        replies.append(f"Taxes, modifications, and other fees are not confirmed. Please call {phone} for current details.")
    if has_any("refund", "complaint", "complain", "terrible", "bad meal", "unhappy", "not happy"):
        replies.append(f"I’m sorry there’s a concern. I can’t promise a refund or compensation; please call {phone} for staff follow-up.")
    if not replies:
        replies.append(f"I don’t have a confirmed Maple & Ember policy for that. Please call {phone} for current details.")
    reply = " ".join(dict.fromkeys(replies))
    return {"facts": reply, "draft_reply": reply, "use_model": False}


def _current_dietary_filters(text: str) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    normalized = normalize(text)
    complement_request = normalized.startswith(
        (
            "which ", "what ", "show ", "list ", "compare ", "are there ", "is there ",
            "are any ", "is any ", "do any ", "do you have ", "can you show ",
            "could you show ", "tell me ", "non ", "not ",
        )
    )
    for label, aliases in VERIFIED_DIETARY_ALIASES.items():
        if not any(contains_term(text, alias) for alias in aliases):
            continue
        complement_query = complement_request and any(
            f"not {normalize(alias)}" in normalized or f"non {normalize(alias)}" in normalized
            for alias in aliases
        )
        if complement_query:
            negative.append(label)
        elif is_label_removal(text, aliases):
            continue
        else:
            positive.append(label)
    return list(dict.fromkeys(positive)), list(dict.fromkeys(negative))


def _filter_menu_items(state: ChatState, text: str) -> list[MenuItem]:
    preferences = _effective_preferences(state)
    items = list(MENU)
    named = _named_menu_items(text)
    if not named and _is_follow_up(text):
        prior_names = set(_prior_item_names(state))
        named = [item for item in MENU if item.name in prior_names]
    if named:
        names = {item.name for item in named}
        items = [item for item in items if item.name in names]

    category = _requested_category(text)
    if not category and _is_follow_up(text):
        category = _prior_category(state)
    if category:
        items = [item for item in items if item.category == category]

    current_positive, current_negative = _current_dietary_filters(text)
    dietary = [label for label in preferences.dietary if label not in current_negative]
    dietary = list(dict.fromkeys([*dietary, *current_positive]))
    if "vegan" in dietary:
        items = [item for item in items if item.vegan]
    elif "vegetarian" in dietary:
        items = [item for item in items if item.vegetarian]
    if "gluten-free" in dietary:
        items = [item for item in items if item.gluten_free]
    for label in current_negative:
        if label == "vegan":
            items = [item for item in items if not item.vegan]
        elif label == "vegetarian":
            items = [item for item in items if not item.vegetarian]
        elif label == "gluten-free":
            items = [item for item in items if not item.gluten_free]

    if preferences.allergens:
        items = [
            item for item in items
            if all(allergen not in item.allergens for allergen in preferences.allergens)
        ]

    normalized = normalize(text)
    if any(term in set(normalized.split()) for term in ("vegetable", "vegetables")):
        items = [item for item in items if item.vegetarian]
    if "fish" in set(normalized.split()) and not _is_allergen_request(text, bool(_prior_item_names(state))):
        items = [item for item in items if "fish" in item.allergens]
    return items


def _format_grouped_menu(items: Sequence[MenuItem]) -> str:
    sections: list[str] = []
    for category, title in (("starter", "Starters"), ("main", "Mains"), ("dessert", "Desserts")):
        category_items = [item for item in items if item.category == category]
        if not category_items:
            continue
        sections.append(title + ":\n" + "\n".join(f"- {format_item(item)}" for item in category_items))
    return "\n".join(sections)


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


def _pairing_response(state: ChatState, text: str, named: Sequence[MenuItem]) -> tuple[str, list[str]] | None:
    normalized = normalize(text)
    if not any(term in normalized for term in ("pair", "pairs", "pairing", "goes with", "go well with")):
        return None
    preferences = _effective_preferences(state)
    allowed = _filter_menu_items(state, "")
    allowed_names = {item.name for item in allowed}
    by_name = {item.name: item for item in MENU}
    source = named[0] if named else None
    preferred_pairs: dict[str, tuple[str, ...]] = {
        "Roasted Beet Salad": ("Wild Mushroom Risotto", "Maple-Glazed Salmon"),
        "Crispy Lake Erie Perch": ("Maple-Glazed Salmon",),
        "Charred Cauliflower Steak": ("Cider-Poached Pear",),
        "Wild Mushroom Risotto": ("Roasted Beet Salad", "Dark Chocolate Torte"),
        "Maple-Glazed Salmon": ("Roasted Beet Salad", "Crispy Lake Erie Perch"),
        "Ember Burger": ("Cider-Poached Pear", "Dark Chocolate Torte"),
        "Cider-Poached Pear": ("Charred Cauliflower Steak",),
        "Dark Chocolate Torte": ("Wild Mushroom Risotto",),
    }
    choices: list[MenuItem] = []
    if source:
        for name in preferred_pairs.get(source.name, ()):
            candidate = by_name[name]
            if "dessert" in normalized and candidate.category != "dessert":
                continue
            if "starter" in normalized and candidate.category != "starter":
                continue
            if name in allowed_names:
                choices.append(candidate)
        if not choices:
            requested_category = _requested_category(text)
            choices = [
                item for item in allowed
                if item.name != source.name and (not requested_category or item.category == requested_category)
            ]
    elif "vegan" in preferences.dietary or contains_term(text, "vegan"):
        source = by_name["Charred Cauliflower Steak"]
        choices = [by_name["Cider-Poached Pear"]] if "Cider-Poached Pear" in allowed_names else []
    if not source or source.name not in allowed_names or not choices:
        reply = (
            "I couldn’t find a listed pairing that matches all of the active preferences. "
            "Restaurant staff can discuss other options and preparation details."
        )
        return reply, []
    suggestion = choices[0]
    reply = (
        f"An optional pairing is {source.name} with {suggestion.name}. "
        f"{suggestion.name} is listed at ${suggestion.price:.0f} and comes with {suggestion.description}. "
        "That is an assistant suggestion, not an official chef pairing or availability guarantee."
    )
    return reply, [source.name, suggestion.name]


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _quantity_for_item(text: str, item: MenuItem) -> int | None:
    aliases = ITEM_ALIASES[item.name]
    normalized = normalize(text)
    for alias in sorted(aliases, key=len, reverse=True):
        term = normalize(alias)
        match = re.search(
            rf"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*)?{re.escape(term)}s?\b",
            normalized,
        )
        if match:
            raw = match.group(1)
            return int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        if re.search(rf"\b(?:a|an)\s+{re.escape(term)}s?\b", normalized):
            return 1
    return None


def _standalone_quantity(text: str) -> int | None:
    match = re.search(
        r"\b(?:make that\s+)(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        normalize(text),
    )
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else NUMBER_WORDS[raw]


def _render_order(quantities: dict[str, int]) -> str:
    by_name = {item.name: item for item in MENU}
    ordered_names = [item.name for item in MENU if item.name in quantities]
    subtotal = sum(by_name[name].price * quantities[name] for name in ordered_names)
    lines = [
        f"- {quantities[name]} × {name} — ${by_name[name].price * quantities[name]:.0f}"
        for name in ordered_names
    ]
    return (
        "Proposed order:\n"
        + "\n".join(lines)
        + f"\nMenu-price subtotal: ${subtotal:.0f}\n"
        "Taxes, modifications, fees, service charges, and gratuity are not confirmed. This proposed order has "
        "not been submitted or paid. Do any diners have food allergies? Would you like to add anything else?"
    )


def _order_update(
    state: ChatState,
    text: str,
    items: Sequence[MenuItem],
) -> tuple[str, dict[str, int]] | None:
    normalized = normalize(text)
    token_set = set(normalized.split())
    prior = {
        name: quantity
        for name, quantity in state.get("proposed_order_quantities", {}).items()
        if any(item.name == name for item in MENU)
        and isinstance(quantity, int)
        and not isinstance(quantity, bool)
        and quantity > 0
    }
    explicit_names = {item.name for item in _named_menu_items(text)}
    named = [item for item in items if item.name in explicit_names]
    explicit = {
        item.name: quantity
        for item in named
        if (quantity := _quantity_for_item(text, item)) is not None
    }
    asks_total = bool(token_set & {"total", "subtotal", "altogether"}) or any(
        phrase in normalized for phrase in ("what is the total", "what s the total", "how much altogether")
    )
    order_cue = bool(set(normalized.split()) & {"order", "ordering", "add"}) or any(
        phrase in normalized
        for phrase in ("i will have", "i would like", "i want", "give me", "make that")
    )
    standalone = _standalone_quantity(text)
    if not prior and not named:
        if standalone is not None and order_cue and len(items) == 1:
            if standalone <= 0:
                return (
                    "Please use a quantity of at least 1. Nothing has been changed.",
                    {},
                )
            updated = {items[0].name: standalone}
            return _render_order(updated), updated
        if standalone is not None and order_cue and len(items) > 1:
            return (
                f"Which item should I change to {standalone}: {natural_join([item.name for item in items], 'or')}?",
                {},
            )
        return None
    if any(quantity <= 0 for quantity in explicit.values()):
        return (
            "Please use a quantity of at least 1 for a proposed order. Nothing has been changed.",
            prior,
        )

    if standalone is not None and not named:
        if standalone <= 0:
            return (
                "Please use a quantity of at least 1. Nothing has been changed.",
                prior,
            )
        if len(prior) > 1:
            names = natural_join(list(prior), "or")
            return (
                f"Which item should I change to {standalone}: {names}?",
                prior,
            )
        if len(prior) == 1:
            updated = dict(prior)
            updated[next(iter(updated))] = standalone
            return _render_order(updated), updated

    if not named:
        if prior and asks_total:
            return _render_order(prior), prior
        return None

    has_explicit_quantity = bool(explicit)
    if not order_cue and not asks_total and not has_explicit_quantity:
        return None

    if "make that" in normalized:
        updated = dict(prior)
        for item in named:
            quantity = explicit.get(item.name, standalone)
            if quantity is None:
                continue
            updated[item.name] = quantity
    elif "add" in token_set and prior:
        updated = dict(prior)
        for item in named:
            updated[item.name] = updated.get(item.name, 0) + explicit.get(item.name, 1)
    else:
        updated = {item.name: explicit.get(item.name, 1) for item in named}

    if not updated:
        return None
    return _render_order(updated), updated


def menu_info(state: ChatState) -> dict[str, object]:
    text = _last_user_text(state.get("messages", []))
    normalized = normalize(text)
    token_set = set(normalized.split())
    preferences = _effective_preferences(state)
    raw_unverified = requested_labels(text, UNVERIFIED_DIETARY_ALIASES)
    current_unverified = [
        label
        for label in raw_unverified
        if not is_label_removal(text, UNVERIFIED_DIETARY_ALIASES[label])
    ]
    unverified = list(dict.fromkeys([*preferences.unverified_diets, *current_unverified]))
    if preferences.untracked_allergens:
        labels = natural_join(preferences.untracked_allergens)
        reply = (
            f"The current menu data does not track {labels} separately, so I can’t confirm an option that meets "
            f"all restrictions. Please call {RESTAURANT['phone']} and discuss cross-contact risk with staff."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}
    if unverified:
        labels = natural_join(unverified)
        reply = (
            f"The current menu does not identify any dishes as {labels}, so I can’t confirm an option that meets "
            f"all of those restrictions. Please call {RESTAURANT['phone']}; ingredients alone do not establish preparation, "
            "nutrition targets, sourcing, or certification."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}

    generic_dietary = bool(token_set & {"diet", "dietary", "preference", "preferences", "restriction", "restrictions"})
    current_dietary, current_negative_dietary = _current_dietary_filters(text)
    removed_dietary = [
        label
        for label, aliases in VERIFIED_DIETARY_ALIASES.items()
        if any(contains_term(text, alias) for alias in aliases) and is_label_removal(text, aliases)
    ]
    removed_unverified = [
        label for label in raw_unverified if is_label_removal(text, UNVERIFIED_DIETARY_ALIASES[label])
    ]
    correction_only = bool(removed_dietary or removed_unverified) and not normalized.startswith(
        ("show ", "list ", "which ", "what ", "compare ", "recommend ", "do you ")
    ) and not bool(
        token_set & {"menu", "price", "prices", "cost", "total", "subtotal", "recommend", "suggest"}
    )
    if correction_only:
        labels = natural_join([*removed_dietary, *removed_unverified], "and")
        reply = f"Understood — I won’t keep {labels} as an active dietary restriction. What would you like to see?"
        return {"facts": reply, "draft_reply": reply, "use_model": False}
    if generic_dietary and not current_dietary:
        reply = (
            "Of course — tell me every dietary preference or restriction that applies; you can list more than one. "
            "I can filter verified vegan, "
            "vegetarian, and gluten-free labels and check declared dairy, egg, fish, gluten, and tree-nut allergens. "
            "For halal, kosher, keto, pescatarian, paleo, and other unverified requirements, staff must confirm."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}

    items = _filter_menu_items(state, text)
    named = _named_menu_items(text)
    if not named and _is_follow_up(text):
        prior_names = set(_prior_item_names(state))
        named = [item for item in MENU if item.name in prior_names]
    category = _requested_category(text) or (_prior_category(state) if _is_follow_up(text) else None)

    pairing = _pairing_response(state, text, named)
    if pairing:
        reply, context_names = pairing
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": context_names,
            "context_category": category or "",
        }

    ingredient_query = _parse_ingredient_query(text)
    if ingredient_query:
        target, subject_items = ingredient_query
        if subject_items:
            lines = [
                f"- {item.name}: its listed description "
                f"{'includes' if _description_lists_ingredient(item, target) else 'does not include'} {target}."
                for item in subject_items
            ]
            reply = "\n".join(lines) + (
                "\nMenu descriptions may not list every ingredient. If this is allergy-related, confirm ingredients "
                "and cross-contact risk with restaurant staff."
            )
            return {
                "facts": reply,
                "draft_reply": reply,
                "use_model": False,
                "context_item_names": [item.name for item in subject_items],
            }

        ingredient_matches = _ingredient_matches(target)
        if ingredient_matches:
            reply = f"{target.title()} is listed in:\n" + "\n".join(
                f"- {format_item(item)}" for item in ingredient_matches
            )
        else:
            reply = (
                f"I couldn’t find {target} in any current listed item name or description. "
                "Menu descriptions may not list every ingredient, so restaurant staff should confirm."
            )
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in ingredient_matches],
        }

    unknown_item_question = ("do you have" in normalized or "do you serve" in normalized) and not named and not category
    recognized_generic = bool(
        token_set
        & {
            "menu", "food", "foods", "serve", "vegetable", "vegetables", "fish", "option", "options",
            "diet", "dietary", "vegan", "vegetarian", "vegitarian", "meatless", "gluten", "celiac", "coeliac",
        }
    ) or bool(current_dietary or current_negative_dietary)
    if unknown_item_question and not recognized_generic:
        reply = "I couldn’t find that item or ingredient on the current listed menu. I can show the full menu or check another item."
        return {"facts": "No matching menu item or ingredient.", "draft_reply": reply, "use_model": False}

    if not items:
        reply = "I couldn’t find a listed item matching all of those filters. Restaurant staff can confirm whether another option or modification is possible."
        return {"facts": "No menu items matched all filters.", "draft_reply": reply, "use_model": False}

    if "cheapest" in token_set and items:
        cheapest = min(items, key=lambda item: item.price)
        reply = f"The lowest listed price among those options is {cheapest.name} at ${cheapest.price:.0f}."
        return {
            "facts": format_item(cheapest),
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [cheapest.name],
            "context_category": cheapest.category,
        }

    asks_price = "how much" in normalized or bool(token_set & {"price", "prices", "cost", "costs"})
    order_items = [item for item in named if item in items]
    if len(order_items) == 1 and asks_price and _quantity_for_item(text, order_items[0]) in {None, 1} and not bool(
        token_set & {"total", "subtotal", "altogether", "order"}
    ):
        item = order_items[0]
        reply = f"{item.name} is listed at ${item.price:.0f}."
        return {
            "facts": format_item(item),
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name],
            "context_category": item.category,
        }

    order = _order_update(state, text, order_items)
    if order:
        order_reply, quantities = order
        return {
            "facts": "\n".join(
                format_item(item) for item in MENU if item.name in quantities
            ),
            "draft_reply": order_reply,
            "use_model": False,
            "context_item_names": list(quantities),
            "proposed_order_quantities": quantities,
        }

    intent = state.get("intent", "menu")
    full_request = (
        bool(token_set & {"full", "entire"})
        or ("menu" in token_set and not named and not category and not current_dietary)
        or "what food do you serve" in normalized
        or "what do you serve" in normalized
        or "what is on the menu" in normalized
        or "what s on the menu" in normalized
        or (bool(token_set & {"all"}) and bool(token_set & {"price", "prices"}))
    )
    if intent == "recommendation":
        selected = items[:4]
        if "signature" in token_set:
            preamble = "I don’t have a verified signature-dish designation. Based on the current menu, I’d suggest:"
        elif "popular" in token_set:
            preamble = "I don’t have verified popularity rankings. Based on the current menu, I’d suggest:"
        else:
            preamble = "Based on what you've told me, I'd suggest:"
        draft = preamble + "\n" + "\n".join(f"- {format_item(item)}" for item in selected[:2])
        if preferences.allergens:
            draft += (
                "\nThese suggestions omit items declaring your saved allergens, but that is not an "
                "allergen-safety guarantee. Please confirm preparation and cross-contact risk with staff."
            )
        return {
            "facts": "\n".join(format_item(item) for item in selected),
            "draft_reply": draft,
            "use_model": not preferences.allergens and len(selected) > 1,
            "candidate_names": [item.name for item in selected],
            "recommendation_preamble": preamble,
            "context_item_names": [item.name for item in selected[:2]],
            "context_category": category or "",
        }

    selected = items if full_request or category or current_dietary or current_negative_dietary else items[:4]
    if full_request:
        reply = "Of course — here are the current listed menu options at Maple & Ember:\n" + _format_grouped_menu(selected)
        if token_set & {"today", "tonight"}:
            reply += "\nLive item availability for tonight is not available in chat."
    elif named:
        reply = "The current listed menu includes:\n" + "\n".join(f"- {format_item(item)}" for item in selected)
        if bool(token_set & {"available", "availability", "today", "tonight"}):
            reply += "\nLive item availability for tonight is not available in chat."
    else:
        reply = "Here are the matching options from the current listed menu:\n" + "\n".join(
            f"- {format_item(item)}" for item in selected
        )
    if preferences.allergens:
        display = natural_join(preferences.allergens)
        reply += (
            f"\nThese results omit items declaring {display}, but that is not an allergen-safety guarantee. "
            "Please confirm preparation and cross-contact risk with restaurant staff."
        )
    return {
        "facts": "\n".join(format_item(item) for item in selected),
        "draft_reply": reply,
        "use_model": False,
        "context_item_names": [item.name for item in selected],
        "context_category": category or "",
    }


def _allergy_emergency(text: str) -> bool:
    normalized = normalize(text)
    return any(
        phrase in normalized
        for phrase in (
            "trouble breathing", "cannot breathe", "difficulty breathing", "throat swelling",
            "throat is swelling", "throat feels swollen", "anaphylaxis", "allergic reaction",
        )
    )


def _allergen_labels_for_text(text: str) -> tuple[list[str], list[str]]:
    tracked = requested_labels(text, TRACKED_ALLERGEN_ALIASES)
    untracked = requested_labels(text, UNTRACKED_ALLERGEN_ALIASES)
    for label, aliases in TRACKED_ALLERGEN_ALIASES.items():
        if label in tracked and is_label_removal(text, aliases, allow_tolerance=True):
            tracked.remove(label)
    for label, aliases in UNTRACKED_ALLERGEN_ALIASES.items():
        if label in untracked and is_label_removal(text, aliases, allow_tolerance=True):
            untracked.remove(label)

    normalized = normalize(text)
    constraint_cue = bool(
        set(normalized.split()) & {"allergy", "allergies", "allergic", "intolerance", "intolerant", "avoid", "avoiding"}
    ) or any(
        phrase in normalized for phrase in ("cannot eat", "cannot have", "do not eat", "without", "makes me sick")
    ) or normalized.startswith("no ")
    if _named_menu_items(text) and not constraint_cue and "fish" in tracked and not contains_term(text, "fish"):
        tracked.remove("fish")

    bare_nuts = contains_term(text, "nut") or contains_term(text, "nuts")
    specific_nuts = any(contains_term(text, term) for term in ("tree nut", "tree nuts", "walnut", "walnuts", "peanut", "peanuts"))
    if bare_nuts and not specific_nuts:
        if "tree nuts" not in tracked:
            tracked.append("tree nuts")
        if "unspecified nuts" not in untracked:
            untracked.append("unspecified nuts")
    if "shellfish" in untracked and "fish" in tracked:
        tracked.remove("fish")
    if specific_nuts and "tree nuts" in tracked and "unspecified nuts" in untracked:
        untracked.remove("unspecified nuts")
    return tracked, untracked


def _item_allergen_summary(items: Sequence[MenuItem]) -> str:
    lines: list[str] = []
    for item in items:
        if item.allergens:
            lines.append(f"- {item.name}: declares {', '.join(item.allergens)}.")
        else:
            lines.append(f"- {item.name}: no allergens are declared in the menu data.")
    return "\n".join(lines)


def allergen_info(state: ChatState) -> dict[str, object]:
    text = _last_user_text(state.get("messages", []))
    normalized = normalize(text)
    token_set = set(normalized.split())
    if _allergy_emergency(text):
        reply = (
            "If someone may be having a severe allergic reaction or another medical emergency, contact local "
            "emergency services now. I can’t diagnose or provide medical treatment. For restaurant follow-up, "
            f"call {RESTAURANT['phone']}."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}

    no_allergies = normalized in {"no allergies", "allergies none", "no known allergies"} or any(
        phrase in normalized
        for phrase in (
            "i have no allergies", "i have no food allergies", "i do not have any allergies",
            "i do not have food allergies", "not allergic to anything",
        )
    )
    if no_allergies:
        reply = (
            "Understood — I won’t keep any allergy restrictions for this session. "
            "If that changes, please tell me before choosing food."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}

    dairy_complement_query = normalized.startswith(("which ", "what ", "show ", "list ")) and any(
        phrase in normalized for phrase in ("not dairy free", "not dairyfree", "with dairy")
    )
    if dairy_complement_query:
        declared = [item for item in MENU if "dairy" in item.allergens]
        reply = (
            "The menu does not define a separate dairy-free flag. These items declare dairy:\n"
            + "\n".join(f"- {format_item(item)}" for item in declared)
            + "\nAn item without a dairy declaration is not guaranteed dairy-free or free from cross-contact."
        )
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in declared],
        }

    preferences = _effective_preferences(state)
    current_tracked, current_untracked = _allergen_labels_for_text(text)
    named = _named_menu_items(text)
    if not named and _is_follow_up(text):
        prior_names = set(_prior_item_names(state))
        named = [item for item in MENU if item.name in prior_names]

    generic_allergen_question = bool(token_set & {"allergen", "allergens"}) and bool(named)
    content_question = _is_explicit_allergen_content_request(text, bool(named)) or (
        bool(named) and normalize(text).startswith("what about")
    )
    if not content_question:
        # For a personal restriction statement, the post-merge structured
        # preference state is authoritative. Raw item/ingredient words in the
        # same sentence must not become additional allergen targets.
        current_tracked = []
        current_untracked = []
    if generic_allergen_question:
        reply = (
            _item_allergen_summary(named)
            + "\nThese are declarations from the menu data, not an allergen-safety or cross-contact guarantee."
        )
        if preferences.untracked_allergens:
            reply += (
                f" I also can’t verify the saved {natural_join(preferences.untracked_allergens)} restriction from "
                "the menu data; please contact staff."
            )
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in named],
        }

    if content_question and current_untracked and not current_tracked:
        display = natural_join(current_untracked)
        reply = (
            f"The menu data does not track {display} separately, so I can’t confirm whether the requested item "
            f"contains it. Please call {RESTAURANT['phone']} and discuss cross-contact risk with staff."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}

    if content_question and "free" in token_set and current_tracked and not named:
        display = natural_join(current_tracked)
        options = [
            item for item in MENU
            if all(label not in item.allergens for label in current_tracked)
        ]
        reply = (
            f"Based only on declared menu allergens, these items do not declare {display}:\n"
            + "\n".join(f"- {format_item(item)}" for item in options)
            + "\nThat does not prove the items are allergen-free or safe from cross-contact; please confirm with staff."
        )
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in options],
        }

    if content_question and current_tracked:
        display = natural_join(current_tracked)
        scope = named or list(MENU)
        if named and len(named) == 1:
            item = named[0]
            declared = [label for label in current_tracked if label in item.allergens]
            not_declared = [label for label in current_tracked if label not in item.allergens]
            if declared and not not_declared:
                reply = f"The menu declares {natural_join(declared, 'and')} for {item.name}."
            elif declared:
                reply = (
                    f"{item.name} declares {natural_join(declared, 'and')}; it does not declare "
                    f"{natural_join(not_declared, 'or')} in the menu data."
                )
            else:
                reply = f"{item.name} does not declare {display} in the menu data."
            if "gluten" in current_tracked:
                reply += f" It {'is' if item.gluten_free else 'is not'} marked gluten-free."
        elif named:
            lines = []
            for item in scope:
                declared = [label for label in current_tracked if label in item.allergens]
                if declared:
                    lines.append(f"- {item.name}: declares {natural_join(declared, 'and')}.")
                else:
                    lines.append(f"- {item.name}: does not declare {display} in the menu data.")
            reply = "\n".join(lines)
        else:
            declared_items = [
                item for item in scope if any(label in item.allergens for label in current_tracked)
            ]
            if declared_items:
                reply = f"The menu declares {display} for:\n" + "\n".join(
                    f"- {format_item(item)}" for item in declared_items
                )
            else:
                reply = f"No current menu item declares {display} in its allergen data."
        if current_untracked:
            reply += (
                f"\nThe menu does not separately track {natural_join(current_untracked)}, so that part requires "
                "direct staff confirmation."
            )
        saved_declared = list(
            dict.fromkeys(
                label
                for item in named
                for label in preferences.allergens
                if label in item.allergens and label not in current_tracked
            )
        )
        if saved_declared:
            reply += f"\nAlso, the item declares your saved {natural_join(saved_declared, 'and')} restriction."
        if preferences.untracked_allergens:
            reply += (
                f"\nYour saved {natural_join(preferences.untracked_allergens)} restriction is not fully tracked "
                "by the menu data."
            )
        reply += "\nThis is not proof that an item is free from that allergen or cross-contact. Please confirm with staff."
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in scope] if named else [],
        }

    untracked = [
        label
        for label in dict.fromkeys([*preferences.untracked_allergens, *current_untracked])
        if not _is_menu_item_word(label)
    ]
    if untracked:
        named_untracked = [label for label in untracked if label != _UNNAMED_ALLERGEN]
        if not named_untracked:
            # Guest reported an allergy without naming one; ask instead of
            # echoing the internal "unnamed allergen" sentinel.
            reply = (
                "Which allergy or intolerance should I check? The menu data tracks dairy, egg, fish, gluten, "
                f"and tree nuts for individual items. For anything else, please call {RESTAURANT['phone']}, and "
                "note the menu data can’t guarantee against cross-contact."
            )
            return {"facts": reply, "draft_reply": reply, "use_model": False}
        display = natural_join(named_untracked)
        reply = (
            f"The current menu data does not track {display} separately, so I can’t identify a safe option. "
            f"Please call {RESTAURANT['phone']} before ordering and tell staff about every allergy and "
            "cross-contact concern."
        )
        return {"facts": reply, "draft_reply": reply, "use_model": False}

    requested = list(dict.fromkeys([*preferences.allergens, *current_tracked]))
    removed_labels = [
        label
        for label, aliases in TRACKED_ALLERGEN_ALIASES.items()
        if any(contains_term(text, alias) for alias in aliases)
        and is_label_removal(text, aliases, allow_tolerance=True)
    ]
    if removed_labels and not requested:
        reply = (
            f"Understood — I won’t keep {natural_join(removed_labels, 'or')} as an active allergy restriction. "
            "Tell me if another allergy or intolerance applies."
        )
        if _is_recommendation_request(text):
            choices = _filter_menu_items(state, text)[:2]
            if choices:
                reply += "\nBased on your updated preferences, I’d suggest:\n" + "\n".join(
                    f"- {format_item(item)}" for item in choices
                )
        return {"facts": reply, "draft_reply": reply, "use_model": False}
    if named and ("safe" in token_set or "celiac" in token_set or "coeliac" in token_set or "free" in token_set):
        status = []
        for item in named:
            labels = []
            if "gluten" in requested:
                labels.append("is marked gluten-free" if item.gluten_free else "is not marked gluten-free")
            labels.append(
                "declares " + ", ".join(item.allergens)
                if item.allergens else "has no allergens declared in the menu data"
            )
            status.append(f"- {item.name}: {'; '.join(labels)}.")
        reply = "\n".join(status) + (
            "\nThese labels are not a safety guarantee. Please confirm preparation and cross-contact risk with staff."
        )
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in named],
        }

    if requested:
        items = _filter_menu_items(state, text)
        items = [item for item in items if all(label not in item.allergens for label in requested)]
        display = natural_join(requested)
        if items:
            reply = (
                f"Based only on menu labels and declared allergens, these options do not declare {display}:\n"
                + "\n".join(f"- {format_item(item)}" for item in items[:4])
                + "\nThis is not an allergen-safety guarantee. Please confirm preparation and cross-contact risk with staff."
            )
        else:
            reply = (
                f"I couldn’t find a listed item matching all filters while omitting items that declare {display}. "
                f"Please call {RESTAURANT['phone']} to discuss the restrictions and cross-contact risk."
            )
        return {
            "facts": reply,
            "draft_reply": reply,
            "use_model": False,
            "context_item_names": [item.name for item in items[:4]],
        }

    reply = (
        "Please name every allergy or intolerance that applies. The menu data declares dairy, egg, fish, gluten, "
        "and tree nuts for individual items, but it cannot guarantee allergen safety or prevent cross-contact."
    )
    return {"facts": reply, "draft_reply": reply, "use_model": False}


def general_info(state: ChatState) -> dict[str, object]:
    text = normalize(_last_user_text(state.get("messages", [])))
    has_proposed_order = bool(state.get("proposed_order_quantities"))
    off_topic = False
    if has_proposed_order and text in {"no thanks", "nothing else", "that is all", "thats all", "all done"}:
        reply = (
            "Understood. Your proposed order has not been submitted or paid; use the restaurant’s ordering "
            "channel or call staff when you’re ready to place it."
        )
    elif has_proposed_order and text == "no":
        reply = (
            "Just to confirm: does “no” mean there are no food allergies, or that you would like nothing else?"
        )
    elif state.get("intent") == "greeting":
        reply = (
            f"Welcome to {RESTAURANT['name']}! I can help with the menu, dietary needs, regular hours, "
            "location, policies, or reservation information."
        )
    elif text in {"thanks", "thank you", "thank you very much", "much appreciated"}:
        reply = "You’re welcome! I’m here if you need anything else about Maple & Ember."
    elif text in {"bye", "goodbye", "see you", "see you later"}:
        reply = "Goodbye — we hope you enjoy your visit to Maple & Ember."
    elif any(
        phrase in text
        for phrase in (
            "kind of restaurant", "about the restaurant", "what is maple and ember", "what cuisine",
            "tell me about maple and ember", "tell me about the restaurant",
            "tell me about maple ember",
        )
    ):
        reply = f"{RESTAURANT['name']} is {RESTAURANT['description'].lower()}"
    else:
        reply = (
            f"I’m the {RESTAURANT['name']} restaurant assistant. I can help with our menu, "
            "dietary preferences, hours, location, and reservation information."
        )
        off_topic = True
    return {"facts": RESTAURANT["description"], "draft_reply": reply, "use_model": False, "off_topic": off_topic}


def route_intent(state: ChatState) -> str:
    intent = state.get("intent", "general")
    if intent in {"menu", "recommendation"}:
        return "menu_info"
    if intent == "allergens":
        return "allergen_info"
    if intent == "hours_location":
        return "restaurant_info"
    if intent == "reservation":
        return "reservation_info"
    if intent == "policy":
        return "policy_info"
    return "general_info"


def _topic_result(topic: Intent, state: ChatState) -> dict[str, object]:
    if topic in {"menu", "recommendation"}:
        return menu_info(state)
    if topic == "allergens":
        return allergen_info(state)
    if topic == "hours_location":
        return restaurant_info(state)
    if topic == "reservation":
        return reservation_info(state)
    if topic == "policy":
        return policy_info(state)
    return general_info(state)


def _grounded_model_text(content: object, draft: str) -> str:
    """Never expose free-form model output; retained as a compatibility safety boundary."""
    return draft


# The off-topic redirect is the one place the model may speak freely, so its
# output is bounded hard: no leaked internals, no invented facts, must steer back.
_OFFTOPIC_CAPABILITY_CUES = (
    "menu", "dish", "hour", "location", "reserv", "book", "table", "dietary", "allerg", "visit",
    "maple & ember", "maple and ember",
)
_OFFTOPIC_BLOCK_PHRASES = (
    "system prompt", "instruction", "internal", "verified facts", "fallback", "as an ai",
    "language model", "output contract", "the guest said", "the user said", "the user just",
    "analysis:", "reasoning:", "i cannot fulfill", "i can't fulfill",
)


def _invents_restaurant_fact(text: str) -> bool:
    """Off-topic chit-chat must stay fact-free: reject prices, numbers, item names, contact details."""
    lowered = text.lower()
    if "$" in text or any(char.isdigit() for char in text):
        return True
    if any(
        alias in lowered
        for aliases in ITEM_ALIASES.values()
        for alias in aliases
        if len(alias) > 4
    ):
        return True
    return any(token in lowered for token in ("king street", "416", "http", "reservation_url"))


def _grounded_offtopic_reply(content: object, draft: str) -> str:
    """Return the model's off-topic redirect only if it is safe; otherwise the verified draft."""
    if not isinstance(content, str):
        return draft
    # Any reasoning/think markup means the model went off-script — do not trust it.
    if re.search(r"</?think\b", content, flags=re.IGNORECASE):
        return draft
    cleaned = content.strip()
    if not cleaned:
        return draft
    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in _OFFTOPIC_BLOCK_PHRASES):
        return draft
    if _invents_restaurant_fact(cleaned):
        return draft
    if len(cleaned) > 320 or cleaned.count("\n") > 2:
        return draft
    # Must actually steer back to a restaurant capability.
    if not any(cue in lowered for cue in _OFFTOPIC_CAPABILITY_CUES):
        return draft
    return cleaned


# Off-topic composition budget. A reasoning model (qwen3) must be given the
# reasoning channel so its chain-of-thought does not leak into the reply, plus
# enough tokens to finish thinking and answer; the timeout bounds worst-case
# latency when it over-deliberates, in which case we fall back to the draft.
_OFFTOPIC_NUM_PREDICT = 512
_OFFTOPIC_TIMEOUT_SECONDS = 6.0


def _offtopic_model(model: BaseChatModel) -> BaseChatModel:
    """Reconfigure a reasoning model for one free-form call; leave others as-is."""
    if hasattr(model, "reasoning") and hasattr(model, "num_predict"):
        try:
            return model.model_copy(update={"reasoning": True, "num_predict": _OFFTOPIC_NUM_PREDICT})
        except Exception:  # pragma: no cover - defensive; provider without model_copy
            return model
    return model


async def _compose_offtopic_reply(model: BaseChatModel, user_text: str, draft: str) -> str:
    """Acknowledge an off-topic message in one playful clause, then steer back — bounded and fact-free."""
    system = SystemMessage(
        content=(
            f"You are the automated assistant for {RESTAURANT['name']}, a restaurant. The guest just said "
            "something off-topic or casual. In ONE short, warm, lightly playful clause, acknowledge what they "
            "said, then steer back to what you can actually help with: the menu, dietary needs, hours, location, "
            "or booking a table. Do not answer the off-topic question. Invent NO facts — no menu items, dishes, "
            "prices, hours, addresses, phone numbers, or claims about the restaurant. Keep it under 40 words. "
            "Return only the reply, with no preamble or quotation marks.\n\n"
            f"Voice and off-topic guidance:\n{persona_offtopic()}"
        )
    )
    try:
        response = await asyncio.wait_for(
            _offtopic_model(model).ainvoke([system, HumanMessage(content=user_text)]),
            timeout=_OFFTOPIC_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("Off-topic composer failed; using deterministic fallback: %s", type(exc).__name__)
        return draft
    return _grounded_offtopic_reply(response.content, draft)


def _validated_model_choice(content: object, candidates: Sequence[str]) -> str | None:
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if not raw or "\n" in raw or len(raw) > 80:
        return None
    cleaned = raw.strip(" `*\"'.,")
    for candidate in candidates:
        if cleaned.casefold() == candidate.casefold():
            return candidate
    return None


def build_graph(model: BaseChatModel | None = None):
    """Build a LangGraph workflow with deterministic, source-grounded final rendering."""

    async def compose_response(state: ChatState) -> dict[str, object]:
        draft = state.get("draft_reply", "How can I help with your visit?")
        primary = state.get("intent", "general")
        topics = state.get("topics", [primary])
        extra_results = [
            _topic_result(topic, state)
            for topic in topics
            if topic != primary and topic not in {"greeting", "general"}
        ]
        if extra_results:
            extra_replies = [str(result["draft_reply"]) for result in extra_results]
            parts = list(dict.fromkeys([draft, *extra_replies]))
            output: dict[str, object] = {"messages": [AIMessage(content="\n\n".join(parts))]}
            secondary_names = [
                str(name)
                for result in extra_results
                for name in result.get("context_item_names", [])  # type: ignore[union-attr]
            ]
            if secondary_names:
                output["context_item_names"] = list(
                    dict.fromkeys([*state.get("context_item_names", []), *secondary_names])
                )
            secondary_categories = [
                str(result.get("context_category", "")) for result in extra_results
                if result.get("context_category")
            ]
            if secondary_categories:
                output["context_category"] = secondary_categories[-1]
            for result in extra_results:
                if "proposed_order_quantities" in result:
                    output["proposed_order_quantities"] = result["proposed_order_quantities"]
            return output

        # R-1: an off-topic message is the one case the model may compose freely,
        # to acknowledge the guest and steer back. Deterministic mode keeps the draft.
        if primary == "general" and state.get("off_topic") and model is not None:
            user_text = _last_user_text(state.get("messages", []))
            reply = await _compose_offtopic_reply(model, user_text, draft)
            return {"messages": [AIMessage(content=reply)]}

        candidates = state.get("candidate_names", [])
        if model is None or not state.get("use_model", False) or not candidates:
            return {"messages": [AIMessage(content=draft)]}

        candidate_lines = [
            format_item(item) for item in MENU if item.name in set(candidates)
        ]
        system = SystemMessage(
            content=(
                "Select exactly one item from the candidates. Return only the exact item name and no other text. "
                "Do not explain the choice.\nCandidates:\n" + "\n".join(candidate_lines)
            )
        )
        try:
            response = await model.ainvoke([system])
        except Exception as exc:
            logger.warning("Recommendation selector failed; using deterministic fallback: %s", type(exc).__name__)
            return {"messages": [AIMessage(content=draft)]}
        selected_name = _validated_model_choice(response.content, candidates)
        if selected_name is None:
            return {"messages": [AIMessage(content=draft)]}
        item = next(item for item in MENU if item.name == selected_name)
        preamble = state.get("recommendation_preamble", "Based on what you've told me, I'd suggest:")
        reply = f"{preamble}\n- {format_item(item)}"
        return {"messages": [AIMessage(content=reply)]}

    workflow = StateGraph(ChatState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("menu_info", menu_info)
    workflow.add_node("allergen_info", allergen_info)
    workflow.add_node("restaurant_info", restaurant_info)
    workflow.add_node("reservation_info", reservation_info)
    workflow.add_node("policy_info", policy_info)
    workflow.add_node("general_info", general_info)
    workflow.add_node("compose_response", compose_response)
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "menu_info": "menu_info",
            "allergen_info": "allergen_info",
            "restaurant_info": "restaurant_info",
            "reservation_info": "reservation_info",
            "policy_info": "policy_info",
            "general_info": "general_info",
        },
    )
    for node in (
        "menu_info", "allergen_info", "restaurant_info", "reservation_info", "policy_info", "general_info",
    ):
        workflow.add_edge(node, "compose_response")
    workflow.add_edge("compose_response", END)
    return workflow.compile()

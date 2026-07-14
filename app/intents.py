"""Intent detection: classify each guest turn into one or more topics."""

import re

from app.context import (
    ChatState,
    Intent,
    _allergy_emergency,
    _asks_about_kids_facilities,
    _has_restaurant_anchor,
    _is_beverage_request,
    _is_personal_child_reference,
    _last_user_text,
    _mentions_off_topic_subject,
    _named_menu_items,
    _parse_ingredient_query,
    _previous_user_texts,
    _prior_item_names,
    _requested_category,
)
from app.preferences import (
    TRACKED_ALLERGEN_ALIASES,
    UNTRACKED_ALLERGEN_ALIASES,
    UNVERIFIED_DIETARY_ALIASES,
    VERIFIED_DIETARY_ALIASES,
    contains_term,
    is_label_removal,
    is_restriction_statement,
    normalize,
    requested_labels,
)


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
            "mushrooms", "steak", "meat", "carnivore", "carnivorous", "option", "options",
            "vegitarian", "vegeterian", "dessets",
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
            "mushroom", "mushrooms", "steak", "meat", "carnivore", "carnivorous", "dessert", "desserts",
            "starter", "starters",
        }
        or any(
            phrase in normalized
            for phrase in (
                "show me options", "show options", "what do you serve", "what can i eat",
                "what can i have", "what can we eat", "anything i can eat", "what are my choices",
                "what works for me", "show me what i can eat", "tell me what i can eat",
                "what can i order",
                "meat options", "meat menu", "meat only", "only meat", "only eat meat", "avoid meat",
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

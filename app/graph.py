"""Answer composition and LangGraph wiring.

The classifier (app.intents) picks the topics; the nodes here turn each topic
into a grounded, source-verified reply, and ``build_graph`` assembles them into
the runnable workflow. Message/menu query primitives live in app.context; the
free-form off-topic redirect lives in app.offtopic.
"""

import logging
import re
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.context import (
    _UNNAMED_ALLERGEN,
    ChatState,
    Intent,
    _allergy_emergency,
    _description_lists_ingredient,
    _effective_preferences,
    _has_restaurant_anchor,
    _ingredient_matches,
    _is_menu_item_word,
    _last_user_text,
    _mentions_off_topic_subject,
    _named_menu_items,
    _parse_ingredient_query,
    _prior_category,
    _prior_item_names,
    _requested_category,
)
from app.intents import (
    _is_allergen_request,
    _is_explicit_allergen_content_request,
    _is_follow_up,
    _is_recommendation_request,
    classify_intent,
)
from app.offtopic import (
    _compose_offtopic_reply,
    _validated_model_choice,
)
from app.preferences import (
    TRACKED_ALLERGEN_ALIASES,
    UNTRACKED_ALLERGEN_ALIASES,
    UNVERIFIED_DIETARY_ALIASES,
    VERIFIED_DIETARY_ALIASES,
    contains_term,
    is_label_removal,
    natural_join,
    normalize,
    requested_labels,
)
from app.rag import render_preset, retrieve_preset
from app.restaurant import (
    ITEM_ALIASES,
    MENU,
    RESTAURANT,
    MenuItem,
    format_item,
)

logger = logging.getLogger(__name__)


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
    return {"draft_reply": reply}


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
    return {"draft_reply": reply}


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
    return {"draft_reply": reply}


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


def _unverified_menu_note(labels: Sequence[str]) -> str:
    """Explain why a query-level menu label is not a safety certification."""
    return (
        f"\nThese options are marked by the menu as requested, but the saved {natural_join(labels)} "
        "restriction is not verified by the menu data. Please call "
        f"{RESTAURANT['phone']} before ordering; staff must confirm preparation, sourcing, and certification."
    )


def _meat_only_menu_note() -> str:
    """Explain the limit of a non-vegetarian/meat-only menu filter."""
    return (
        "\nThese items are not marked vegetarian, but the menu does not define or verify a meat-only requirement. "
        f"Please call {RESTAURANT['phone']} before ordering if that distinction is safety-critical."
    )


def _is_meat_only_request(text: str) -> bool:
    """Recognize meat-focused menu requests without treating them as verified."""
    normalized = normalize(text)
    token_set = set(normalized.split())
    if any(
        phrase in normalized
        for phrase in (
            "do not eat meat", "do not have meat", "cannot eat meat", "cannot have meat",
            "do not only eat meat", "cannot only eat meat", "not meat", "not a meat eater",
            "not carnivore", "not a carnivore", "not carnivorous", "not a carnivorous",
            "avoid meat", "no meat", "without meat",
            "meatless", "meat free", "vegetarian", "vegan",
        )
    ):
        return False
    if "meat-only" in requested_labels(text, UNVERIFIED_DIETARY_ALIASES):
        return True
    if not ("meat" in token_set or token_set & {"carnivore", "carnivorous"}):
        return False
    return bool(
        token_set
        & {"option", "options", "menu", "menus", "eat", "eating", "have", "only", "dish", "dishes"}
    ) or normalized in {"meat", "i eat meat", "we eat meat", "i can eat meat", "we can eat meat"}


def _filter_menu_items(state: ChatState, text: str) -> list[MenuItem]:
    preferences = _effective_preferences(state)
    items = list(MENU)
    current_meat_only = _is_meat_only_request(text)
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
    if current_meat_only:
        # A direct meat query is a query-level override of a saved vegetarian
        # display preference; it still carries an explicit limitation note.
        dietary = [label for label in dietary if label not in {"vegan", "vegetarian"}]
    dietary = list(dict.fromkeys([*dietary, *current_positive]))
    if "vegan" in dietary:
        items = [item for item in items if item.vegan]
    elif "vegetarian" in dietary:
        items = [item for item in items if item.vegetarian]
    if "gluten-free" in dietary:
        items = [item for item in items if item.gluten_free]
    if current_meat_only:
        items = [item for item in items if not item.vegetarian]
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
        and not (label == "meat-only" and not _is_meat_only_request(text))
    ]
    unverified = list(dict.fromkeys([*preferences.unverified_diets, *current_unverified]))
    current_dietary, current_negative_dietary = _current_dietary_filters(text)
    current_meat_only = _is_meat_only_request(text)
    query_dietary_filter = bool(current_dietary or current_meat_only)
    active_meat_only = current_meat_only or "meat-only" in preferences.unverified_diets
    other_unverified = [label for label in unverified if label != "meat-only"]
    # A current, explicit menu label (for example, "vegetarian menu") is a
    # useful query filter even when an older unverified certification remains
    # saved. We show the matching labels with a clear warning rather than
    # claiming that they satisfy that certification.
    unverified_menu_note = (
        _unverified_menu_note(other_unverified)
        if other_unverified and query_dietary_filter
        else ""
    )
    meat_only_menu_note = _meat_only_menu_note() if active_meat_only and query_dietary_filter else ""
    if preferences.untracked_allergens:
        labels = natural_join(preferences.untracked_allergens)
        reply = (
            f"The current menu data does not track {labels} separately, so I can’t confirm an option that meets "
            f"all restrictions. Please call {RESTAURANT['phone']} and discuss cross-contact risk with staff."
        )
        return {"draft_reply": reply}
    if unverified and not query_dietary_filter:
        labels = natural_join(unverified)
        reply = (
            f"The current menu does not identify any dishes as {labels}, so I can’t confirm an option that meets "
            f"all of those restrictions. Please call {RESTAURANT['phone']}; ingredients alone do not establish preparation, "
            "nutrition targets, sourcing, or certification."
        )
        return {"draft_reply": reply}

    generic_dietary = bool(token_set & {"diet", "dietary", "preference", "preferences", "restriction", "restrictions"})
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
        return {"draft_reply": reply}
    if generic_dietary and not current_dietary and not current_meat_only:
        preset = retrieve_preset(text)
        if preset is not None and preset.kind in {
            "dietary-capabilities", "vegetarian-menu", "vegan-menu", "meat-only-menu",
        }:
            return {
                "draft_reply": render_preset(preset),
                "semantic_preset_id": preset.id,
            }
        reply = (
            "Of course — tell me every dietary preference or restriction that applies; you can list more than one. "
            "I can filter verified vegan, "
            "vegetarian, and gluten-free labels and check declared dairy, egg, fish, gluten, and tree-nut allergens. "
            "For halal, kosher, keto, pescatarian, paleo, meat-only, and other unverified requirements, staff must confirm."
        )
        return {"draft_reply": reply}

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
            "draft_reply": reply + unverified_menu_note + meat_only_menu_note,
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
                "draft_reply": reply,
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
            "draft_reply": reply,
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
        return {"draft_reply": reply}

    if not items:
        reply = "I couldn’t find a listed item matching all of those filters. Restaurant staff can confirm whether another option or modification is possible."
        return {"draft_reply": reply}

    if "cheapest" in token_set and items:
        cheapest = min(items, key=lambda item: item.price)
        reply = f"The lowest listed price among those options is {cheapest.name} at ${cheapest.price:.0f}."
        return {
            "draft_reply": reply + unverified_menu_note + meat_only_menu_note,
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
            "draft_reply": reply + unverified_menu_note + meat_only_menu_note,
            "context_item_names": [item.name],
            "context_category": item.category,
        }

    order = _order_update(state, text, order_items)
    if order:
        order_reply, quantities = order
        return {
            "draft_reply": order_reply,
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
        draft += unverified_menu_note + meat_only_menu_note
        return {
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
    reply += unverified_menu_note + meat_only_menu_note
    return {
        "draft_reply": reply,
        "context_item_names": [item.name for item in selected],
        "context_category": category or "",
    }


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
        return {"draft_reply": reply}

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
        return {"draft_reply": reply}

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
            "draft_reply": reply,
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
            "draft_reply": reply,
            "context_item_names": [item.name for item in named],
        }

    if content_question and current_untracked and not current_tracked:
        display = natural_join(current_untracked)
        reply = (
            f"The menu data does not track {display} separately, so I can’t confirm whether the requested item "
            f"contains it. Please call {RESTAURANT['phone']} and discuss cross-contact risk with staff."
        )
        return {"draft_reply": reply}

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
            "draft_reply": reply,
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
            "draft_reply": reply,
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
            return {"draft_reply": reply}
        display = natural_join(named_untracked)
        reply = (
            f"The current menu data does not track {display} separately, so I can’t identify a safe option. "
            f"Please call {RESTAURANT['phone']} before ordering and tell staff about every allergy and "
            "cross-contact concern."
        )
        return {"draft_reply": reply}

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
        return {"draft_reply": reply}
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
            "draft_reply": reply,
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
            "draft_reply": reply,
            "context_item_names": [item.name for item in items[:4]],
        }

    reply = (
        "Please name every allergy or intolerance that applies. The menu data declares dairy, egg, fish, gluten, "
        "and tree nuts for individual items, but it cannot guarantee allergen safety or prevent cross-contact."
    )
    return {"draft_reply": reply}


def general_info(state: ChatState) -> dict[str, object]:
    text = normalize(_last_user_text(state.get("messages", [])))
    has_proposed_order = bool(state.get("proposed_order_quantities"))
    off_topic = bool(state.get("off_topic")) or (
        _mentions_off_topic_subject(text) and not _has_restaurant_anchor(text)
    )
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
    elif off_topic:
        reply = (
            f"I’m the {RESTAURANT['name']} restaurant assistant. I can help with our menu, "
            "dietary preferences, hours, location, and reservation information."
        )
    else:
        preset = None if state.get("off_topic") else retrieve_preset(text)
        if preset is not None:
            return {
                "draft_reply": render_preset(preset),
                "off_topic": False,
                "semantic_preset_id": preset.id,
            }
        reply = (
            f"I’m the {RESTAURANT['name']} restaurant assistant. I can help with our menu, "
            "dietary preferences, hours, location, and reservation information."
        )
        off_topic = True
    return {"draft_reply": reply, "off_topic": off_topic}


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


def build_graph(
    model: BaseChatModel | None = None,
    *,
    offtopic_model: str = "",
):
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
            reply = await _compose_offtopic_reply(model, user_text, draft, offtopic_model)
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

"""Off-topic conversational redirect and model-output safety boundaries.

This is the one place the model may compose free-form text, so its output is
bounded hard (no leaked internals, no invented facts, must steer back). The
recommendation path's model-choice validators live here too, since both concern
turning raw model output into something safe to show a guest.
"""

import asyncio
import logging
import re
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.knowledge import persona_offtopic
from app.restaurant import ITEM_ALIASES, RESTAURANT

logger = logging.getLogger(__name__)


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
    # Leaked internals / reasoning.
    "system prompt", "instruction", "internal", "verified facts", "fallback", "as an ai",
    "language model", "output contract", "the guest said", "the user said", "the user just",
    "analysis:", "reasoning:", "i cannot fulfill", "i can't fulfill",
    # Menu-content claims or table-service promises the assistant cannot make.
    "on our menu", "on the menu", "part of our menu", "part of the menu", "bring you",
    "i'll bring", "i will bring", "i can bring", "we serve", "glass of", "we offer you a",
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
    """Pick the model for the one free-form off-topic call.

    Prefer a lightweight non-reasoning Ollama model when ``OLLAMA_OFFTOPIC_MODEL``
    is configured (reliable, fast, no chain-of-thought to leak). Otherwise coax a
    reasoning primary model into using its reasoning channel so its thoughts do
    not bleed into the reply. Non-Ollama providers are used unchanged.
    """
    if not (hasattr(model, "reasoning") and hasattr(model, "num_predict")):
        return model
    try:
        override = get_settings().ollama_offtopic_model.strip()
        if override and hasattr(model, "model"):
            return model.model_copy(
                update={"model": override, "reasoning": False, "num_predict": 256, "temperature": 0.7}
            )
        return model.model_copy(update={"reasoning": True, "num_predict": _OFFTOPIC_NUM_PREDICT})
    except Exception:  # pragma: no cover - defensive; provider without model_copy
        return model


async def _compose_offtopic_reply(model: BaseChatModel, user_text: str, draft: str) -> str:
    """Acknowledge an off-topic message in one playful clause, then steer back — bounded and fact-free."""
    system = SystemMessage(
        content=(
            f"You are the assistant for {RESTAURANT['name']}, a restaurant. The guest said something off-topic. "
            "Reply in ONE short, warm, playful sentence that FIRST reacts to their exact words with a specific "
            "nod, THEN pivots to helping with the menu, dietary needs, hours, or a reservation. "
            'Example — guest: "meow" -> "Aw, sounds like we have a cat in the house! I can only fetch menus and '
            'book tables though — what are you hungry for?" '
            "Do not answer their question. Invent no facts — no dishes, prices, hours, addresses, or phone "
            "numbers. Under 35 words. Reply only, no quotation marks.\n\n"
            f"Tone reference (do not copy verbatim):\n{persona_offtopic()}"
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
    """Return the model's pick only if it exactly matches one offered candidate name."""
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

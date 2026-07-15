import hashlib
from dataclasses import dataclass
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.bookings import BookingLog, is_valid_draft
from app.config import Settings
from app.graph import build_graph
from app.preferences import merge_preferences
from app.restaurant import MENU, MENU_CATEGORIES
from app.sessions import (
    CachedTurn,
    InMemorySessionStore,
    RequestConflictError,
    SessionExpiredError,
    SessionLimitError,
    SessionRecord,
)

_MENU_ITEM_NAMES = frozenset(item.name for item in MENU)
_VALID_INTENTS = frozenset(
    {"greeting", "menu", "recommendation", "hours_location", "reservation", "allergens", "policy", "general"}
)
_VALID_CATEGORIES = frozenset(MENU_CATEGORIES)


def _validate_proposed_order_quantities(value: object) -> dict[str, int]:
    """Return a defensive copy of canonical, positive proposed-order quantities."""
    if not isinstance(value, dict):
        raise RuntimeError("chat graph produced invalid proposed-order state")
    if len(value) > len(_MENU_ITEM_NAMES):
        raise RuntimeError("chat graph produced invalid proposed-order state")

    validated: dict[str, int] = {}
    for name, quantity in value.items():
        if type(name) is not str or name not in _MENU_ITEM_NAMES:
            raise RuntimeError("chat graph produced invalid proposed-order state")
        if type(quantity) is not int or quantity <= 0:
            raise RuntimeError("chat graph produced invalid proposed-order state")
        validated[name] = quantity
    return validated


def _validate_intent(value: object) -> str:
    if type(value) is not str or value not in _VALID_INTENTS:
        raise RuntimeError("chat graph produced invalid intent metadata")
    return value


def _validate_context_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > len(_MENU_ITEM_NAMES):
        raise RuntimeError("chat graph produced invalid item context")
    if any(type(name) is not str or name not in _MENU_ITEM_NAMES for name in value):
        raise RuntimeError("chat graph produced invalid item context")
    return tuple(dict.fromkeys(value))


def _validate_context_category(value: object) -> str | None:
    if value in (None, ""):
        return None
    if type(value) is not str or value not in _VALID_CATEGORIES:
        raise RuntimeError("chat graph produced invalid category context")
    return value


def _validate_booking_draft(value: object) -> dict[str, str]:
    if not is_valid_draft(value):
        raise RuntimeError("chat graph produced invalid booking-draft state")
    return dict(value)  # type: ignore[arg-type]


@dataclass(slots=True)
class TurnResult:
    session: SessionRecord
    response: str
    intent: str
    turns_used: int


class ChatService:
    def __init__(
        self,
        settings: Settings,
        store: InMemorySessionStore,
        model: BaseChatModel | None = None,
        chat_mode: str = "deterministic",
        booking_log: BookingLog | None = None,
    ):
        self.settings = settings
        self.store = store
        self.booking_log = booking_log or BookingLog(settings.bookings_csv_path)
        self.graph = build_graph(
            model,
            offtopic_model=settings.ollama_offtopic_model,
            enable_offtopic_model=settings.ollama_offtopic_enabled,
            booking_log=self.booking_log,
        )
        self.chat_mode = chat_mode

    async def create_session(self) -> SessionRecord:
        return await self.store.create()

    async def chat(
        self,
        message: str,
        session_id: UUID | None = None,
        request_id: UUID | None = None,
    ) -> TurnResult:
        session = await self.store.get_or_create(session_id)
        anonymous_session = session_id is None
        try:
            return await self._chat_existing_session(
                session=session,
                message=message,
                request_id=request_id,
            )
        except BaseException:
            if anonymous_session:
                await self.store.discard(session)
            raise

    async def _chat_existing_session(
        self,
        *,
        session: SessionRecord,
        message: str,
        request_id: UUID | None,
    ) -> TurnResult:
        message_fingerprint = hashlib.sha256(message.encode("utf-8")).hexdigest()

        async with session.lock:
            await self.store.ensure_current(session)
            if self.store.is_expired(session):
                raise SessionExpiredError(str(session.session_id))

            if request_id is not None and request_id in session.request_cache:
                cached = session.request_cache[request_id]
                if cached.message_fingerprint != message_fingerprint:
                    raise RequestConflictError(str(request_id))
                session.request_cache.move_to_end(request_id)
                await self.store.touch(session)
                return TurnResult(session, cached.response, cached.intent, session.turns_used)

            if session.turns_used >= self.settings.max_turns_per_session:
                raise SessionLimitError(str(session.session_id))

            # All production follow-up context is carried in the structured
            # fields below. Passing the complete transcript through LangGraph
            # only caused repeated message copies and reducer work; the current
            # user turn is sufficient here.
            input_messages = [HumanMessage(content=message)]
            pending_preferences = merge_preferences(
                message,
                dietary=sorted(session.dietary_preferences),
                allergens=sorted(session.allergen_restrictions),
                untracked_allergens=sorted(session.untracked_allergen_restrictions),
                unverified_diets=sorted(session.unverified_dietary_restrictions),
                allergen_context=session.last_intent == "allergens",
            )
            graph_input = {
                "messages": input_messages,
                **pending_preferences.as_graph_input(),
                "prior_intent": session.last_intent or "",
                "prior_item_names": list(session.last_item_names),
                "prior_category": session.last_category or "",
                "proposed_order_quantities": dict(session.proposed_order_quantities),
                "booking_draft": dict(session.booking_draft),
            }
            result = await self.graph.ainvoke(graph_input)
            messages = result.get("messages", [])
            if not isinstance(messages, list) or len(messages) <= len(input_messages) or not isinstance(messages[-1], AIMessage):
                raise RuntimeError("chat graph did not produce a response")
            response_message = messages[-1]
            response = response_message.content
            if not isinstance(response, str) or not response.strip():
                raise RuntimeError("chat graph produced a non-text or empty response")
            pending_order_quantities = (
                _validate_proposed_order_quantities(result["proposed_order_quantities"])
                if "proposed_order_quantities" in result
                else dict(session.proposed_order_quantities)
            )
            pending_booking_draft = (
                _validate_booking_draft(result["booking_draft"])
                if "booking_draft" in result
                else dict(session.booking_draft)
            )
            result_intent = _validate_intent(result.get("intent", "general"))
            context_items = _validate_context_items(result.get("context_item_names", []))
            context_category = _validate_context_category(result.get("context_category"))

            pending_last_intent = session.last_intent
            if result_intent not in {"general", "greeting"}:
                pending_last_intent = result_intent
            pending_last_items = session.last_item_names
            pending_last_category = session.last_category
            if context_items:
                pending_last_items = context_items
            if context_category:
                pending_last_category = context_category
            explicit_menu_context = bool(context_items or context_category)
            if result_intent not in {"menu", "recommendation", "allergens"} and not explicit_menu_context:
                pending_last_items = ()
                pending_last_category = None

            safety_context_changed = (
                set(pending_preferences.dietary) != session.dietary_preferences
                or set(pending_preferences.allergens) != session.allergen_restrictions
                or set(pending_preferences.untracked_allergens) != session.untracked_allergen_restrictions
                or set(pending_preferences.unverified_diets) != session.unverified_dietary_restrictions
            )

            # Commit only after every graph output field is validated. Rebuild the
            # transcript from trusted user/assistant text so a graph cannot mutate
            # prior message objects or persist hidden intermediate messages.
            session.messages = [
                *session.messages,
                HumanMessage(content=message),
                AIMessage(content=response),
            ][-self.settings.max_history_messages :]
            session.dietary_preferences = set(pending_preferences.dietary)
            session.allergen_restrictions = set(pending_preferences.allergens)
            session.untracked_allergen_restrictions = set(pending_preferences.untracked_allergens)
            session.unverified_dietary_restrictions = set(pending_preferences.unverified_diets)
            session.proposed_order_quantities = pending_order_quantities
            session.booking_draft = pending_booking_draft
            session.last_intent = pending_last_intent
            session.last_item_names = pending_last_items
            session.last_category = pending_last_category
            session.turns_used += 1
            await self.store.touch(session)

            if safety_context_changed:
                session.request_cache.clear()
            if request_id is not None:
                session.request_cache[request_id] = CachedTurn(
                    message_fingerprint=message_fingerprint,
                    response=response,
                    intent=result_intent,
                )
                session.request_cache.move_to_end(request_id)
                while len(session.request_cache) > self.settings.max_turns_per_session:
                    session.request_cache.popitem(last=False)

            return TurnResult(
                session=session,
                response=response,
                intent=result_intent,
                turns_used=session.turns_used,
            )

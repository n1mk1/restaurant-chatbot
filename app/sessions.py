import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from langchain_core.messages import AnyMessage


class SessionNotFoundError(Exception):
    pass


class SessionExpiredError(Exception):
    pass


class SessionLimitError(Exception):
    pass


class SessionCapacityError(Exception):
    pass


class RequestConflictError(Exception):
    pass


@dataclass(slots=True)
class CachedTurn:
    message_fingerprint: str
    response: str
    intent: str


@dataclass(slots=True)
class SessionRecord:
    session_id: UUID
    created_at: datetime
    last_activity: datetime
    messages: list[AnyMessage] = field(default_factory=list)
    turns_used: int = 0
    # Keep safety-critical preferences outside the trimmed transcript. A food
    # restriction must not disappear merely because the chat history reached
    # max_history_messages.
    dietary_preferences: set[str] = field(default_factory=set)
    allergen_restrictions: set[str] = field(default_factory=set)
    untracked_allergen_restrictions: set[str] = field(default_factory=set)
    unverified_dietary_restrictions: set[str] = field(default_factory=set)
    last_intent: str | None = None
    last_item_names: tuple[str, ...] = ()
    last_category: str | None = None
    # Canonical menu item name -> positive quantity for the active, unsubmitted order.
    proposed_order_quantities: dict[str, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_cache: OrderedDict[UUID, CachedTurn] = field(default_factory=OrderedDict)


class InMemorySessionStore:
    """A bounded, process-local session store for development and small deployments."""

    def __init__(self, ttl_seconds: float, max_sessions: int):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_sessions = max_sessions
        self._sessions: dict[UUID, SessionRecord] = {}
        self._expired: OrderedDict[UUID, None] = OrderedDict()
        self._lock = asyncio.Lock()

    def now(self) -> datetime:
        return datetime.now(UTC)

    def expires_at(self, session: SessionRecord) -> datetime:
        return session.last_activity + self.ttl

    def is_expired(self, session: SessionRecord) -> bool:
        return self.now() >= self.expires_at(session)

    async def _prune_locked(self) -> None:
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if self.is_expired(session) and not session.lock.locked()
        ]
        for session_id in expired_ids:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.proposed_order_quantities.clear()
            self._remember_expired(session_id)

    def _remember_expired(self, session_id: UUID) -> None:
        self._expired[session_id] = None
        self._expired.move_to_end(session_id)
        while len(self._expired) > self.max_sessions:
            self._expired.popitem(last=False)

    async def create(self) -> SessionRecord:
        async with self._lock:
            await self._prune_locked()
            if len(self._sessions) >= self.max_sessions:
                raise SessionCapacityError("active session capacity reached")
            now = self.now()
            session = SessionRecord(session_id=uuid4(), created_at=now, last_activity=now)
            self._sessions[session.session_id] = session
            return session

    async def get(self, session_id: UUID) -> SessionRecord:
        async with self._lock:
            await self._prune_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                return session
            if session_id in self._expired:
                raise SessionExpiredError(str(session_id))
            raise SessionNotFoundError(str(session_id))

    async def get_or_create(self, session_id: UUID | None) -> SessionRecord:
        return await self.create() if session_id is None else await self.get(session_id)

    async def discard(self, session: SessionRecord) -> None:
        """Remove a newly-created session whose anonymous first turn failed."""
        async with self._lock:
            if self._sessions.get(session.session_id) is session:
                self._sessions.pop(session.session_id, None)
                session.messages.clear()
                session.dietary_preferences.clear()
                session.allergen_restrictions.clear()
                session.untracked_allergen_restrictions.clear()
                session.unverified_dietary_restrictions.clear()
                session.proposed_order_quantities.clear()
                session.request_cache.clear()

    async def ensure_current(self, session: SessionRecord) -> None:
        """Confirm a fetched record was not deleted while its caller waited for the record lock."""
        async with self._lock:
            current = self._sessions.get(session.session_id)
            if current is session:
                return
            if session.session_id in self._expired:
                raise SessionExpiredError(str(session.session_id))
            raise SessionNotFoundError(str(session.session_id))

    async def delete(self, session_id: UUID) -> None:
        async with self._lock:
            await self._prune_locked()
            session = self._sessions.get(session_id)
            if session is None:
                if session_id in self._expired:
                    raise SessionExpiredError(str(session_id))
                raise SessionNotFoundError(str(session_id))

        async with session.lock:
            async with self._lock:
                current = self._sessions.get(session_id)
                if current is not session:
                    if session_id in self._expired:
                        raise SessionExpiredError(str(session_id))
                    raise SessionNotFoundError(str(session_id))
                if self.is_expired(session):
                    self._sessions.pop(session_id, None)
                    session.proposed_order_quantities.clear()
                    self._remember_expired(session_id)
                    raise SessionExpiredError(str(session_id))
                self._sessions.pop(session_id, None)
                session.proposed_order_quantities.clear()
                self._expired.pop(session_id, None)

    async def touch(self, session: SessionRecord) -> None:
        session.last_activity = self.now()

    async def count_active(self) -> int:
        async with self._lock:
            await self._prune_locked()
            return len(self._sessions)

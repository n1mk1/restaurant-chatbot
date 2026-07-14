"""Chat time-slot booking: slots, turn parsing, and the CSV booking log.

The flow is deterministic and explicit-consent-only: the guest supplies a day,
an hourly time slot, a name, and a phone number (in any order, across any
number of turns); the assistant echoes ``confirm time slot <day> <time> for
<name> - <phone>``; and only a literal "confirm" reply appends the record to
the CSV booking log. "cancel" discards the pending request at any point.
Slots are derived from ``RESTAURANT["hours"]`` — hourly, from opening until
one hour before close — so a menu-data change to hours changes the bookable
slots without touching this module.
"""

from __future__ import annotations

import asyncio
import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from app.preferences import natural_join, normalize
from app.restaurant import RESTAURANT

WEEKDAYS: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)
_FIELDS = ("day", "time", "name", "phone")
_FIELD_LABELS = {
    "day": "the day of the week",
    "time": "a time slot",
    "name": "a name for the booking",
    "phone": "a contact phone number",
}

_HOURS_RANGE_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(AM|PM)\s*[–-]\s*(\d{1,2}):(\d{2})\s*(AM|PM)"
)

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'’\-]{0,59}$")
PHONE_RE = re.compile(r"^\+?\d{7,15}$")
SLOT_RE = re.compile(r"^\d{1,2}:00 [AP]M$")


def _hour_24(hour: int, meridiem: str) -> int:
    hour = hour % 12
    return hour + 12 if meridiem.upper() == "PM" else hour


def _format_slot(hour24: int) -> str:
    meridiem = "PM" if hour24 >= 12 else "AM"
    hour = hour24 % 12 or 12
    return f"{hour}:00 {meridiem}"


@cache
def _day_hours() -> dict[str, str]:
    """Expand grouped RESTAURANT hour keys ("Tuesday–Thursday") per weekday."""
    expanded: dict[str, str] = {}
    for key, value in RESTAURANT["hours"].items():
        parts = [part.strip() for part in re.split(r"[–-]", key)]
        if len(parts) == 2 and parts[0] in WEEKDAYS and parts[1] in WEEKDAYS:
            start, end = WEEKDAYS.index(parts[0]), WEEKDAYS.index(parts[1])
            for day in WEEKDAYS[start : end + 1]:
                expanded[day] = value
        elif parts[0] in WEEKDAYS:
            expanded[parts[0]] = value
    return expanded


@cache
def time_slots(day: str) -> tuple[str, ...]:
    """Hourly bookable slots for a weekday; empty when the restaurant is closed."""
    match = _HOURS_RANGE_RE.search(_day_hours().get(day, ""))
    if not match:
        return ()
    opening = _hour_24(int(match.group(1)), match.group(3))
    closing = _hour_24(int(match.group(4)), match.group(6))
    return tuple(_format_slot(hour) for hour in range(opening, closing))


@cache
def _all_slots() -> frozenset[str]:
    return frozenset(slot for day in WEEKDAYS for slot in time_slots(day))


def slot_overview() -> str:
    """One-line summary of the weekly bookable slots, grouped like the source hours."""
    groups: list[str] = []
    for key, value in RESTAURANT["hours"].items():
        match = _HOURS_RANGE_RE.search(value)
        if not match:
            continue
        first_day = re.split(r"[–-]", key)[0].strip()
        slots = time_slots(first_day)
        groups.append(f"{key} {slots[0]}–{slots[-1]}")
    return "; ".join(groups) + " (on the hour; closed Mondays)"


# --- Turn parsing -----------------------------------------------------------

_DAY_ALIASES: dict[str, str] = {
    alias: day
    for day, aliases in {
        "Monday": ("monday", "mondays", "mon"),
        "Tuesday": ("tuesday", "tuesdays", "tue", "tues"),
        "Wednesday": ("wednesday", "wednesdays", "wed"),
        "Thursday": ("thursday", "thursdays", "thu", "thur", "thurs"),
        "Friday": ("friday", "fridays", "fri"),
        "Saturday": ("saturday", "saturdays", "sat"),
        "Sunday": ("sunday", "sundays", "sun"),
    }.items()
    for alias in aliases
}

_PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s().\-]{5,}\d")
_AMPM_TIME_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_OCLOCK_TIME_RE = re.compile(r"\b(\d{1,2})\s*o\s*'?\s*clock\b", re.IGNORECASE)
_AT_TIME_RE = re.compile(r"\b(?:at|around)\s+(\d{1,2})(?:[:.](\d{2}))?\b", re.IGNORECASE)
_CLOCK_TIME_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})\b")
_BARE_TIME_RE = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*[.!]?\s*$")

_NAME_PATTERN_RE = re.compile(
    r"(?:\bname\s+is\b|\bname\s*:|\bunder\s+the\s+name\b|\bunder\b|"
    r"\bbooking\s+for\b|\breservation\s+for\b|\bfor\b)\s+([A-Za-z][A-Za-z .'’\-]{0,79})"
)
_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z.'’\-]*$")
_NAME_STOPWORDS = frozenset(
    {
        *(_DAY_ALIASES),
        "a", "an", "and", "am", "around", "at", "book", "booking", "cancel", "confirm",
        "dinner", "eight", "five", "four", "guest", "guests", "later", "lunch", "me",
        "menu", "my", "nine", "noon", "on", "one", "our", "people", "person", "phone",
        "please", "pm", "reservation", "seven", "six", "slot", "table", "ten", "the",
        "them", "then", "three", "time", "today", "tomorrow", "tonight", "two", "us",
    }
)

_CONFIRM_REPLIES = frozenset(
    {
        "confirm", "confirmed", "i confirm", "yes confirm", "yes i confirm",
        "confirm it", "confirm booking", "confirm the booking", "confirm the slot",
        "confirm time slot", "confirm please", "please confirm",
    }
)
_CANCEL_REPLIES = frozenset(
    {
        "cancel", "cancel it", "cancel that", "cancel booking", "cancel the booking",
        "cancel my booking", "never mind", "nevermind", "forget it", "no cancel",
        "stop", "cancel please", "please cancel",
    }
)
_START_PHRASES = (
    "book a table", "book a slot", "book a time", "book me", "book us", "book for",
    "reserve a table", "reserve a slot", "reserve a spot", "reserve for", "reserve us",
    "make a reservation", "make a reservaton", "make a booking", "make a res",
    "can i book", "can we book", "could i book", "could we book",
    "can i reserve", "can we reserve", "could i reserve", "could we reserve",
    "like to book", "want to book", "like to reserve", "want to reserve",
    "get a table", "need a table", "want a table",
)


def is_booking_confirmation(text: str) -> bool:
    return normalize(text) in _CONFIRM_REPLIES


def is_booking_cancellation(text: str) -> bool:
    return normalize(text) in _CANCEL_REPLIES


def wants_slot_booking(text: str) -> bool:
    normalized = normalize(text)
    return any(phrase in normalized for phrase in _START_PHRASES)


def parse_day(text: str) -> str | None:
    for token in normalize(text).split():
        if token in _DAY_ALIASES:
            return _DAY_ALIASES[token]
    return None


def parse_phone(raw_text: str) -> str | None:
    phone, _ = _extract_phone(raw_text)
    return phone


def _extract_phone(raw_text: str) -> tuple[str | None, str]:
    """Pull the first plausible phone number out of the raw text.

    Returns the canonical number and the text with that span removed, so digit
    runs inside the number are never re-parsed as a time expression.
    """
    for match in _PHONE_CANDIDATE_RE.finditer(raw_text):
        digits = re.sub(r"\D", "", match.group())
        if 7 <= len(digits) <= 15:
            prefix = "+" if match.group().startswith("+") else ""
            remainder = raw_text[: match.start()] + " " + raw_text[match.end() :]
            return prefix + digits, remainder
    return None, raw_text


def _assume_dinner(hour: int) -> int:
    """Map a bare 1–11 to the evening; dinner-only service has no morning slots."""
    return hour + 12 if 1 <= hour <= 11 else hour


def parse_time_expression(raw_text: str) -> tuple[int, int] | None:
    """(hour24, minute) from "7pm" / "7:00 PM" / "at 7" / "19:00" style text."""
    match = _AMPM_TIME_RE.search(raw_text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if 1 <= hour <= 12 and minute < 60:
            hour24 = _hour_24(hour, match.group(3))
            return hour24, minute
        return None
    match = _OCLOCK_TIME_RE.search(raw_text)
    if match:
        hour = int(match.group(1))
        if 1 <= hour <= 12:
            return _assume_dinner(hour), 0
        return None
    match = _AT_TIME_RE.search(raw_text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if 1 <= hour <= 23 and minute < 60:
            return _assume_dinner(hour), minute
        return None
    match = _CLOCK_TIME_RE.search(raw_text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if 1 <= hour <= 23 and minute < 60:
            return _assume_dinner(hour), minute
    return None


def _sanitize_name(candidate: str) -> str:
    return " ".join(candidate.split())[:60].strip()


def _valid_name_tokens(candidate: str) -> str | None:
    tokens = candidate.split()
    kept: list[str] = []
    for token in tokens:
        cleaned = token.rstrip(".,!?")
        if not cleaned or normalize(cleaned) in _NAME_STOPWORDS:
            break
        if not _NAME_TOKEN_RE.match(cleaned):
            return None
        kept.append(cleaned)
        if len(kept) == 5:
            break
    if not kept:
        return None
    return _sanitize_name(" ".join(kept))


def _extract_name(raw_text: str) -> str | None:
    match = _NAME_PATTERN_RE.search(raw_text)
    if not match:
        return None
    return _valid_name_tokens(match.group(1))


def _whole_message_name(raw_text: str) -> str | None:
    stripped = raw_text.strip()
    if not stripped or "?" in stripped or any(char.isdigit() for char in stripped):
        return None
    if len(stripped.split()) > 5:
        return None
    name = _valid_name_tokens(stripped)
    if name is None or len(name.split()) != len(stripped.split()):
        return None
    return name


# --- The booking turn -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookingTurn:
    """Outcome of one guest turn inside the booking flow.

    ``record`` is set only when the guest replied "confirm" with a complete
    draft; the caller performs the CSV append and composes the final reply.
    """

    reply: str
    draft: dict[str, str]
    record: dict[str, str] | None = None


def _missing_fields(draft: dict[str, str]) -> list[str]:
    return [name for name in _FIELDS if not draft.get(name)]


def _booking_alternatives() -> str:
    return (
        f"You can also book at {RESTAURANT['reservation_url']} or call {RESTAURANT['phone']}."
    )


def confirmation_echo(draft: dict[str, str]) -> str:
    return (
        f"Please review: confirm time slot {draft['day']} {draft['time']} for "
        f"{draft['name']} - {draft['phone']}. Reply \"confirm\" to record the booking, "
        "or \"cancel\" to discard it."
    )


def _prompt_for(draft: dict[str, str], notes: list[str], *, initial: bool, parsed_any: bool) -> str:
    missing = _missing_fields(draft)
    if not missing:
        prompt = confirmation_echo(draft)
    elif "day" in missing and "time" in missing:
        prompt = f"Which day and time would you like? Slots are hourly: {slot_overview()}."
    elif "day" in missing:
        prompt = (
            f"Which day should I book {draft['time']} for? Chat bookings run Tuesday to Sunday."
        )
    elif "time" in missing:
        prompt = (
            f"What time on {draft['day']}? Bookable {draft['day']} slots: "
            f"{', '.join(time_slots(draft['day']))}."
        )
    elif "name" in missing:
        prompt = "What name should the booking be under?"
    else:
        prompt = f"Thanks, {draft['name']}. What phone number should we use for the booking?"

    parts: list[str] = []
    if initial:
        parts.append(
            "I can’t see live table availability, but I can record a chat booking in "
            "Maple & Ember’s booking log."
        )
    if notes:
        parts.extend(notes)
    elif not parsed_any and not initial and missing:
        parts.append("Sorry, I didn’t catch that.")
    parts.append(prompt)
    if initial:
        parts.append(_booking_alternatives())
    return " ".join(parts)


def booking_turn(raw_text: str, draft: dict[str, str], *, prompted: bool = False) -> BookingTurn | None:
    """Advance the slot-booking flow by one guest turn.

    Returns ``None`` when the turn should not enter the flow: there is no
    active draft and the message neither asks to book nor names a day or time.
    ``prompted`` marks that the previous turn was already a reservation reply,
    which suppresses the repeated availability caveat and lets a bare "7"
    answer the "which time?" question.
    """
    current = {
        field: value
        for field, value in (draft or {}).items()
        if field in _FIELDS and isinstance(value, str) and value
    }
    active = bool(current)

    if is_booking_cancellation(raw_text):
        if not active:
            return None
        reply = (
            "No problem — I’ve discarded that booking request. Tell me a day and time "
            f"whenever you’d like to start another, or call {RESTAURANT['phone']}."
        )
        return BookingTurn(reply=reply, draft={})

    if is_booking_confirmation(raw_text):
        if not active:
            return None
        missing = _missing_fields(current)
        if missing:
            labels = natural_join([_FIELD_LABELS[name] for name in missing], "and")
            return BookingTurn(
                reply=f"Almost there — before I can record the booking I still need {labels}.",
                draft=current,
            )
        return BookingTurn(reply="", draft={}, record=dict(current))

    phone, remainder = _extract_phone(raw_text)
    day = parse_day(remainder)
    time = parse_time_expression(remainder)
    if time is None and (active or prompted) and not current.get("time"):
        bare = _BARE_TIME_RE.match(remainder)
        if bare:
            hour, minute = int(bare.group(1)), int(bare.group(2) or 0)
            if 1 <= hour <= 23 and minute < 60:
                time = _assume_dinner(hour), minute
    name = _extract_name(remainder)
    parsed_any = bool(phone or day or time or name)

    if not active and not (wants_slot_booking(raw_text) or day or time is not None):
        return None

    updated = dict(current)
    notes: list[str] = []
    if day:
        if time_slots(day):
            updated["day"] = day
        else:
            notes.append(f"Maple & Ember is closed on {day}s, so there are no {day} slots.")
    if time is not None:
        hour24, minute = time
        slot = _format_slot(hour24)
        if minute != 0:
            notes.append("Chat bookings are on the hour.")
        elif slot not in _all_slots():
            notes.append(f"{slot} isn’t within the bookable dinner slots.")
        else:
            updated["time"] = slot
    if phone:
        updated["phone"] = phone
    if name:
        updated["name"] = name

    if updated.get("day") and updated.get("time") and updated["time"] not in time_slots(updated["day"]):
        notes.append(f"{updated['time']} isn’t a bookable slot on {updated['day']}.")
        updated.pop("time")

    if not parsed_any and updated.get("day") and updated.get("time") and not updated.get("name"):
        fallback_name = _whole_message_name(raw_text)
        if fallback_name:
            updated["name"] = fallback_name
            parsed_any = True

    reply = _prompt_for(updated, notes, initial=not active and not prompted, parsed_any=parsed_any)
    return BookingTurn(reply=reply, draft=updated)


def is_valid_draft(draft: object) -> bool:
    """Shape-check a booking draft coming back from the graph."""
    if not isinstance(draft, dict):
        return False
    for field, value in draft.items():
        if field not in _FIELDS or type(value) is not str:
            return False
        if field == "day" and (value not in WEEKDAYS or not time_slots(value)):
            return False
        if field == "time" and (not SLOT_RE.match(value) or value not in _all_slots()):
            return False
        if field == "name" and not NAME_RE.match(value):
            return False
        if field == "phone" and not PHONE_RE.match(value):
            return False
    return True


# --- Replies rendered around the CSV append ---------------------------------


def booking_confirmed_reply(record: dict[str, str], booking_id: int) -> str:
    return (
        f"Booked! Time slot {record['day']} {record['time']} for {record['name']} - "
        f"{record['phone']} is recorded in the booking log as booking #{booking_id}. "
        "Staff will call that number if the slot can’t be honoured. To change or "
        f"cancel, please call {RESTAURANT['phone']}."
    )


def booking_log_failure_reply() -> str:
    return (
        "Sorry — I couldn’t record the booking just now, so nothing has been saved. "
        f"Please reply \"confirm\" to try again, or call {RESTAURANT['phone']} to book directly."
    )


# --- CSV booking log --------------------------------------------------------


class BookingLog:
    """Append-only CSV log of confirmed chat bookings.

    Appends are serialized through one asyncio lock and performed in a worker
    thread, so concurrent sessions cannot interleave rows. Booking ids continue
    from any rows already present in the file.
    """

    HEADER = ("booking_id", "logged_at_utc", "day", "time", "name", "phone")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._next_id: int | None = None

    async def append(self, record: dict[str, str]) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._append_sync, record)

    def _append_sync(self, record: dict[str, str]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._next_id is None:
            self._next_id = self._count_rows() + 1
        booking_id = self._next_id
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(self.HEADER)
            writer.writerow(
                [
                    booking_id,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    record["day"],
                    record["time"],
                    record["name"],
                    record["phone"],
                ]
            )
        self._next_id += 1
        return booking_id

    def _count_rows(self) -> int:
        try:
            with self.path.open(newline="", encoding="utf-8") as handle:
                rows = sum(1 for row in csv.reader(handle) if row)
        except OSError:
            return 0
        return max(rows - 1, 0)

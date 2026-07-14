import asyncio
import csv

import pytest

from app.bookings import (
    BookingLog,
    booking_turn,
    is_valid_draft,
    parse_day,
    parse_phone,
    parse_time_expression,
    time_slots,
)
from app.config import Settings
from app.restaurant import RESTAURANT
from app.service import ChatService
from app.sessions import InMemorySessionStore


def make_service(tmp_path, **overrides) -> ChatService:
    settings = Settings(
        chat_provider="deterministic",
        bookings_csv_path=str(tmp_path / "bookings.csv"),
        max_turns_per_session=overrides.pop("turns", 50),
        session_ttl_seconds=60,
        max_active_sessions=50,
        max_history_messages=8,
        **overrides,
    )
    return ChatService(settings, InMemorySessionStore(60, 50))


async def converse(service: ChatService, *messages: str):
    session_id = None
    results = []
    for message in messages:
        result = await service.chat(message, session_id=session_id)
        session_id = result.session.session_id
        results.append(result)
    return results


def read_rows(tmp_path) -> list[dict[str, str]]:
    path = tmp_path / "bookings.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Slot derivation ---------------------------------------------------------


def test_slots_follow_restaurant_hours_with_last_seating_before_close():
    assert time_slots("Monday") == ()
    assert time_slots("Tuesday") == ("5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM")
    assert time_slots("Friday")[-1] == "10:00 PM"
    assert time_slots("Sunday") == ("5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Friday at 7pm", ("Friday", (19, 0))),
        ("book saturday 6:00 PM", ("Saturday", (18, 0))),
        ("sunday at 8", ("Sunday", (20, 0))),
        ("Tuesday 19:00 works", ("Tuesday", (19, 0))),
    ],
)
def test_day_and_time_parse_together(text, expected):
    assert (parse_day(text), parse_time_expression(text)) == expected


def test_party_size_numbers_are_not_read_as_times():
    assert parse_time_expression("a table for 4 please") is None


def test_phone_parses_separators_and_ignores_times():
    assert parse_phone("call me at (416) 555-0123") == "4165550123"
    assert parse_phone("+1 416 555 0199") == "+14165550199"
    assert parse_phone("see you at 7pm") is None


# --- The booking state machine ----------------------------------------------


def test_non_booking_reservation_questions_stay_out_of_the_flow():
    assert booking_turn("Do you take reservations?", {}) is None
    assert booking_turn("What is your cancellation policy?", {}) is None


def test_single_message_with_all_details_goes_straight_to_confirmation():
    turn = booking_turn("Book a table Friday at 7pm for Jane Doe, 416-555-0123", {})
    assert turn is not None
    assert turn.draft == {"day": "Friday", "time": "7:00 PM", "name": "Jane Doe", "phone": "4165550123"}
    assert "confirm time slot Friday 7:00 PM for Jane Doe - 4165550123" in turn.reply
    assert turn.record is None


def test_confirm_with_complete_draft_produces_the_record():
    draft = {"day": "Friday", "time": "7:00 PM", "name": "Jane Doe", "phone": "4165550123"}
    turn = booking_turn("confirm", draft)
    assert turn.record == draft
    assert turn.draft == {}


def test_confirm_without_complete_details_asks_for_the_gaps():
    turn = booking_turn("confirm", {"day": "Friday", "time": "7:00 PM"})
    assert turn.record is None
    assert "a name for the booking" in turn.reply
    assert "phone" in turn.reply
    assert turn.draft == {"day": "Friday", "time": "7:00 PM"}


def test_cancel_discards_the_pending_draft():
    turn = booking_turn("cancel", {"day": "Friday", "time": "7:00 PM"})
    assert turn.draft == {}
    assert turn.record is None
    assert "discarded" in turn.reply


def test_monday_and_off_slot_times_are_rejected_with_guidance():
    monday = booking_turn("book a table monday", {})
    assert "closed on Mondays" in monday.reply
    assert "day" not in monday.draft

    late = booking_turn("Sunday at 10pm", {}, prompted=True)
    assert "time" not in late.draft
    assert late.draft.get("day") == "Sunday"
    assert "5:00 PM" in late.reply  # lists the valid Sunday slots

    half_hour = booking_turn("Friday at 7:30 pm", {})
    assert "time" not in half_hour.draft
    assert "on the hour" in half_hour.reply


def test_details_can_be_corrected_before_confirming():
    draft = {"day": "Friday", "time": "7:00 PM", "name": "Jane Doe", "phone": "4165550123"}
    turn = booking_turn("actually make it saturday at 8pm", draft)
    assert turn.draft["day"] == "Saturday"
    assert turn.draft["time"] == "8:00 PM"
    assert "confirm time slot Saturday 8:00 PM for Jane Doe - 4165550123" in turn.reply


def test_draft_shape_validation_rejects_tampered_state():
    assert is_valid_draft({"day": "Friday", "time": "7:00 PM"})
    assert is_valid_draft({})
    assert not is_valid_draft({"day": "Monday"})
    assert not is_valid_draft({"time": "7:30 PM"})
    assert not is_valid_draft({"phone": "call-me"})
    assert not is_valid_draft({"name": "x" * 80})
    assert not is_valid_draft({"unexpected": "field"})


# --- Full conversation through the service -----------------------------------


@pytest.mark.asyncio
async def test_multi_turn_booking_confirms_into_the_csv_log(tmp_path):
    service = make_service(tmp_path)
    results = await converse(
        service,
        "Can I book a table?",
        "Friday at 7pm",
        "John Smith",
        "416-555-0123",
        "confirm",
    )

    assert [result.intent for result in results] == ["reservation"] * 5
    assert 'confirm time slot Friday 7:00 PM for John Smith - 4165550123' in results[3].response
    assert "booking #1" in results[4].response
    assert results[4].session.booking_draft == {}

    rows = read_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert (row["day"], row["time"], row["name"], row["phone"]) == (
        "Friday", "7:00 PM", "John Smith", "4165550123",
    )
    assert row["booking_id"] == "1"
    assert row["logged_at_utc"]


@pytest.mark.asyncio
async def test_nothing_is_logged_before_the_guest_confirms(tmp_path):
    service = make_service(tmp_path)
    await converse(service, "Book Friday 7pm for Jane Doe, 4165550123")
    assert read_rows(tmp_path) == []


@pytest.mark.asyncio
async def test_cancel_mid_flow_leaves_no_record_and_clears_the_draft(tmp_path):
    service = make_service(tmp_path)
    results = await converse(service, "Book Friday at 7pm", "cancel")
    assert "discarded" in results[-1].response
    assert results[-1].session.booking_draft == {}
    assert read_rows(tmp_path) == []


@pytest.mark.asyncio
async def test_booking_ids_increment_across_sessions(tmp_path):
    service = make_service(tmp_path)
    await converse(service, "Book Friday at 7pm for Jane Doe, 4165550123", "confirm")
    second = await converse(service, "Book Sunday at 6pm for Ana Lee, 6475550111", "confirm")

    assert "booking #2" in second[-1].response
    rows = read_rows(tmp_path)
    assert [row["booking_id"] for row in rows] == ["1", "2"]
    assert rows[1]["name"] == "Ana Lee"


@pytest.mark.asyncio
async def test_menu_question_mid_flow_is_answered_and_the_draft_survives(tmp_path):
    service = make_service(tmp_path)
    results = await converse(
        service,
        "Book Friday at 7pm",
        "How much is the burger?",
        "name is John Smith",
    )
    assert results[1].intent == "menu"
    assert "Ember Burger is listed at $25" in results[1].response
    assert results[2].intent == "reservation"
    assert results[2].session.booking_draft["name"] == "John Smith"
    assert results[2].session.booking_draft["day"] == "Friday"


@pytest.mark.asyncio
async def test_repeated_request_id_does_not_double_log_a_confirmation(tmp_path):
    from uuid import uuid4

    service = make_service(tmp_path)
    setup = await converse(service, "Book Friday at 7pm for Jane Doe, 4165550123")
    session_id = setup[-1].session.session_id
    request_id = uuid4()

    first = await service.chat("confirm", session_id=session_id, request_id=request_id)
    replay = await service.chat("confirm", session_id=session_id, request_id=request_id)

    assert "booking #1" in first.response
    assert replay.response == first.response
    assert len(read_rows(tmp_path)) == 1


@pytest.mark.asyncio
async def test_concurrent_confirms_from_different_sessions_write_distinct_rows(tmp_path):
    service = make_service(tmp_path)
    async def book(name: str, phone: str, day: str) -> None:
        await converse(service, f"Book {day} at 7pm for {name}, {phone}", "confirm")

    await asyncio.gather(
        book("Jane Doe", "4165550123", "Friday"),
        book("Ana Lee", "6475550111", "Saturday"),
        book("Sam Roy", "9055550188", "Sunday"),
    )

    rows = read_rows(tmp_path)
    assert len(rows) == 3
    assert sorted(row["booking_id"] for row in rows) == ["1", "2", "3"]
    assert {row["name"] for row in rows} == {"Jane Doe", "Ana Lee", "Sam Roy"}


@pytest.mark.asyncio
async def test_changing_an_existing_booking_still_defers_to_staff(tmp_path):
    service = make_service(tmp_path)
    results = await converse(service, "I need to cancel my reservation for tomorrow")
    assert results[0].intent == "reservation"
    assert "can’t change or cancel" in results[0].response
    assert RESTAURANT["phone"] in results[0].response


def test_booking_log_appends_and_numbers_rows(tmp_path):
    log = BookingLog(tmp_path / "nested" / "bookings.csv")
    record = {"day": "Friday", "time": "7:00 PM", "name": "Jane Doe", "phone": "4165550123"}

    first = asyncio.run(log.append(record))
    second = asyncio.run(log.append({**record, "name": "Ana Lee"}))

    assert (first, second) == (1, 2)
    with (tmp_path / "nested" / "bookings.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["name"] for row in rows] == ["Jane Doe", "Ana Lee"]

    resumed = BookingLog(tmp_path / "nested" / "bookings.csv")
    third = asyncio.run(resumed.append(record))
    assert third == 3

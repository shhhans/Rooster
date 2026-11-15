"""Normalization tests — constructed model JSON, no personal data, offline."""
from __future__ import annotations

import pytest

from app.roster_parser.errors import InvalidModelOutputError
from app.roster_parser.normalize import (
    AIRPORT_TIMEZONES,
    normalize_preview,
    parse_model_json,
)
from tests.roster_parser.conftest import duty_event, flight_event, model_output

HKG_LHR = {"HKG": "Asia/Hong_Kong", "LHR": "Europe/London"}


def run(events, unresolved=None, blank_dates=None, **kwargs):
    defaults = dict(
        page_count=1,
        home_timezone="Asia/Hong_Kong",
        airport_timezone_map=dict(HKG_LHR),
        roster_year=2026,
        roster_month=11,
    )
    defaults.update(kwargs)
    return normalize_preview(
        {"events": events, "unresolved": unresolved or [], "blank_dates": blank_dates or []},
        **defaults,
    )


def codes(items):
    return [item["code"] for item in items]


class TestFlightEvents:
    def test_full_flight_with_offsets_and_cross_day(self):
        result = run([flight_event()])
        assert result["unresolved_items"] == []
        event = result["proposed_events"][0]
        assert event["event_type"] == "flight"
        assert event["confirm_required"] is True
        payload = event["schedule_payload"]
        # Mirrors FlightIn semantics: ISO-8601 with explicit offset
        assert payload["flight_id"] == "XX089"  # leading zero preserved (string)
        assert payload["departure"] == "2026-11-03T23:55:00+08:00"
        # +1 cross-day arrival, London in November = GMT (+00:00)
        assert payload["arrival"] == "2026-11-04T06:10:00+00:00"
        assert payload["origin_tz"] == "Asia/Hong_Kong"
        assert payload["destination_tz"] == "Europe/London"
        assert payload["pending"] == []
        # Mirrors timeline semantics: naive local + independent timezone
        timeline = event["timeline"]
        assert timeline["departure_local"] == "2026-11-03T23:55"
        assert timeline["departure_timezone"] == "Asia/Hong_Kong"
        assert timeline["arrival_local"] == "2026-11-04T06:10"
        assert timeline["arrival_timezone"] == "Europe/London"
        assert event["provenance"]["page"] == 1
        assert event["provenance"]["quote"] == "XX089  HKG LHR  23:55  06:10+1"

    def test_sydney_summer_time_offset(self):
        # Sydney in November is AEDT (+11:00) — DST-independent check
        event = flight_event(
            origin="SYD", destination="HKG", dep="10:30", arr="18:20", arr_day_offset=0,
            raw_row="XX101  SYD HKG  10:30  18:20", markers=[],
        )
        result = run([event], airport_timezone_map={"HKG": "Asia/Hong_Kong"})
        payload = result["proposed_events"][0]["schedule_payload"]
        # Builtin table resolves SYD; explicit map resolves HKG
        assert payload["departure"] == "2026-11-03T10:30:00+11:00"
        assert payload["arrival"] == "2026-11-03T18:20:00+08:00"

    def test_unknown_airport_stays_pending(self):
        event = flight_event(origin="XXX", destination="ZZZ")
        result = run([event])
        assert result["proposed_events"][0]["schedule_payload"]["pending"] == [
            "arrival",
            "departure",
            "destination_tz",
            "origin_tz",
        ]
        assert "departure" not in result["proposed_events"][0]["schedule_payload"]
        assert codes(result["unresolved_items"]) == [
            "timezone_unknown",
            "timezone_unknown",
        ]

    def test_missing_date_stays_pending_with_roster_context(self):
        event = flight_event()
        del event["date"]
        result = run([event], roster_year=None, roster_month=None)
        assert codes(result["unresolved_items"]) == ["date_unknown"]
        payload = result["proposed_events"][0]["schedule_payload"]
        assert "departure" not in payload and "arrival" not in payload

    def test_day_only_date_resolved_via_roster_context(self):
        event = flight_event()
        del event["date"]
        event["day"] = 5
        result = run([event])
        assert result["unresolved_items"] == []
        assert result["proposed_events"][0]["schedule_payload"]["departure"].startswith("2026-11-05T")

    def test_invalid_time_goes_unresolved(self):
        event = flight_event(dep="25:99")
        result = run([event])
        assert "invalid_time_format" in codes(result["unresolved_items"])
        assert "departure" not in result["proposed_events"][0]["schedule_payload"]


class TestPrintedClockFormats:
    """Printed roster forms observed in real model output (attempt #8):
    HHMM, HHMML, HHMML+1, HH:MM — deterministic normalization, verbatim
    string always kept in raw."""

    def test_hhmm_local_marker_accepted(self):
        event = flight_event(dep="1505L", arr="1745L", arr_day_offset=0,
                             markers=["L"], date="2026-11-02")
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["departure"] == "2026-11-02T15:05:00+08:00"
        assert payload["arrival"] == "2026-11-02T17:45:00+00:00"
        assert result["unresolved_items"] == []

    def test_embedded_suffix_agrees_with_reported_offset_not_doubled(self):
        # Attempt #8 real shape (flight 0255): row 12-Nov, dep '0015L+1',
        # arr '0650L+1', model reported arr_day_offset=1 (already includes
        # the printed '+1'). Departure date shifts by its own suffix;
        # arrival must NOT be double-counted.
        event = flight_event(dep="0015L+1", arr="0650L+1", arr_day_offset=1,
                             markers=["L", "+1"], date="2026-11-12")
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["departure"].startswith("2026-11-13T00:15")
        assert payload["arrival"].startswith("2026-11-13T06:50")
        warning_codes = [w["code"] for w in result["warnings"]]
        assert "departure_day_offset_suffix" in warning_codes

    def test_suffix_disagreement_warns_and_uses_reported(self):
        event = flight_event(arr="0650L+1", arr_day_offset=2, markers=["+1"])
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["arrival"].startswith("2026-11-05T06:50")  # row 11-03 + 2
        warning_codes = [w["code"] for w in result["warnings"]]
        assert "arrival_day_offset_disagreement" in warning_codes

    def test_suffix_only_offset_applied(self):
        event = flight_event(arr="0650L+1", arr_day_offset=None, markers=["L", "+1"])
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["arrival"].startswith("2026-11-04T06:50")  # row 11-03 +1

    def test_departure_offset_suffix_warns(self):
        event = flight_event(dep="0015L+1", arr="0650L", arr_day_offset=0)
        result = run([event])
        warning_codes = [w["code"] for w in result["warnings"]]
        assert "departure_day_offset_suffix" in warning_codes

    def test_plain_hhmm_accepted(self):
        event = duty_event(start="0900L", end="1730L")
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["duty_start"] == "2026-11-04T09:00:00+08:00"
        assert payload["duty_end"] == "2026-11-04T17:30:00+08:00"

    def test_duty_offset_suffix_warns_not_applied(self):
        event = duty_event(start="0900", end="1730L+1")
        result = run([event])
        warning_codes = [w["code"] for w in result["warnings"]]
        assert "duty_time_day_offset_suffix" in warning_codes
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["duty_end"] == "2026-11-04T17:30:00+08:00"  # suffix not applied

    def test_garbage_time_still_rejected(self):
        event = flight_event(dep="1340LATE")
        result = run([event])
        assert "invalid_time_format" in codes(result["unresolved_items"])

    def test_l_marker_preserved_and_warned(self):
        event = flight_event(markers=["L", "+1"])
        result = run([event])
        warning_codes = [w["code"] for w in result["warnings"]]
        assert "uninterpreted_marker" in warning_codes
        assert result["proposed_events"][0]["raw"]["markers"] == ["L", "+1"]

    def test_provenance_page_out_of_range_still_reported(self):
        event = flight_event(page=9)
        result = run([event])
        assert result["proposed_events"][0]["provenance"]["page"] == 9


class TestDutyEvents:
    def test_duty_with_explicit_times(self):
        result = run([duty_event()])
        event = result["proposed_events"][0]
        payload = event["schedule_payload"]
        assert payload["duty_start"] == "2026-11-04T03:15:00+08:00"
        assert payload["duty_end"] == "2026-11-04T12:30:00+08:00"
        assert payload["pending"] == []
        assert event["timeline"]["duty_start_local"] == "2026-11-04T03:15"

    def test_duty_end_before_start_assumed_next_day(self):
        event = duty_event(start="23:00", end="06:00", raw_row="CRD01  23:00  06:00")
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["duty_end"] == "2026-11-05T06:00:00+08:00"
        assert "duty_end_next_day_assumed" in [w["code"] for w in result["warnings"]]

    def test_duty_without_home_timezone_pending(self):
        result = run([duty_event()], home_timezone=None)
        assert codes(result["unresolved_items"]) == ["home_timezone_unknown"]
        payload = result["proposed_events"][0]["schedule_payload"]
        assert "duty_start" not in payload
        assert set(payload["pending"]) == {"duty_start", "duty_end"}

    def test_duty_without_times_is_not_an_event(self):
        event = duty_event(start=None, end=None)
        result = run([event])
        assert result["proposed_events"] == []
        assert codes(result["unresolved_items"]) == ["duty_without_times"]
        # Raw marker preserved
        assert result["unresolved_items"][0]["raw"]["code"] == "PSJ00"


class TestMarkersAndBlanks:
    def test_g_marker_becomes_unresolved_not_rest(self):
        marker = {
            "type": "marker", "page": 1, "raw_row": "G", "code": "G",
            "date": "2026-11-08", "markers": ["G"],
        }
        result = run([marker])
        assert result["proposed_events"] == []
        assert codes(result["unresolved_items"]) == ["marker_without_times"]
        assert result["unresolved_items"][0]["raw"]["code"] == "G"
        # No rest/sleep/duty event fabricated
        assert not any(
            e["event_type"] in {"rest", "sleep"} for e in result["proposed_events"]
        )

    def test_blank_dates_become_warnings(self):
        result = run([], blank_dates=["2026-11-08", "2026-11-15"])
        assert result["proposed_events"] == []
        assert [w["code"] for w in result["warnings"]] == ["blank_day", "blank_day"]
        assert "not treated as rest" in result["warnings"][0]["message"]

    def test_continuation_without_link_unresolved(self):
        event = flight_event(continuation=True, continues_index=None)
        result = run([event])
        assert "unlinked_continuation" in codes(result["unresolved_items"])

    def test_model_unresolved_passthrough(self):
        result = run(
            [],
            unresolved=[{"code": "unreadable_cell", "message": "smudged", "page": 1, "raw": "..."}],
        )
        assert codes(result["unresolved_items"]) == ["unreadable_cell"]
        assert result["unresolved_items"][0]["provenance"]["page"] == 1

    def test_empty_result_warns(self):
        result = run([])
        assert [w["code"] for w in result["warnings"]] == ["empty_result"]


class TestMalformedEntries:
    def test_non_dict_event(self):
        result = run(["not a dict"])
        assert codes(result["unresolved_items"]) == ["malformed_model_event"]

    def test_unknown_type(self):
        result = run([{"type": "sleep", "page": 1}])
        assert codes(result["unresolved_items"]) == ["malformed_model_event"]

    def test_flight_without_number(self):
        result = run([flight_event(flight_number="")])
        assert codes(result["unresolved_items"]) == ["malformed_model_event"]

    def test_invalid_airport_code(self):
        result = run([flight_event(origin="12", destination="HK")])
        assert set(codes(result["unresolved_items"])) == {"invalid_airport_code"}


class TestDstHandling:
    def test_fold_ambiguous_resolves_fold0_with_warning(self):
        # London 2026-10-25 01:30 happens twice (BST -> GMT at 02:00)
        event = flight_event(date="2026-10-25", origin="LHR", destination="LHR",
                             dep="01:30", arr="01:30", arr_day_offset=0, markers=[])
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert payload["departure"] == "2026-10-25T01:30:00+01:00"  # fold=0 (BST)
        assert "dst_ambiguous" in [w["code"] for w in result["warnings"]]

    def test_dst_gap_goes_unresolved(self):
        # London 2026-03-29 01:30 does not exist (clocks jump 01:00 -> 02:00)
        event = flight_event(date="2026-03-29", origin="LHR", destination="LHR",
                             dep="01:30", arr="01:30", arr_day_offset=0, markers=[])
        result = run([event])
        payload = result["proposed_events"][0]["schedule_payload"]
        assert "departure" not in payload
        assert "dst_gap" in codes(result["unresolved_items"])


class TestParseModelJson:
    def test_plain_json(self):
        assert parse_model_json('{"events": []}') == {"events": []}

    def test_fenced_json(self):
        assert parse_model_json('```json\n{"events": []}\n```') == {"events": []}

    def test_think_block_stripped(self):
        assert parse_model_json('<think>reasoning</think>{"events": []}') == {"events": []}

    def test_json_with_surrounding_prose(self):
        text = 'Here is the extraction:\n{"events": [], "extra": 1}\nDone.'
        assert parse_model_json(text) == {"events": [], "extra": 1}

    def test_non_object_rejected(self):
        with pytest.raises(InvalidModelOutputError):
            parse_model_json('[1, 2, 3]')

    def test_garbage_rejected(self):
        with pytest.raises(InvalidModelOutputError):
            parse_model_json("no json at all")


class TestBuiltinTable:
    def test_sample_airports_covered(self):
        assert AIRPORT_TIMEZONES["HKG"] == "Asia/Hong_Kong"
        assert AIRPORT_TIMEZONES["LHR"] == "Europe/London"
        assert AIRPORT_TIMEZONES["SYD"] == "Australia/Sydney"

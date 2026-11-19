"""Model-output contract and normalization for the roster preview.

This module owns both directions of the MiniMax extraction contract:

* the system prompt (JSON-only extraction schema, verbatim preservation
  rules, no invented times, no expanded abbreviations, no confidence), and
* normalization of the parsed model JSON into the standardized preview:
  ``proposed_events`` / ``unresolved_items`` / ``warnings``.

Time semantics (aligned with timeline-api and the schedule contract):
* ``schedule_payload`` mirrors app/api/schedule_state.py ``FlightIn`` /
  ``DutyIn`` — departure/arrival/duty timestamps are ISO-8601 WITH explicit
  UTC offset, computed deterministically from printed local times + resolved
  IANA timezones (explicit airport map > built-in reference table > unknown).
* ``timeline`` mirrors the timeline event semantics — naive local timestamps
  plus an independent IANA timezone name.
* When the timezone, date or time cannot be determined, the field is left
  out of ``schedule_payload`` and listed under ``pending``; a matching
  ``unresolved_items`` entry is emitted. UTC is never fabricated and blank
  days / ``G`` markers never become rest or sleep events.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.roster_parser.errors import InvalidModelOutputError

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_AIRPORT_RE = re.compile(r"^[A-Z]{3}$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Built-in IATA -> IANA reference table for common ports (deterministic,
# factual data — not a guess). An explicitly supplied airport_timezone_map
# always wins; airports absent from both stay "pending" (unresolved), never
# guessed. Documented in docs/roster-parser.md.
AIRPORT_TIMEZONES: dict[str, str] = {
    "HKG": "Asia/Hong_Kong",
    "LHR": "Europe/London",
    "LGW": "Europe/London",
    "STN": "Europe/London",
    "SYD": "Australia/Sydney",
    "MEL": "Australia/Melbourne",
    "BNE": "Australia/Brisbane",
    "PER": "Australia/Perth",
    "AKL": "Pacific/Auckland",
    "SIN": "Asia/Singapore",
    "BKK": "Asia/Bangkok",
    "TPE": "Asia/Taipei",
    "ICN": "Asia/Seoul",
    "NRT": "Asia/Tokyo",
    "HND": "Asia/Tokyo",
    "KIX": "Asia/Tokyo",
    "PEK": "Asia/Shanghai",
    "PVG": "Asia/Shanghai",
    "CAN": "Asia/Shanghai",
    "SZX": "Asia/Shanghai",
    "MNL": "Asia/Manila",
    "DEL": "Asia/Kolkata",
    "BOM": "Asia/Kolkata",
    "DXB": "Asia/Dubai",
    "DOH": "Asia/Qatar",
    "CDG": "Europe/Paris",
    "FRA": "Europe/Berlin",
    "MUC": "Europe/Berlin",
    "AMS": "Europe/Amsterdam",
    "FCO": "Europe/Rome",
    "MAD": "Europe/Madrid",
    "ZRH": "Europe/Zurich",
    "SVO": "Europe/Moscow",
    "JFK": "America/New_York",
    "EWR": "America/New_York",
    "LAX": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
    "ORD": "America/Chicago",
    "YYZ": "America/Toronto",
    "YVR": "America/Vancouver",
    "HNL": "Pacific/Honolulu",
    "CAI": "Africa/Cairo",
    "JNB": "Africa/Johannesburg",
}

# Markers whose cross-day offset is already interpreted (via arr_day_offset).
_INTERPRETED_MARKERS = {"+1"}

SYSTEM_PROMPT = """You are a crew-roster extraction engine. You receive \
rendered roster page image(s) and extract every roster entry verbatim. \
Output ONLY one JSON object matching the schema below — no prose, no code \
fences, no confidence scores.

Schema:
{
  "events": [
    {"type": "flight", "page": <int, 1-based>, "raw_row": "<exact printed row text>",
     "flight_number": "<verbatim string incl. leading zeros>",
     "origin": "<IATA>", "destination": "<IATA>",
     "date": "<YYYY-MM-DD>",
     "dep": "<HH:MM as printed>", "arr": "<HH:MM as printed>",
     "duty_start": "<printed duty/report start time, or null if absent or unclear>",
     "duty_end": "<printed duty end/release time, or null if absent or unclear>",
     "duty_group": "<same roster-supported group identifier for flights in one continuous duty, or null if unclear>",
     "arr_day_offset": <int days to add to the roster ROW date for the arrival, e.g. 1 when the arrival prints '+1', else 0>,
     "markers": ["<every marker printed on the row, verbatim, e.g. 'L', '+1', 'G'>"],
     "continuation": <true only if this row continues the previous row and you could not merge it>,
     "continues_index": <index (0-based) of the event this row continues, else null>},
    {"type": "duty", "page": <int>, "raw_row": "<exact printed row text>",
     "code": "<verbatim duty code, e.g. PSJ00, CRD01, ELD02, AEU02, AEU03>",
     "date": "<YYYY-MM-DD>",
     "start": "<HH:MM or null if not printed>", "end": "<HH:MM or null if not printed>",
     "markers": ["<verbatim>"]},
    {"type": "marker", "page": <int>, "raw_row": "<exact printed row text>",
     "code": "<verbatim marker code, e.g. 'G'>", "date": "<YYYY-MM-DD>",
     "markers": ["<verbatim>"]}
  ],
  "unresolved": [
    {"code": "<short_snake_case_reason>", "message": "<what is unclear>",
     "page": <int>, "raw": "<exact printed text>"}
  ],
  "blank_dates": ["<YYYY-MM-DD days with no roster entry>"]
}

Rules:
1. Preserve every string verbatim: flight numbers keep leading zeros, codes \
and abbreviations are NEVER expanded or translated, times are exactly as \
printed (both "HH:MM" and printed forms like "1340L", "0650L+1", "0900" \
are acceptable; keep the L / +1 markers attached AND list them in markers).
2. Rows continuing a previous sector (e.g. arrival time printed on the \
following line) must be merged into that event; only if merging is not \
possible output the row with continuation=true and continues_index.
3. NEVER invent times. 'G' denotes a rest day and must be reported as a type "marker", not a scheduled event; blank days go to "blank_dates". Never \
present them as flights or duties with start/end times, and never infer \
rest or sleep.
4. If the roster header shows the year/month, use it for dates; when a date \
is ambiguous or missing report the entry in "unresolved".
5. If a printed time carries a day marker like '+1', set arr_day_offset \
accordingly (it already includes the marker) and keep '+1' in markers. The \
same marker on a departure clock shifts the departure date; keep it \
attached. Any other marker (e.g. 'L') is kept in markers exactly as \
printed and not interpreted.
6. Report unreadable, ambiguous or contradictory cells in "unresolved" \
instead of guessing.
7. Use page numbers counted from the first attached image (1-based).
8. For flights extract printed duty start/report and duty end/release only when their meaning is supported by the roster headings or layout. Keep the printed values verbatim, including any date or time-zone markers. Do not confuse these with departure or arrival; use null and report ambiguity in unresolved when uncertain. Never infer missing duty times.
9. Identify flights that form one continuous duty when supported by the roster. On a same-day return or other continuous sequence without a hotel rest, the duty start is normally printed only on the first flight and the duty end only on the last flight. Give all flights in that duty the same duty_group identifier, unique within this output (for example duty-1). Keep the unprinted duty_start or duty_end as null on each individual row; do not copy times between rows. Do not assume a hotel rest or group flights where the roster is ambiguous; use null and report the ambiguity in unresolved."""


def build_user_text(page_count: int, roster_year: int | None, roster_month: int | None) -> str:
    parts = [
        f"Attached: {page_count} roster page image(s), in order (page 1..{page_count}).",
        "Extract every roster entry and return ONLY the JSON object defined in the schema.",
    ]
    if roster_year is not None and roster_month is not None:
        parts.append(f"Context: the roster month is {roster_year:04d}-{roster_month:02d}.")
    return "\n".join(parts)


def parse_model_json(content: str) -> dict:
    """Extract the contracted JSON object from raw model text.

    Strips reasoning blocks and code fences, then parses. Anything that is
    not a JSON object raises InvalidModelOutputError (never a fabricated
    empty result).
    """
    text = _THINK_RE.sub("", content)
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    excerpt = content.strip()[:200]
    raise InvalidModelOutputError(
        f"model output is not the contracted JSON object; excerpt: {excerpt!r}"
    )


def normalize_preview(
    model_json: dict,
    *,
    page_count: int,
    home_timezone: str | None,
    airport_timezone_map: dict[str, str],
    roster_year: int | None,
    roster_month: int | None,
) -> dict:
    """Convert parsed model JSON into the standardized preview structure.

    Malformed individual entries become unresolved items — one bad row never
    crashes the whole preview and nothing is silently dropped.
    """
    if not isinstance(model_json, dict):
        raise InvalidModelOutputError("model JSON must be an object")
    raw_events = model_json.get("events")
    if not isinstance(raw_events, list):
        raise InvalidModelOutputError("model JSON is missing the 'events' list")

    proposed_events: list[dict] = []
    unresolved_items: list[dict] = []
    warnings: list[dict] = []

    for index, entry in enumerate(raw_events):
        if not isinstance(entry, dict):
            unresolved_items.append(
                _unresolved("malformed_model_event", f"event #{index} is not an object", entry)
            )
            continue
        event_type = entry.get("type")
        try:
            if event_type == "flight":
                _normalize_flight(
                    entry,
                    index=index,
                    out_events=proposed_events,
                    out_unresolved=unresolved_items,
                    out_warnings=warnings,
                    airport_timezone_map=airport_timezone_map,
                    roster_year=roster_year,
                    roster_month=roster_month,
                )
            elif event_type == "duty":
                _normalize_duty(
                    entry,
                    out_events=proposed_events,
                    out_unresolved=unresolved_items,
                    out_warnings=warnings,
                    home_timezone=home_timezone,
                    roster_year=roster_year,
                    roster_month=roster_month,
                )
            elif event_type == "marker":
                if str(entry.get('code', '')).strip().upper() == 'G':
                    continue
                unresolved_items.append(
                    _unresolved(
                        "marker_without_times",
                        f"marker {entry.get('code')!r} has no printed times; "
                        "recorded verbatim, not interpreted as rest/duty",
                        entry,
                    )
                )
            else:
                unresolved_items.append(
                    _unresolved(
                        "malformed_model_event",
                        f"event #{index} has unknown type {event_type!r}",
                        entry,
                    )
                )
        except _MalformedEntry as exc:
            unresolved_items.append(
                _unresolved("malformed_model_event", str(exc), entry)
            )

    for item in model_json.get("unresolved", []) or []:
        if not isinstance(item, dict):
            unresolved_items.append(
                _unresolved("malformed_model_event", "unresolved entry is not an object", item)
            )
            continue
        unresolved_items.append(
            {
                "code": str(item.get("code") or "model_unresolved"),
                "message": str(item.get("message") or ""),
                "raw": item.get("raw"),
                "provenance": {
                    "page": _page_of(item),
                    "quote": item.get("raw") if isinstance(item.get("raw"), str) else None,
                },
            }
        )

    for blank in model_json.get("blank_dates", []) or []:
        warnings.append(
            {
                "code": "blank_day",
                "message": f"no roster entries reported for {blank!r}; "
                "no event inferred (not treated as rest)",
                "provenance": None,
            }
        )

    if not proposed_events and not unresolved_items and not warnings:
        warnings.append(
            {
                "code": "empty_result",
                "message": "model reported no events and no unresolved items; "
                "verify the document is a roster page",
                "provenance": None,
            }
        )

    return {
        "proposed_events": proposed_events,
        "unresolved_items": unresolved_items,
        "warnings": warnings,
    }


class _MalformedEntry(Exception):
    """Internal: one model entry cannot be normalized (goes to unresolved)."""


# ---------------------------------------------------------------------------
# Flight events
# ---------------------------------------------------------------------------


def _normalize_flight(
    entry: dict,
    *,
    index: int,
    out_events: list,
    out_unresolved: list,
    out_warnings: list,
    airport_timezone_map: dict[str, str],
    roster_year: int | None,
    roster_month: int | None,
) -> None:
    flight_number = entry.get("flight_number")
    if not isinstance(flight_number, str) or not flight_number.strip():
        raise _MalformedEntry(f"flight #{index} has no usable flight_number")
    flight_number = flight_number.strip()

    schedule_payload: dict = {"flight_id": flight_number}
    if isinstance(entry.get('duty_group'), str) and entry['duty_group'].strip():
        schedule_payload['duty_group'] = entry['duty_group'].strip()
    for field in ("duty_start", "duty_end"):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            schedule_payload[field + "_printed"] = value
    timeline: dict = {
        "departure_local": None,
        "departure_timezone": None,
        "arrival_local": None,
        "arrival_timezone": None,
    }
    flight_day: date | None = None

    departure = None
    arrival = None
    origin_tz = None
    destination_tz = None

    origin = _airport_of(entry.get("origin"))
    destination = _airport_of(entry.get("destination"))
    if origin:
        origin_tz = airport_timezone_map.get(origin) or AIRPORT_TIMEZONES.get(origin)
        if origin_tz is None:
            out_unresolved.append(
                _unresolved(
                    "timezone_unknown",
                    f"no IANA timezone for origin airport {origin!r}; "
                    "departure offset cannot be computed",
                    entry,
                )
            )
    else:
        out_unresolved.append(
            _unresolved("invalid_airport_code", f"origin {entry.get('origin')!r} is not a 3-letter IATA code", entry)
        )
    if destination:
        destination_tz = airport_timezone_map.get(destination) or AIRPORT_TIMEZONES.get(destination)
        if destination_tz is None:
            out_unresolved.append(
                _unresolved(
                    "timezone_unknown",
                    f"no IANA timezone for destination airport {destination!r}; "
                    "arrival offset cannot be computed",
                    entry,
                )
            )
    else:
        out_unresolved.append(
            _unresolved("invalid_airport_code", f"destination {entry.get('destination')!r} is not a 3-letter IATA code", entry)
        )

    day = _resolve_date(entry, roster_year, roster_month)
    if day is None:
        out_unresolved.append(
            _unresolved("date_unknown", f"flight {flight_number!r} has no determinable date", entry)
        )
    else:
        flight_day = day

    dep_parsed = parse_printed_clock(entry.get("dep"))
    arr_parsed = parse_printed_clock(entry.get("arr"))
    dep_clock = dep_parsed[0] if dep_parsed else None
    arr_clock = arr_parsed[0] if arr_parsed else None
    if dep_parsed is None:
        out_unresolved.append(
            _unresolved("invalid_time_format", f"flight {flight_number!r} departure time {entry.get('dep')!r} is not a printed clock (HH:MM/HHMM/L/+1)", entry)
        )
    if arr_parsed is None:
        out_unresolved.append(
            _unresolved("invalid_time_format", f"flight {flight_number!r} arrival time {entry.get('arr')!r} is not a printed clock (HH:MM/HHMM/L/+1)", entry)
        )

    offset = _day_offset_of(entry.get("arr_day_offset"), out_warnings, entry)
    # Arrival day offset: the model-reported value already includes a printed
    # '+N' when it followed the contract; the suffix is used only as a
    # fallback, and a disagreement is surfaced, never silently doubled.
    if arr_parsed is not None and arr_parsed[1]:
        if offset and arr_parsed[1] != offset:
            out_warnings.append(
                {
                    "code": "arrival_day_offset_disagreement",
                    "message": f"arrival clock suffix '+{arr_parsed[1]}' disagrees with "
                    f"reported arr_day_offset={offset}; using {offset} pending confirmation",
                    "provenance": _provenance_of(entry),
                }
            )
        elif not offset:
            offset = arr_parsed[1]
            out_warnings.append(
                {
                    "code": "arrival_day_offset_from_suffix",
                    "message": f"arr_day_offset was not reported; taken from the printed '+{arr_parsed[1]}' suffix",
                    "provenance": _provenance_of(entry),
                }
            )
    # Departure day offset: a '+N' printed on the departure clock shifts the
    # departure date relative to the roster row (deterministic printed data).
    dep_day = flight_day
    if flight_day is not None and dep_parsed is not None and dep_parsed[1]:
        dep_day = flight_day + timedelta(days=dep_parsed[1])
        out_warnings.append(
            {
                "code": "departure_day_offset_suffix",
                "message": f"departure time {entry.get('dep')!r} carries '+{dep_parsed[1]}'; "
                "departure date taken as row date + "
                f"{dep_parsed[1]} (={dep_day.isoformat()})",
                "provenance": _provenance_of(entry),
            }
        )

    if flight_day is not None and dep_clock is not None and origin_tz is not None:
        iso, warning_code = _local_to_aware_iso(dep_day, dep_clock, origin_tz)
        if iso:
            departure = iso
            timeline["departure_local"] = f"{dep_day.isoformat()}T{dep_clock}"
            timeline["departure_timezone"] = origin_tz
        elif warning_code == "dst_gap":
            out_unresolved.append(
                _unresolved("dst_gap", f"departure local time does not exist on {dep_day} in {origin_tz} (DST gap)", entry)
            )
        if warning_code == "dst_ambiguous":
            out_warnings.append(
                {
                    "code": "dst_ambiguous",
                    "message": f"departure local time is ambiguous on {dep_day} in {origin_tz}; fold=0 (earlier offset) used",
                    "provenance": _provenance_of(entry),
                }
            )

    if flight_day is not None and arr_clock is not None and destination_tz is not None:
        arrival_day = flight_day + timedelta(days=offset)
        iso, warning_code = _local_to_aware_iso(arrival_day, arr_clock, destination_tz)
        if iso:
            arrival = iso
            timeline["arrival_local"] = f"{arrival_day.isoformat()}T{arr_clock}"
            timeline["arrival_timezone"] = destination_tz
        elif warning_code == "dst_gap":
            out_unresolved.append(
                _unresolved("dst_gap", f"arrival local time does not exist on {arrival_day} in {destination_tz} (DST gap)", entry)
            )
        if warning_code == "dst_ambiguous":
            out_warnings.append(
                {
                    "code": "dst_ambiguous",
                    "message": f"arrival local time is ambiguous on {arrival_day} in {destination_tz}; fold=0 (earlier offset) used",
                    "provenance": _provenance_of(entry),
                }
            )

    for field, reference, zone, direction in (('duty_start', departure, origin_tz, -1), ('duty_end', arrival, destination_tz, 1)):
        raw = entry.get(field)
        if not raw:
            continue
        parsed = parse_printed_clock(raw)
        if not parsed or not reference or not zone:
            out_unresolved.append(_unresolved('flight_duty_time_unresolved', f'{field} cannot be resolved from {raw!r}', entry))
            continue
        anchor = datetime.fromisoformat(reference)
        day = flight_day + timedelta(days=parsed[1]) if parsed[1] and flight_day else anchor.date()
        iso, warning = _local_to_aware_iso(day, parsed[0], zone)
        if iso and not parsed[1]:
            instant = datetime.fromisoformat(iso)
            if (direction == -1 and instant > anchor) or (direction == 1 and instant < anchor):
                day += timedelta(days=direction)
                iso, warning = _local_to_aware_iso(day, parsed[0], zone)
                out_warnings.append({'code': 'flight_duty_date_inferred', 'message': f'{field} date inferred as {day.isoformat()}; confirm against roster', 'provenance': _provenance_of(entry)})
        if not iso or warning:
            out_unresolved.append(_unresolved('flight_duty_time_unresolved', f'{field} has ambiguous or invalid local time {raw!r}', entry))
            continue
        instant = datetime.fromisoformat(iso)
        if (direction == -1 and instant > anchor) or (direction == 1 and instant < anchor):
            out_unresolved.append(_unresolved('flight_duty_time_unresolved', f'{field} is inconsistent with flight time', entry))
            continue
        schedule_payload[field + '_utc'] = instant.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    for name, value in (
        ("departure", departure),
        ("arrival", arrival),
        ("origin_tz", origin_tz),
        ("destination_tz", destination_tz),
    ):
        if value is not None:
            schedule_payload[name] = value
    # Every uncomputable schedule field is listed as pending — derived from
    # the final payload so it can never disagree with it.
    schedule_payload["pending"] = sorted(
        {"departure", "arrival", "origin_tz", "destination_tz"} - set(schedule_payload)
    )

    if entry.get("continuation"):
        if entry.get("continues_index") is None:
            out_unresolved.append(
                _unresolved(
                    "unlinked_continuation",
                    f"row #{index} continues a previous sector but the link "
                    "could not be resolved; kept unmerged pending confirmation",
                    entry,
                )
            )
        else:
            out_warnings.append(
                {
                    "code": "continuation_row",
                    "message": f"row #{index} is a continuation row (raw kept unmerged)",
                    "provenance": _provenance_of(entry),
                }
            )

    markers = [str(m) for m in entry.get("markers", []) if str(m)]
    uninterpreted = [m for m in markers if m not in _INTERPRETED_MARKERS]
    if uninterpreted:
        out_warnings.append(
            {
                "code": "uninterpreted_marker",
                "message": f"markers {uninterpreted!r} preserved verbatim; semantics not interpreted",
                "provenance": _provenance_of(entry),
            }
        )

    out_events.append(
        {
            "event_type": "flight",
            "confirm_required": True,
            "schedule_payload": schedule_payload,
            "timeline": timeline,
            "raw": entry,
            "provenance": _provenance_of(entry),
        }
    )


# ---------------------------------------------------------------------------
# Duty events
# ---------------------------------------------------------------------------


def _normalize_duty(
    entry: dict,
    *,
    out_events: list,
    out_unresolved: list,
    out_warnings: list,
    home_timezone: str | None,
    roster_year: int | None,
    roster_month: int | None,
) -> None:
    code = entry.get("code")
    start_parsed = parse_printed_clock(entry.get("start"))
    end_parsed = parse_printed_clock(entry.get("end"))
    start_clock = start_parsed[0] if start_parsed else None
    end_clock = end_parsed[0] if end_parsed else None
    # Duty times are same-day rows; a '+N' suffix on duty clocks is not
    # interpreted (warned) — only flight arrival offsets are applied.
    for label, parsed, original in (
        ("start", start_parsed, entry.get("start")),
        ("end", end_parsed, entry.get("end")),
    ):
        if parsed is not None and parsed[1]:
            out_warnings.append(
                {
                    "code": "duty_time_day_offset_suffix",
                    "message": f"duty {code!r} {label} time {original!r} carries a '+N' day-offset suffix; "
                    "not interpreted pending confirmation",
                    "provenance": _provenance_of(entry),
                }
            )

    if start_clock is None and end_clock is None:
        out_unresolved.append(
            _unresolved(
                "duty_without_times",
                f"duty {code!r} has no printed start/end times; recorded "
                "verbatim, not interpreted as rest",
                entry,
            )
        )
        return

    schedule_payload: dict = {"duty_code": str(code) if code is not None else None}
    timeline: dict = {
        "duty_start_local": None,
        "duty_end_local": None,
        "duty_timezone": None,
    }
    duty_start = None
    duty_end = None

    if home_timezone is None:
        out_unresolved.append(
            _unresolved(
                "home_timezone_unknown",
                f"duty {code!r} times cannot be resolved without home_timezone context",
                entry,
            )
        )
    else:
        day = _resolve_date(entry, roster_year, roster_month)
        if day is None:
            out_unresolved.append(
                _unresolved("date_unknown", f"duty {code!r} has no determinable date", entry)
            )
        else:
            if start_clock is not None:
                iso, warning_code = _local_to_aware_iso(day, start_clock, home_timezone)
                if iso:
                    duty_start = iso
                    timeline["duty_start_local"] = f"{day.isoformat()}T{start_clock}"
                    timeline["duty_timezone"] = home_timezone
                elif warning_code == "dst_gap":
                    out_unresolved.append(
                        _unresolved("dst_gap", f"duty start local time does not exist on {day} in {home_timezone}", entry)
                    )
                if warning_code == "dst_ambiguous":
                    out_warnings.append(
                        {
                            "code": "dst_ambiguous",
                            "message": f"duty start local time is ambiguous on {day} in {home_timezone}; fold=0 used",
                            "provenance": _provenance_of(entry),
                        }
                    )
            if end_clock is not None:
                end_day = day
                if start_clock is not None and end_clock <= start_clock:
                    end_day = day + timedelta(days=1)
                    out_warnings.append(
                        {
                            "code": "duty_end_next_day_assumed",
                            "message": f"duty {code!r} end time is not after start time; end assumed on the next day",
                            "provenance": _provenance_of(entry),
                        }
                    )
                iso, warning_code = _local_to_aware_iso(end_day, end_clock, home_timezone)
                if iso:
                    duty_end = iso
                    timeline["duty_end_local"] = f"{end_day.isoformat()}T{end_clock}"
                    timeline["duty_timezone"] = home_timezone
                elif warning_code == "dst_gap":
                    out_unresolved.append(
                        _unresolved("dst_gap", f"duty end local time does not exist on {end_day} in {home_timezone}", entry)
                    )
                if warning_code == "dst_ambiguous":
                    out_warnings.append(
                        {
                            "code": "dst_ambiguous",
                            "message": f"duty end local time is ambiguous on {end_day} in {home_timezone}; fold=0 used",
                            "provenance": _provenance_of(entry),
                        }
                    )
    for name, value in (("duty_start", duty_start), ("duty_end", duty_end)):
        if value is not None:
            schedule_payload[name] = value
    schedule_payload["pending"] = sorted({"duty_start", "duty_end"} - set(schedule_payload))

    if start_clock is not None and end_clock is None:
        out_unresolved.append(
            _unresolved("duty_time_missing", f"duty {code!r} has a start but no end time", entry)
        )
    if end_clock is not None and start_clock is None:
        out_unresolved.append(
            _unresolved("duty_time_missing", f"duty {code!r} has an end but no start time", entry)
        )

    out_events.append(
        {
            "event_type": "duty",
            "confirm_required": True,
            "schedule_payload": schedule_payload,
            "timeline": timeline,
            "raw": entry,
            "provenance": _provenance_of(entry),
        }
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _airport_of(value) -> str | None:
    if isinstance(value, str):
        candidate = value.strip().upper()
        if _AIRPORT_RE.match(candidate):
            return candidate
    return None


def _clock_of(value) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if _TIME_RE.match(candidate):
        return candidate
    return None


_PRINTED_CLOCK_RE = re.compile(
    r"^(\d{2}):?(\d{2})"          # HH:MM or HHMM
    r"(L?)"                        # optional 'L' local-time suffix
    r"(\+\d+)?$",                  # optional '+N' day-offset suffix
    re.IGNORECASE,
)


def parse_printed_clock(value):
    """Accept a printed roster clock and normalize it to (HH:MM, day_offset).

    Accepts the forms actually printed on rosters and returned verbatim by
    the model: ``HH:MM``, ``HHMM``, with optional trailing ``L`` (local-time
    marker, preserved in markers) and ``+N`` (day offset). Returns
    ``("HH:MM", extra_day_offset)`` or None when the value is not a usable
    clock. Deterministic normalization of printed formats — the verbatim
    string is always kept in ``raw``.
    """
    if not isinstance(value, str):
        return None
    match = _PRINTED_CLOCK_RE.match(value.strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    day_offset = int(match.group(4) or 0)
    return f"{hour:02d}:{minute:02d}", day_offset


def _day_offset_of(value, out_warnings: list, entry: dict) -> int:
    if value is None:
        return 0
    try:
        offset = int(str(value).strip().lstrip("+"))
    except ValueError:
        out_warnings.append(
            {
                "code": "invalid_day_offset",
                "message": f"arrival day offset {value!r} is not an integer; treated as 0 pending confirmation",
                "provenance": _provenance_of(entry),
            }
        )
        return 0
    if offset < 0:
        out_warnings.append(
            {
                "code": "invalid_day_offset",
                "message": f"arrival day offset {value!r} is negative; treated as 0 pending confirmation",
                "provenance": _provenance_of(entry),
            }
        )
        return 0
    return offset


def _resolve_date(entry: dict, roster_year: int | None, roster_month: int | None) -> date | None:
    raw_date = entry.get("date")
    if isinstance(raw_date, str):
        try:
            return date.fromisoformat(raw_date.strip())
        except ValueError:
            pass
    year = _int_field(entry.get("year")) or roster_year
    month = _int_field(entry.get("month")) or roster_month
    day = _int_field(entry.get("day"))
    if year and month and day:
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def _int_field(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _local_to_aware_iso(day: date, clock: str, tz_name: str) -> tuple[str | None, str | None]:
    """Convert a printed local wall time to an ISO-8601 string with offset.

    Deterministic rules: DST gaps produce no instant (unresolved); ambiguous
    folds resolve to fold=0 with an explicit warning. Returns (iso, warning)
    where exactly one of the two is set.
    """
    hour, minute = clock.split(":")
    naive = datetime(day.year, day.month, day.day, int(hour), int(minute))
    zone = ZoneInfo(tz_name)
    aware = naive.replace(tzinfo=zone, fold=0)
    roundtrip = aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if roundtrip != naive:
        return None, "dst_gap"
    if aware.replace(fold=1).utcoffset() != aware.utcoffset():
        return aware.isoformat(), "dst_ambiguous"
    return aware.isoformat(), None


def _provenance_of(entry: dict) -> dict:
    page = _page_of(entry)
    quote = entry.get("raw_row") if isinstance(entry.get("raw_row"), str) else None
    return {"page": page, "quote": quote}


def _page_of(entry: dict) -> int | None:
    page = entry.get("page") if isinstance(entry, dict) else None
    page_int = _int_field(page)
    return page_int


def _unresolved(code: str, message: str, raw) -> dict:
    entry = raw if isinstance(raw, dict) else None
    return {
        "code": code,
        "message": message,
        "raw": raw if isinstance(raw, (dict, str)) else repr(raw)[:300],
        "provenance": {
            "page": _page_of(entry) if entry else None,
            "quote": entry.get("raw_row") if entry and isinstance(entry.get("raw_row"), str) else None,
        },
    }

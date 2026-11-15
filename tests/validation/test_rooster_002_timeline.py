"""Black-box validation tests for the ROOSTER-002 timeline kernel contract.

Frozen contract: ROOSTER-TIMELINE-CORE (module_id), frozen 2026-09-20.
Lead ruling applied: content idempotency evaluated BEFORE superseded/retracted
checks in correct_schedule_event; new successors inherit source=target.source.

Interfaces under test (public_interfaces of the frozen contract):
  ingest_schedule_event, correct_schedule_event, retract_schedule_event,
  list_schedule_events, get_schedule_event

These tests are written strictly from the frozen contract (black box). They do
not read or depend on implementation internals.
"""

import datetime as dt
import os
import re
import uuid

import pytest

from rooster_client import RoosterClient

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> RoosterClient:
    return RoosterClient()


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Hermetic per-test DB, per the invocation artifact's recommended fixture.

    Points ROOSTER_TIMELINE_DB at a fresh tmp file and resets the service so
    every test starts from an empty store.
    """
    monkeypatch.setenv("ROOSTER_TIMELINE_DB", str(tmp_path / "kernel.db"))
    import app.invocation as invocation
    invocation.reset_isolated_db()
    yield


def _tz_utc_instant(local: str, tz_name: str) -> dt.datetime:
    """Resolve a naive local ISO timestamp in an IANA zone to a UTC datetime.

    Mirror of the contract's resolution rule (UTC instant via IANA database).
    DST-gap local times raise.
    """
    from zoneinfo import ZoneInfo

    naive = dt.datetime.fromisoformat(local)
    tz = ZoneInfo(tz_name)
    try:
        aware = naive.replace(tzinfo=tz)
    except Exception:
        raise
    # detect non-existent local times (DST gap): round-trip check
    back = aware.astimezone(tz)
    if (back.replace(tzinfo=None), back.utcoffset()) != (naive, aware.utcoffset()):
        raise ValueError(f"local time does not exist in zone {tz_name}: {local}")
    return aware.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _parse_utc(s: str) -> dt.datetime:
    """Parse an ISO-8601 UTC string with Z suffix (or +00:00)."""
    s = s.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(s).replace(tzinfo=None)


def _iso_z(d: dt.datetime) -> str:
    """Format a naive UTC datetime as ISO-8601 with Z suffix (seconds)."""
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# ingest_schedule_event
# ---------------------------------------------------------------------------


def test_ingest_creates_full_unified_shape(client: RoosterClient):
    rec = client.ingest_schedule_event(
        user_id="u1",
        dedupe_key="k1",
        occurred_at="2025-04-12T01:30:00",
        timezone="Asia/Shanghai",
        source="manual",
        status="planned",
        title="Pre-flight rest",
        payload={"flight": "CA981"},
    )
    assert rec["track"] == "schedule"
    assert rec["user_id"] == "u1"
    assert rec["dedupe_key"] == "k1"
    assert rec["source"] == "manual"
    assert rec["status"] == "planned"
    assert rec["title"] == "Pre-flight rest"
    assert rec["payload"] == {"flight": "CA981"}
    assert rec["timezone"] == "Asia/Shanghai"
    # timezone normalization: 2025-04-12T01:30 +08:00 -> 2025-04-11T17:30:00Z
    assert _parse_utc(rec["occurred_at_utc"]) == dt.datetime(2025, 4, 11, 17, 30)
    assert rec["occurred_at_utc"].endswith("Z")
    # local_time preserved naive (no zone designator)
    assert rec["local_time"] == "2025-04-12T01:30:00"
    assert "Z" not in rec["local_time"]
    # chain/retraction defaults
    assert rec["supersedes"] is None
    assert rec["superseded_by"] is None
    assert rec["retracted"] is False
    assert rec["retracted_at"] is None
    assert rec["retracted_reason"] is None
    # timestamps parse and created_at == updated_at
    assert _parse_utc(rec["created_at"]) == _parse_utc(rec["updated_at"])
    # fresh UUID
    uuid.UUID(rec["event_id"])


def test_ingest_payload_default_empty(client: RoosterClient):
    rec = client.ingest_schedule_event(
        user_id="u1",
        dedupe_key="k2",
        occurred_at="2025-04-12T09:00:00",
        timezone="UTC",
        source="manual",
        status="occurred",
        title="No payload",
    )
    assert rec["payload"] == {}
    assert rec["status"] == "occurred"


def test_ingest_timezone_normalization_negative_offset(client: RoosterClient):
    rec = client.ingest_schedule_event(
        user_id="u1",
        dedupe_key="k3",
        occurred_at="2025-04-12T01:30:00",
        timezone="America/New_York",
        source="manual",
        status="planned",
        title="NY event",
    )
    # 2025-04-12 is EDT (UTC-4): 01:30 EDT -> 05:30 UTC
    assert _parse_utc(rec["occurred_at_utc"]) == dt.datetime(2025, 4, 12, 5, 30)
    assert rec["local_time"] == "2025-04-12T01:30:00"


def test_ingest_idempotent_returns_original_verbatim(client: RoosterClient):
    first = client.ingest_schedule_event(
        user_id="u1",
        dedupe_key="k4",
        occurred_at="2025-05-01T10:00:00",
        timezone="UTC",
        source="manual",
        status="planned",
        title="Original title",
    )
    second = client.ingest_schedule_event(
        user_id="u1",
        dedupe_key="k4",
        occurred_at="2025-05-01T11:00:00",  # different content
        timezone="Asia/Tokyo",  # different everything
        source="google_calendar",
        status="occurred",
        title="Different title",
    )
    assert second == first
    assert second["event_id"] == first["event_id"]
    assert second["title"] == "Original title"
    assert second["created_at"] == first["created_at"]


def test_ingest_idempotency_scoped_to_user(client: RoosterClient):
    a = client.ingest_schedule_event(
        user_id="uA", dedupe_key="same", occurred_at="2025-06-01T08:00:00",
        timezone="UTC", source="manual", status="planned", title="A",
    )
    b = client.ingest_schedule_event(
        user_id="uB", dedupe_key="same", occurred_at="2025-06-01T08:00:00",
        timezone="UTC", source="manual", status="planned", title="B",
    )
    assert a["event_id"] != b["event_id"]
    assert a["user_id"] == "uA"
    assert b["user_id"] == "uB"


def test_ingest_invalid_timezone(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.ingest_schedule_event(
            user_id="u1", dedupe_key="k-bad-tz",
            occurred_at="2025-04-12T01:30:00",
            timezone="Not/AZone", source="manual", status="planned",
            title="Bad tz",
        )
    assert "UNKNOWN_TIMEZONE" in str(ei.value) or "UNKNOWN_TIMEZONE" in str(
        ei.value.__dict__.get("body", "")
    ) or "UNKNOWN_TIMEZONE" in _err_text(ei.value)


def test_ingest_dst_gap_rejected(client: RoosterClient):
    # America/New_York spring-forward 2025-03-09: 02:00-03:00 does not exist
    with pytest.raises(Exception) as ei:
        client.ingest_schedule_event(
            user_id="u1", dedupe_key="k-dst-gap",
            occurred_at="2025-03-09T02:30:00",
            timezone="America/New_York", source="manual", status="planned",
            title="Gap",
        )
    assert "UNKNOWN_TIMEZONE" in _err_text(ei.value)


def test_ingest_invalid_input_cases(client: RoosterClient):
    base = dict(
        user_id="u1", dedupe_key="k-iv", occurred_at="2025-04-12T01:30:00",
        timezone="UTC", source="manual", status="planned", title="T",
    )
    # missing user_id
    with pytest.raises(Exception) as ei:
        client.ingest_schedule_event(**{**base, "user_id": ""})
    assert "INVALID_INPUT" in _err_text(ei.value)
    # empty title
    with pytest.raises(Exception) as ei:
        client.ingest_schedule_event(**{**base, "title": ""})
    assert "INVALID_INPUT" in _err_text(ei.value)
    # unknown status
    with pytest.raises(Exception) as ei:
        client.ingest_schedule_event(**{**base, "status": "maybe"})
    assert "INVALID_INPUT" in _err_text(ei.value)
    # payload not an object
    with pytest.raises(Exception) as ei:
        client.ingest_schedule_event(**{**base, "payload": "not-an-object"})
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# correct_schedule_event
# ---------------------------------------------------------------------------


def _mk(client: RoosterClient, user_id="u1", dedupe="c-k", t="2025-04-12T10:00:00",
        tz="UTC", source="manual", status="planned", title="Base",
        payload=None):
    return client.ingest_schedule_event(
        user_id=user_id, dedupe_key=dedupe, occurred_at=t, timezone=tz,
        source=source, status=status, title=title,
        payload=payload if payload is not None else {},
    )


def test_correct_creates_successor_with_inheritance(client: RoosterClient):
    orig = _mk(client, dedupe="corr-1", source="google_calendar")
    succ = client.correct_schedule_event(
        event_id=orig["event_id"],
        occurred_at="2025-04-12T11:00:00",
        timezone="UTC",
        status="occurred",
        title="Corrected title",
    )
    # chain wiring
    assert succ["supersedes"] == orig["event_id"]
    assert succ["superseded_by"] is None
    assert succ["event_id"] != orig["event_id"]
    # inheritance from target (Lead ruling): track, user_id, source
    assert succ["track"] == "schedule"
    assert succ["user_id"] == orig["user_id"]
    assert succ["source"] == "google_calendar"  # source=target.source
    # new content from input
    assert succ["title"] == "Corrected title"
    assert succ["status"] == "occurred"
    assert _parse_utc(succ["occurred_at_utc"]) == dt.datetime(2025, 4, 12, 11, 0)
    # fresh auto-generated dedupe_key (not the client key, non-null)
    assert succ["dedupe_key"] is not None
    assert succ["dedupe_key"] != orig["dedupe_key"]
    # retraction defaults
    assert succ["retracted"] is False
    assert succ["retracted_at"] is None
    assert succ["retracted_reason"] is None
    # original now superseded (visible via get)
    got = client.get_schedule_event(event_id=orig["event_id"])
    assert got["superseded_by"] == succ["event_id"]


def test_correct_content_idempotency_returns_existing(client: RoosterClient):
    orig = _mk(client, dedupe="corr-2")
    c1 = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
        timezone="UTC", status="planned", title="Same correction",
    )
    # identical re-submission must return the existing correction unchanged
    # (Lead ruling: content idempotency evaluated before superseded_by check)
    c2 = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
        timezone="UTC", status="planned", title="Same correction",
    )
    assert c2["event_id"] == c1["event_id"]
    assert c2["created_at"] == c1["created_at"]
    # no second successor was created: original superseded_by points at c1
    got = client.get_schedule_event(event_id=orig["event_id"])
    assert got["superseded_by"] == c1["event_id"]


def test_correct_already_superseded_with_payload(client: RoosterClient):
    orig = _mk(client, dedupe="corr-3")
    c1 = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T13:00:00",
        timezone="UTC", status="planned", title="V1",
    )
    # different content against the now-superseded original -> EVENT_ALREADY_SUPERSEDED
    with pytest.raises(Exception) as ei:
        client.correct_schedule_event(
            event_id=orig["event_id"], occurred_at="2025-04-12T14:00:00",
            timezone="UTC", status="occurred", title="V2 different",
        )
    err = _err_text(ei.value)
    assert "EVENT_ALREADY_SUPERSEDED" in err
    # error payload MUST include target.superseded_by so caller can re-issue
    assert c1["event_id"] in err


def test_correct_chain_multiple_versions(client: RoosterClient):
    orig = _mk(client, dedupe="corr-4")
    c1 = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T15:00:00",
        timezone="UTC", status="planned", title="V1",
    )
    c2 = client.correct_schedule_event(
        event_id=c1["event_id"], occurred_at="2025-04-12T16:00:00",
        timezone="UTC", status="occurred", title="V2",
    )
    assert c2["supersedes"] == c1["event_id"]
    got_c1 = client.get_schedule_event(event_id=c1["event_id"])
    assert got_c1["superseded_by"] == c2["event_id"]


def test_correct_event_not_found(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.correct_schedule_event(
            event_id=str(uuid.uuid4()), occurred_at="2025-04-12T12:00:00",
            timezone="UTC", status="planned", title="X",
        )
    assert "EVENT_NOT_FOUND" in _err_text(ei.value)


def test_correct_retracted_rejected(client: RoosterClient):
    orig = _mk(client, dedupe="corr-5")
    client.retract_schedule_event(event_id=orig["event_id"], reason="wrong")
    with pytest.raises(Exception) as ei:
        client.correct_schedule_event(
            event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
            timezone="UTC", status="planned", title="X",
        )
    assert "EVENT_RETRACTED" in _err_text(ei.value)


def test_correct_invalid_input_and_timezone(client: RoosterClient):
    orig = _mk(client, dedupe="corr-6")
    # unknown status
    with pytest.raises(Exception) as ei:
        client.correct_schedule_event(
            event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
            timezone="UTC", status="maybe", title="X",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)
    # unknown timezone
    with pytest.raises(Exception) as ei:
        client.correct_schedule_event(
            event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
            timezone="Bad/Zone", status="planned", title="X",
        )
    assert "UNKNOWN_TIMEZONE" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# retract_schedule_event
# ---------------------------------------------------------------------------


def test_retract_marks_record_and_keeps_fields(client: RoosterClient):
    orig = _mk(client, dedupe="retr-1")
    ret = client.retract_schedule_event(event_id=orig["event_id"], reason="user cancel")
    assert ret["retracted"] is True
    assert ret["retracted_at"] is not None
    assert _parse_utc(ret["retracted_at"]) <= dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(minutes=1)
    assert ret["retracted_reason"] == "user cancel"
    # dedupe_key / supersedes / superseded_by unchanged
    assert ret["dedupe_key"] == orig["dedupe_key"]
    assert ret["supersedes"] == orig["supersedes"]
    assert ret["superseded_by"] == orig["superseded_by"]
    # updated_at == retracted_at on retraction
    assert _parse_utc(ret["updated_at"]) == _parse_utc(ret["retracted_at"])


def test_retract_without_reason_null(client: RoosterClient):
    orig = _mk(client, dedupe="retr-2")
    ret = client.retract_schedule_event(event_id=orig["event_id"])
    assert ret["retracted"] is True
    assert ret["retracted_reason"] is None


def test_retract_idempotent_preserves_retracted_at(client: RoosterClient):
    orig = _mk(client, dedupe="retr-3")
    first = client.retract_schedule_event(event_id=orig["event_id"], reason="r1")
    second = client.retract_schedule_event(event_id=orig["event_id"], reason="r2")
    assert second["event_id"] == first["event_id"]
    assert second["retracted_at"] == first["retracted_at"]  # MUST NOT update
    assert second["retracted_reason"] == first["retracted_reason"]  # unchanged


def test_retract_event_not_found(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.retract_schedule_event(event_id=str(uuid.uuid4()))
    assert "EVENT_NOT_FOUND" in _err_text(ei.value)


def test_retract_invalid_reason_type(client: RoosterClient):
    orig = _mk(client, dedupe="retr-4")
    with pytest.raises(Exception) as ei:
        client.retract_schedule_event(event_id=orig["event_id"], reason=123)
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# list_schedule_events
# ---------------------------------------------------------------------------


def test_list_chronological_order(client: RoosterClient):
    # three events at known distinct UTC instants
    client.ingest_schedule_event(
        user_id="ul1", dedupe_key="l-1", occurred_at="2025-04-10T08:00:00",
        timezone="UTC", source="manual", status="planned", title="early",
    )
    client.ingest_schedule_event(
        user_id="ul1", dedupe_key="l-2", occurred_at="2025-04-11T08:00:00",
        timezone="UTC", source="manual", status="planned", title="middle",
    )
    client.ingest_schedule_event(
        user_id="ul1", dedupe_key="l-3", occurred_at="2025-04-12T08:00:00",
        timezone="UTC", source="manual", status="planned", title="late",
    )
    res = client.list_schedule_events(
        user_id="ul1",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert isinstance(res, list)
    titles = [r["title"] for r in res]
    assert titles == ["early", "middle", "late"]
    stamps = [_parse_utc(r["occurred_at_utc"]) for r in res]
    assert stamps == sorted(stamps)


def test_list_window_inclusive_start_exclusive_end(client: RoosterClient):
    client.ingest_schedule_event(
        user_id="ul2", dedupe_key="w-1", occurred_at="2025-04-10T00:00:00",
        timezone="UTC", source="manual", status="planned", title="at-start",
    )
    client.ingest_schedule_event(
        user_id="ul2", dedupe_key="w-2", occurred_at="2025-04-20T00:00:00",
        timezone="UTC", source="manual", status="planned", title="at-end",
    )
    res = client.list_schedule_events(
        user_id="ul2",
        window_start="2025-04-10T00:00:00Z",  # inclusive
        window_end="2025-04-20T00:00:00Z",    # exclusive
    )
    titles = [r["title"] for r in res]
    assert titles == ["at-start"]  # at-end excluded (== window_end)


def test_list_empty_window_and_user_isolation(client: RoosterClient):
    client.ingest_schedule_event(
        user_id="ul3", dedupe_key="iso-1", occurred_at="2025-04-10T08:00:00",
        timezone="UTC", source="manual", status="planned", title="mine",
    )
    # no events in an empty window
    res_empty = client.list_schedule_events(
        user_id="ul3",
        window_start="2026-01-01T00:00:00Z",
        window_end="2026-02-01T00:00:00Z",
    )
    assert res_empty == []
    # other user sees nothing
    res_other = client.list_schedule_events(
        user_id="nobody",
        window_start="2025-01-01T00:00:00Z",
        window_end="2026-01-01T00:00:00Z",
    )
    assert res_other == []


def test_list_excludes_history_by_default(client: RoosterClient):
    orig = _mk(client, user_id="ul4", dedupe="h-1")
    succ = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T10:30:00",
        timezone="UTC", status="occurred", title="head",
    )
    client.retract_schedule_event(event_id=orig["event_id"])  # retract superseded
    res = client.list_schedule_events(
        user_id="ul4",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    ids = {r["event_id"] for r in res}
    assert succ["event_id"] in ids          # head shown
    assert orig["event_id"] not in ids      # superseded + retracted hidden


def test_list_include_history_true(client: RoosterClient):
    orig = _mk(client, user_id="ul5", dedupe="h-2")
    succ = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T10:30:00",
        timezone="UTC", status="occurred", title="head",
    )
    client.retract_schedule_event(event_id=succ["event_id"])  # retract the head
    res = client.list_schedule_events(
        user_id="ul5",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
        include_history=True,
    )
    ids = {r["event_id"] for r in res}
    assert orig["event_id"] in ids
    assert succ["event_id"] in ids
    # and with include_history=false (default) both are hidden now
    res_default = client.list_schedule_events(
        user_id="ul5",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert res_default == []


def test_list_retract_head_hides_chain(client: RoosterClient):
    orig = _mk(client, user_id="ul6", dedupe="h-3")
    succ = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T10:30:00",
        timezone="UTC", status="occurred", title="head",
    )
    client.retract_schedule_event(event_id=succ["event_id"])  # retract head
    res = client.list_schedule_events(
        user_id="ul6",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert res == []


def test_list_invalid_input(client: RoosterClient):
    # window_end <= window_start
    with pytest.raises(Exception) as ei:
        client.list_schedule_events(
            user_id="u1",
            window_start="2025-05-01T00:00:00Z",
            window_end="2025-04-01T00:00:00Z",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)
    # missing user_id
    with pytest.raises(Exception) as ei:
        client.list_schedule_events(
            user_id="",
            window_start="2025-04-01T00:00:00Z",
            window_end="2025-05-01T00:00:00Z",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# get_schedule_event
# ---------------------------------------------------------------------------


def test_get_returns_record_as_is(client: RoosterClient):
    orig = _mk(client, dedupe="get-1")
    got = client.get_schedule_event(event_id=orig["event_id"])
    assert got["event_id"] == orig["event_id"]
    assert got["title"] == orig["title"]
    assert got["superseded_by"] is None


def test_get_resolve_head_walks_chain(client: RoosterClient):
    orig = _mk(client, dedupe="get-2")
    c1 = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
        timezone="UTC", status="planned", title="V1",
    )
    c2 = client.correct_schedule_event(
        event_id=c1["event_id"], occurred_at="2025-04-12T13:00:00",
        timezone="UTC", status="occurred", title="V2",
    )
    head = client.get_schedule_event(event_id=orig["event_id"], resolve_head=True)
    assert head["event_id"] == c2["event_id"]
    assert head["title"] == "V2"


def test_get_resolve_head_stops_at_retracted(client: RoosterClient):
    orig = _mk(client, dedupe="get-3")
    c1 = client.correct_schedule_event(
        event_id=orig["event_id"], occurred_at="2025-04-12T12:00:00",
        timezone="UTC", status="planned", title="V1",
    )
    client.retract_schedule_event(event_id=c1["event_id"], reason="nope")
    head = client.get_schedule_event(event_id=orig["event_id"], resolve_head=True)
    # chain walk stops at the first retracted event and returns it
    assert head["event_id"] == c1["event_id"]
    assert head["retracted"] is True


def test_get_event_not_found(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.get_schedule_event(event_id=str(uuid.uuid4()))
    assert "EVENT_NOT_FOUND" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# helper: extract error text from any error representation
# ---------------------------------------------------------------------------


def _err_text(exc: BaseException) -> str:
    """Extract contract error-code text from any error representation.

    The invocation artifact documents typed exceptions whose class names are
    camelCase variants of the frozen contract's SCREAMING_SNAKE error codes
    (e.g. EventAlreadySupersededError -> EVENT_ALREADY_SUPERSEDED). We include
    both the raw exception text and the normalized class name.
    """
    parts = [str(exc)]
    cls = type(exc).__name__
    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cls)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake).upper()
    snake = re.sub(r"_ERROR$", "", snake)
    parts.append(snake)
    d = getattr(exc, "__dict__", {})
    for k, v in d.items():
        parts.append(str(v))
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    return " | ".join(parts)

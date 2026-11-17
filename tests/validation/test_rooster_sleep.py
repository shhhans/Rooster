"""Black-box validation tests for the SLEEP-001 sleep interval track.

Frozen contract: ROOSTER-SLEEP, revision 9 (frozen after Validator ACK 2026-10-15).
Lead ruling applied: none pending; validation.command runs pytest via Windows
conda run -n 205agent (this file is run by the Lead on the Windows 205agent
environment; the Validator writes these tests independently from the frozen
contract without reading implementation source).

Interfaces under test (public_interfaces of the frozen contract):
  ingest_sleep_event   (import; entrypoint app.invocation.ingest_sleep_event)
  correct_sleep_event  (import; app.invocation.correct_sleep_event)
  retract_sleep_event  (import; app.invocation.retract_sleep_event)
  list_sleep_events    (import; app.invocation.list_sleep_events)
  get_sleep_event      (import; app.invocation.get_sleep_event)
  reset_isolated_db    (existing reset hook, unmodified)
  HTTP: POST/GET /timeline/sleep/events, GET /timeline/sleep/events/{event_id},
        POST /timeline/sleep/events/{event_id}/correct,
        POST /timeline/sleep/events/{event_id}/retract

Scope notes (from the frozen contract):
  * Sleep events carry track='sleep', title='sleep', status='occurred'
    (hard-coded), source preserved verbatim, timezone at event level, payload
    {sleep_start, sleep_end} as NAIVE LOCAL strings (no zone designator).
  * occurred_at_utc / local_time are anchored to sleep_start; the window query
    sorts by occurred_at_utc (the sleep_start_utc).
  * Idempotency: ingest by (track='sleep', user_id, dedupe_key); correct by
    content (occurred_at_utc, payload); retract idempotent.
  * DST: a sleep_start in a spring-forward gap is UNKNOWN_TIMEZONE; an
    ambiguous fall-back start resolves to the EARLIER UTC instant (fold 0).
  * No regression: the existing schedule-track suites (ROOSTER-002/010/011)
    must still pass under the same validation command.

These tests are written strictly from the frozen contract (black box). They do
not read or depend on implementation internals.
"""

import datetime as dt
import json
import re
import uuid

import pytest

from rooster_client import RoosterClient


class SleepClient(RoosterClient):
    """RoosterClient extended with the sleep-track interfaces.

    The shared RoosterClient only knows the five schedule-track interfaces;
    the sleep interfaces are resolved through the same generic `_call`
    transport wrapper (mirrors how ROOSTER-011 added classify_* via _call).
    """

    def ingest_sleep_event(self, **payload):
        return self._call("ingest_sleep_event", payload)

    def correct_sleep_event(self, **payload):
        return self._call("correct_sleep_event", payload)

    def retract_sleep_event(self, **payload):
        return self._call("retract_sleep_event", payload)

    def list_sleep_events(self, **payload):
        return self._call("list_sleep_events", payload)

    def get_sleep_event(self, **payload):
        return self._call("get_sleep_event", payload)

UTC = dt.timezone.utc

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path) -> RoosterClient:
    """Hermetic per-test binding: isolated DB + direct-import invocation surface.

    Mirrors the frozen validation.command env contract (ROOSTER_IMPORT_PATH /
    ROOSTER_RESET_FN) and the invocation artifact's recommended isolated-DB fixture.
    """
    monkeypatch.setenv("ROOSTER_IMPORT_PATH", "app.invocation")
    monkeypatch.setenv("ROOSTER_RESET_FN", "app.invocation.reset_isolated_db")
    monkeypatch.setenv("ROOSTER_TIMELINE_DB", str(tmp_path / "kernel.db"))
    import app.invocation as invocation
    invocation.reset_isolated_db()
    return SleepClient()


@pytest.fixture
def http_env(monkeypatch, tmp_path):
    """HTTP-surface binding sharing one isolated DB with the import surface.

    Returns (RoosterClient, TestClient) so tests can ingest via the public sleep
    interfaces and then exercise the /timeline/sleep/events routes against the
    same store.
    """
    monkeypatch.setenv("ROOSTER_IMPORT_PATH", "app.invocation")
    monkeypatch.setenv("ROOSTER_RESET_FN", "app.invocation.reset_isolated_db")
    monkeypatch.setenv("ROOSTER_TIMELINE_DB", str(tmp_path / "kernel.db"))
    import app.invocation as invocation
    invocation.reset_isolated_db()
    from app.main import create_app
    from fastapi.testclient import TestClient

    rc = SleepClient()
    tc = TestClient(create_app(), raise_server_exceptions=False)
    tc.__enter__()  # trigger app lifespan/startup so app.state.timeline_service is bound
    try:
        yield rc, tc
    finally:
        tc.__exit__(None, None, None)
    return rc, tc


def _ingest(client: RoosterClient, user_id="u1", dedupe="s-1",
            sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
            timezone="UTC", source="manual"):
    return client._call("ingest_sleep_event", dict(
        user_id=user_id, dedupe_key=dedupe, sleep_start=sleep_start,
        sleep_end=sleep_end, timezone=timezone, source=source,
    ))


def _correct(client: RoosterClient, event_id, sleep_start, sleep_end, timezone="UTC"):
    return client._call("correct_sleep_event", dict(
        event_id=event_id, sleep_start=sleep_start, sleep_end=sleep_end,
        timezone=timezone,
    ))


def _retract(client: RoosterClient, event_id, reason=None):
    payload = {"event_id": event_id}
    if reason is not None:
        payload["reason"] = reason
    return client._call("retract_sleep_event", payload)


def _list(client: RoosterClient, user_id, window_start, window_end,
          include_history=False):
    payload = {"user_id": user_id, "window_start": window_start,
               "window_end": window_end, "include_history": include_history}
    return client._call("list_sleep_events", payload)


def _get(client: RoosterClient, event_id, resolve_head=False):
    return client._call("get_sleep_event",
                        {"event_id": event_id, "resolve_head": resolve_head})



def _err_text(exc: BaseException) -> str:
    """Extract contract error-code text from any error representation (same
    approach as the ROOSTER-002/010/011 suites)."""
    parts = [str(exc)]
    cls = type(exc).__name__
    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cls)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake).upper()
    snake = re.sub(r"_ERROR$", "", snake)
    parts.append(snake)
    for k, v in getattr(exc, "__dict__", {}).items():
        parts.append(str(v))
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    return " | ".join(parts)


def _parse_utc(s: str) -> dt.datetime:
    """Parse an ISO-8601 UTC string with Z suffix (or +00:00)."""
    s = s.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(s).replace(tzinfo=None)


def _iso_z(d: dt.datetime) -> str:
    """Format a naive UTC datetime as ISO-8601 with Z suffix (seconds)."""
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# ingest_sleep_event: unified event shape
# ---------------------------------------------------------------------------


def test_ingest_full_unified_shape(client: RoosterClient):
    rec = _ingest(client, user_id="u1", dedupe="k1", sleep_start="2025-04-12T22:00:00",
                  sleep_end="2025-04-13T06:00:00", timezone="UTC", source="apple-health")
    assert rec["track"] == "sleep"
    assert rec["user_id"] == "u1"
    assert rec["dedupe_key"] == "k1"
    assert rec["source"] == "apple-health"
    assert rec["status"] == "occurred"          # hard-coded
    assert rec["title"] == "sleep"              # hard-coded
    # payload holds NAIVE local strings, no zone designator
    assert rec["payload"] == {"sleep_start": "2025-04-12T22:00:00",
                              "sleep_end": "2025-04-13T06:00:00"}
    assert rec["timezone"] == "UTC"
    # UTC anchor = sleep_start
    assert rec["occurred_at_utc"] == "2025-04-12T22:00:00Z"
    assert rec["occurred_at_utc"].endswith("Z")
    assert rec["local_time"] == "2025-04-12T22:00:00"
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


def test_ingest_timezone_normalization_ny(client: RoosterClient):
    # 2025-04-12 is EDT (UTC-4): 22:00 EDT -> 2025-04-13T02:00:00Z
    rec = _ingest(client, user_id="u-tz-ny", dedupe="k-ny",
                  sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                  timezone="America/New_York", source="manual")
    assert rec["occurred_at_utc"] == "2025-04-13T02:00:00Z"
    assert rec["local_time"] == "2025-04-12T22:00:00"
    assert rec["timezone"] == "America/New_York"


def test_ingest_timezone_normalization_tokyo(client: RoosterClient):
    # Asia/Tokyo is UTC+9 (no DST): 22:00 JST -> 2025-04-12T13:00:00Z
    rec = _ingest(client, user_id="u-tz-tokyo", dedupe="k-tokyo",
                  sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T07:00:00",
                  timezone="Asia/Tokyo", source="oura-import")
    assert rec["occurred_at_utc"] == "2025-04-12T13:00:00Z"
    assert rec["timezone"] == "Asia/Tokyo"
    assert rec["source"] == "oura-import"


def test_ingest_payload_naive_no_offset(client: RoosterClient):
    # zone designator on sleep_start is rejected as INVALID_INPUT (naive only)
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-zone", dedupe="k-zone",
                sleep_start="2025-04-12T22:00:00-04:00", sleep_end="2025-04-13T06:00:00",
                timezone="America/New_York", source="manual")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_ingest_end_with_zone_designator_rejected(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-zone2", dedupe="k-zone2",
                sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00-04:00",
                timezone="America/New_York", source="manual")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_ingest_invalid_timestamp(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-bad", dedupe="k-bad",
                sleep_start="not-a-timestamp", sleep_end="2025-04-13T06:00:00",
                timezone="UTC", source="manual")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_ingest_zero_length_interval_invalid(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-zero", dedupe="k-zero",
                sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-12T22:00:00",
                timezone="UTC", source="manual")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_ingest_inverted_interval_invalid(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-inv", dedupe="k-inv",
                sleep_start="2025-04-13T06:00:00", sleep_end="2025-04-12T22:00:00",
                timezone="UTC", source="manual")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_ingest_end_not_strictly_after_start_across_tz(client: RoosterClient):
    # sleep_end normalized to UTC must be strictly greater than sleep_start_utc
    # 22:00 EDT = 02:00Z; 23:00 EDT = 03:00Z -> valid. Use equal-UTC case:
    # sleep_start 22:00 EDT (02:00Z), sleep_end 22:00 EDT same instant is equal;
    # but naive equal strings are caught as zero-length. Here test a case where
    # naive end > naive start but UTC equal is impossible with valid offsets in
    # the same zone; instead verify end strictly-after start in naive terms.
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-edge", dedupe="k-edge",
                sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-12T22:00:00",
                timezone="America/New_York", source="manual")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_ingest_unknown_timezone(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-badtz", dedupe="k-badtz",
                sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                timezone="Not/AZone", source="manual")
    assert "UNKNOWN_TIMEZONE" in _err_text(ei.value)


def test_ingest_dst_gap_rejected(client: RoosterClient):
    # America/New_York spring-forward 2025-03-09: 02:00-03:00 does not exist.
    # A sleep_start in the gap is UNKNOWN_TIMEZONE per the frozen contract.
    with pytest.raises(Exception) as ei:
        _ingest(client, user_id="u-gap", dedupe="k-gap",
                sleep_start="2025-03-09T02:30:00", sleep_end="2025-03-09T06:00:00",
                timezone="America/New_York", source="manual")
    assert "UNKNOWN_TIMEZONE" in _err_text(ei.value)


def test_ingest_dst_fallback_fold0_earlier_utc(client: RoosterClient):
    # America/New_York fall-back 2025-11-02: 01:30 local is ambiguous.
    # Contract: resolved to the EARLIER UTC instant (fold 0) = 05:30Z.
    rec = _ingest(client, user_id="u-fold", dedupe="k-fold",
                  sleep_start="2025-11-02T01:30:00", sleep_end="2025-11-02T05:00:00",
                  timezone="America/New_York", source="manual")
    assert rec["occurred_at_utc"] == "2025-11-02T05:30:00Z"   # fold 0, earlier
    assert rec["local_time"] == "2025-11-02T01:30:00"


def test_ingest_empty_required_fields(client: RoosterClient):
    base = dict(user_id="u1", dedupe_key="k-empty", sleep_start="2025-04-12T22:00:00",
                sleep_end="2025-04-13T06:00:00", timezone="UTC", source="manual")
    for field in ("user_id", "dedupe_key", "source", "timezone"):
        payload = dict(base)
        payload[field] = ""
        with pytest.raises(Exception) as ei:
            client.ingest_sleep_event(**payload)
        assert "INVALID_INPUT" in _err_text(ei.value), field


def test_ingest_idempotent_returns_original_verbatim(client: RoosterClient):
    first = _ingest(client, user_id="u1", dedupe="k-dup",
                    sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                    timezone="UTC", source="manual")
    second = _ingest(client, user_id="u1", dedupe="k-dup",
                     sleep_start="2025-04-13T23:00:00", sleep_end="2025-04-14T07:00:00",
                     timezone="Asia/Tokyo", source="apple-health")
    assert second == first
    assert second["event_id"] == first["event_id"]
    assert second["payload"] == {"sleep_start": "2025-04-12T22:00:00",
                                 "sleep_end": "2025-04-13T06:00:00"}
    assert second["source"] == "manual"
    assert second["created_at"] == first["created_at"]


def test_ingest_idempotency_scoped_to_user(client: RoosterClient):
    a = _ingest(client, user_id="uA", dedupe="same",
                sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                timezone="UTC", source="manual")
    b = _ingest(client, user_id="uB", dedupe="same",
                sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                timezone="UTC", source="manual")
    assert a["event_id"] != b["event_id"]
    assert a["user_id"] == "uA"
    assert b["user_id"] == "uB"


def test_ingest_separate_from_schedule_track(client: RoosterClient):
    # same dedupe_key across tracks must NOT collide (track is part of the key)
    sleep_rec = _ingest(client, user_id="u-x", dedupe="shared",
                        sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                        timezone="UTC", source="manual")
    sched_rec = client.ingest_schedule_event(
        user_id="u-x", dedupe_key="shared", occurred_at="2025-04-12T22:00:00",
        timezone="UTC", source="manual", status="occurred", title="flight",
        payload={},
    )
    assert sleep_rec["event_id"] != sched_rec["event_id"]
    assert sleep_rec["track"] == "sleep"
    assert sched_rec["track"] == "schedule"


# ---------------------------------------------------------------------------
# correct_sleep_event
# ---------------------------------------------------------------------------


def test_correct_creates_successor_with_inheritance(client: RoosterClient):
    orig = _ingest(client, user_id="u-corr", dedupe="c-1",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="apple-health")
    succ = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    # chain wiring
    assert succ["supersedes"] == orig["event_id"]
    assert succ["superseded_by"] is None
    assert succ["event_id"] != orig["event_id"]
    # inheritance from target (Lead ruling pattern from schedule track):
    # user_id and source inherited; title/status hard-coded for sleep
    assert succ["track"] == "sleep"
    assert succ["user_id"] == orig["user_id"]
    assert succ["source"] == "apple-health"
    assert succ["title"] == "sleep"
    assert succ["status"] == "occurred"
    # new content from input
    assert succ["payload"] == {"sleep_start": "2025-04-12T23:00:00",
                               "sleep_end": "2025-04-13T07:00:00"}
    assert succ["occurred_at_utc"] == "2025-04-12T23:00:00Z"
    assert succ["local_time"] == "2025-04-12T23:00:00"
    assert succ["timezone"] == "UTC"
    # fresh auto-generated dedupe_key (not the client key, non-null)
    assert succ["dedupe_key"] is not None
    assert succ["dedupe_key"] != orig["dedupe_key"]
    # original now superseded (visible via get)
    got = client.get_sleep_event(event_id=orig["event_id"])
    assert got["superseded_by"] == succ["event_id"]


def test_correct_content_idempotency_returns_existing(client: RoosterClient):
    orig = _ingest(client, user_id="u-corr2", dedupe="c-2",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    c1 = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:30:00", sleep_end="2025-04-13T07:30:00",
        timezone="UTC",
    )
    # identical re-submission returns the existing correction unchanged
    c2 = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:30:00", sleep_end="2025-04-13T07:30:00",
        timezone="UTC",
    )
    assert c2["event_id"] == c1["event_id"]
    assert c2["created_at"] == c1["created_at"]
    # no second successor created: original still points at c1
    got = client.get_sleep_event(event_id=orig["event_id"])
    assert got["superseded_by"] == c1["event_id"]


def test_correct_already_superseded(client: RoosterClient):
    orig = _ingest(client, user_id="u-corr3", dedupe="c-3",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    c1 = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    # different content against the now-superseded original -> EVENT_ALREADY_SUPERSEDED
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id=orig["event_id"],
            sleep_start="2025-04-12T23:30:00", sleep_end="2025-04-13T07:30:00",
            timezone="UTC",
        )
    err = _err_text(ei.value)
    assert "EVENT_ALREADY_SUPERSEDED" in err
    # error payload MUST include target.superseded_by so caller can re-issue
    assert c1["event_id"] in err


def test_correct_chain_multiple_versions(client: RoosterClient):
    orig = _ingest(client, user_id="u-corr4", dedupe="c-4",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    c1 = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    c2 = client.correct_sleep_event(
        event_id=c1["event_id"],
        sleep_start="2025-04-12T23:30:00", sleep_end="2025-04-13T07:30:00",
        timezone="UTC",
    )
    assert c2["supersedes"] == c1["event_id"]
    got_c1 = client.get_sleep_event(event_id=c1["event_id"])
    assert got_c1["superseded_by"] == c2["event_id"]


def test_correct_event_not_found(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id=str(uuid.uuid4()),
            sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
            timezone="UTC",
        )
    assert "EVENT_NOT_FOUND" in _err_text(ei.value)


def test_correct_retracted_rejected(client: RoosterClient):
    orig = _ingest(client, user_id="u-corr5", dedupe="c-5",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    client.retract_sleep_event(event_id=orig["event_id"], reason="wrong")
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id=orig["event_id"],
            sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
            timezone="UTC",
        )
    assert "EVENT_RETRACTED" in _err_text(ei.value)


def test_correct_invalid_input_and_timezone(client: RoosterClient):
    orig = _ingest(client, user_id="u-corr6", dedupe="c-6",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    # empty event_id
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id="",
            sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
            timezone="UTC",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)
    # unknown timezone
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id=orig["event_id"],
            sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
            timezone="Bad/Zone",
        )
    assert "UNKNOWN_TIMEZONE" in _err_text(ei.value)
    # inverted interval
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id=orig["event_id"],
            sleep_start="2025-04-13T07:00:00", sleep_end="2025-04-12T23:00:00",
            timezone="UTC",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)
    # zone designator on start
    with pytest.raises(Exception) as ei:
        client.correct_sleep_event(
            event_id=orig["event_id"],
            sleep_start="2025-04-12T23:00:00-04:00", sleep_end="2025-04-13T07:00:00",
            timezone="UTC",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# retract_sleep_event
# ---------------------------------------------------------------------------


def test_retract_marks_record_and_keeps_fields(client: RoosterClient):
    orig = _ingest(client, user_id="u-retr", dedupe="r-1",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    ret = client.retract_sleep_event(event_id=orig["event_id"], reason="user cancel")
    assert ret["retracted"] is True
    assert ret["retracted_at"] is not None
    assert ret["retracted_at"].endswith("Z")
    assert _parse_utc(ret["retracted_at"]) <= dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(minutes=1)
    assert ret["retracted_reason"] == "user cancel"
    # dedupe_key / supersedes / superseded_by unchanged
    assert ret["dedupe_key"] == orig["dedupe_key"]
    assert ret["supersedes"] == orig["supersedes"]
    assert ret["superseded_by"] == orig["superseded_by"]
    # updated_at == retracted_at on retraction
    assert _parse_utc(ret["updated_at"]) == _parse_utc(ret["retracted_at"])


def test_retract_without_reason_null(client: RoosterClient):
    orig = _ingest(client, user_id="u-retr2", dedupe="r-2",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    ret = client.retract_sleep_event(event_id=orig["event_id"])
    assert ret["retracted"] is True
    assert ret["retracted_reason"] is None


def test_retract_idempotent_preserves_retracted_at(client: RoosterClient):
    orig = _ingest(client, user_id="u-retr3", dedupe="r-3",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    first = client.retract_sleep_event(event_id=orig["event_id"], reason="r1")
    second = client.retract_sleep_event(event_id=orig["event_id"], reason="r2")
    assert second["event_id"] == first["event_id"]
    assert second["retracted_at"] == first["retracted_at"]   # MUST NOT update
    assert second["retracted_reason"] == first["retracted_reason"]  # unchanged


def test_retract_event_not_found(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.retract_sleep_event(event_id=str(uuid.uuid4()))
    assert "EVENT_NOT_FOUND" in _err_text(ei.value)


def test_retract_invalid_reason_type(client: RoosterClient):
    orig = _ingest(client, user_id="u-retr4", dedupe="r-4",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    with pytest.raises(Exception) as ei:
        client.retract_sleep_event(event_id=orig["event_id"], reason=123)
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_retract_empty_event_id(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.retract_sleep_event(event_id="")
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# list_sleep_events
# ---------------------------------------------------------------------------


def test_list_chronological_order(client: RoosterClient):
    _ingest(client, user_id="ul1", dedupe="l-1",
            sleep_start="2025-04-10T22:00:00", sleep_end="2025-04-11T06:00:00",
            timezone="UTC", source="manual")
    _ingest(client, user_id="ul1", dedupe="l-2",
            sleep_start="2025-04-11T22:00:00", sleep_end="2025-04-12T06:00:00",
            timezone="UTC", source="manual")
    _ingest(client, user_id="ul1", dedupe="l-3",
            sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
            timezone="UTC", source="manual")
    res = client.list_sleep_events(
        user_id="ul1",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert isinstance(res, list)
    starts = [r["occurred_at_utc"] for r in res]
    assert starts == ["2025-04-10T22:00:00Z", "2025-04-11T22:00:00Z", "2025-04-12T22:00:00Z"]
    assert starts == sorted(starts)


def test_list_sorted_by_occurred_at_utc_across_timezones(client: RoosterClient):
    # 23:00 Tokyo = 2025-04-12T14:00Z (same day); 22:00 NY EDT = 2025-04-13T02:00Z
    _ingest(client, user_id="ul-tz", dedupe="t-1",
            sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
            timezone="Asia/Tokyo", source="manual")     # 2025-04-12T14:00:00Z
    _ingest(client, user_id="ul-tz", dedupe="t-2",
            sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
            timezone="America/New_York", source="manual")  # 2025-04-13T02:00:00Z
    res = client.list_sleep_events(
        user_id="ul-tz",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert [r["occurred_at_utc"] for r in res] == [
        "2025-04-12T14:00:00Z",   # Tokyo start (earlier UTC)
        "2025-04-13T02:00:00Z",   # NY start
    ]


def test_list_window_inclusive_start_exclusive_end(client: RoosterClient):
    _ingest(client, user_id="ul2", dedupe="w-1",
            sleep_start="2025-04-10T22:00:00", sleep_end="2025-04-11T06:00:00",
            timezone="UTC", source="manual")
    _ingest(client, user_id="ul2", dedupe="w-2",
            sleep_start="2025-04-20T22:00:00", sleep_end="2025-04-21T06:00:00",
            timezone="UTC", source="manual")
    res = client.list_sleep_events(
        user_id="ul2",
        window_start="2025-04-10T22:00:00Z",  # inclusive
        window_end="2025-04-20T22:00:00Z",    # exclusive
    )
    starts = [r["occurred_at_utc"] for r in res]
    assert starts == ["2025-04-10T22:00:00Z"]  # at-end excluded (== window_end)


def test_list_empty_window_and_user_isolation(client: RoosterClient):
    _ingest(client, user_id="ul3", dedupe="iso-1",
            sleep_start="2025-04-10T22:00:00", sleep_end="2025-04-11T06:00:00",
            timezone="UTC", source="manual")
    # empty window
    res_empty = client.list_sleep_events(
        user_id="ul3",
        window_start="2026-01-01T00:00:00Z",
        window_end="2026-02-01T00:00:00Z",
    )
    assert res_empty == []
    # other user sees nothing
    res_other = client.list_sleep_events(
        user_id="nobody",
        window_start="2025-01-01T00:00:00Z",
        window_end="2026-01-01T00:00:00Z",
    )
    assert res_other == []


def test_list_only_sleep_track(client: RoosterClient):
    # schedule events must NOT leak into the sleep list
    _ingest(client, user_id="ul-x", dedupe="sl-1",
            sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
            timezone="UTC", source="manual")
    client.ingest_schedule_event(
        user_id="ul-x", dedupe_key="sc-1", occurred_at="2025-04-12T22:00:00",
        timezone="UTC", source="manual", status="occurred", title="flight",
        payload={},
    )
    res = client.list_sleep_events(
        user_id="ul-x",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert len(res) == 1
    assert res[0]["track"] == "sleep"


def test_list_excludes_history_by_default(client: RoosterClient):
    orig = _ingest(client, user_id="ul4", dedupe="h-1",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    succ = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    client.retract_sleep_event(event_id=orig["event_id"])  # retract superseded
    res = client.list_sleep_events(
        user_id="ul4",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    ids = {r["event_id"] for r in res}
    assert succ["event_id"] in ids          # head shown
    assert orig["event_id"] not in ids      # superseded + retracted hidden


def test_list_include_history_true(client: RoosterClient):
    orig = _ingest(client, user_id="ul5", dedupe="h-2",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    succ = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    client.retract_sleep_event(event_id=succ["event_id"])  # retract the head
    res = client.list_sleep_events(
        user_id="ul5",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
        include_history=True,
    )
    ids = {r["event_id"] for r in res}
    assert orig["event_id"] in ids
    assert succ["event_id"] in ids
    # default (include_history=False) hides both now
    res_default = client.list_sleep_events(
        user_id="ul5",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert res_default == []


def test_list_retract_head_hides_chain(client: RoosterClient):
    orig = _ingest(client, user_id="ul6", dedupe="h-3",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    succ = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    client.retract_sleep_event(event_id=succ["event_id"])  # retract head
    res = client.list_sleep_events(
        user_id="ul6",
        window_start="2025-04-01T00:00:00Z",
        window_end="2025-05-01T00:00:00Z",
    )
    assert res == []


def test_list_naive_window_interpreted_as_utc(client: RoosterClient):
    _ingest(client, user_id="ul7", dedupe="n-1",
            sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
            timezone="UTC", source="manual")
    # naive window strings are interpreted as UTC per the contract
    res = client.list_sleep_events(
        user_id="ul7",
        window_start="2025-04-12T00:00:00",
        window_end="2025-04-13T00:00:00",
    )
    assert [r["occurred_at_utc"] for r in res] == ["2025-04-12T22:00:00Z"]


def test_list_invalid_input(client: RoosterClient):
    # window_end <= window_start
    with pytest.raises(Exception) as ei:
        client.list_sleep_events(
            user_id="u1",
            window_start="2025-05-01T00:00:00Z",
            window_end="2025-04-01T00:00:00Z",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)
    # missing user_id
    with pytest.raises(Exception) as ei:
        client.list_sleep_events(
            user_id="",
            window_start="2025-04-01T00:00:00Z",
            window_end="2025-05-01T00:00:00Z",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)
    # include_history not a boolean
    with pytest.raises(Exception) as ei:
        client.list_sleep_events(
            user_id="u1",
            window_start="2025-04-01T00:00:00Z",
            window_end="2025-05-01T00:00:00Z",
            include_history="yes",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_list_invalid_timestamp(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.list_sleep_events(
            user_id="u1",
            window_start="not-a-date",
            window_end="2025-05-01T00:00:00Z",
        )
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# get_sleep_event
# ---------------------------------------------------------------------------


def test_get_returns_record_as_is(client: RoosterClient):
    orig = _ingest(client, user_id="u-get", dedupe="g-1",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    got = client.get_sleep_event(event_id=orig["event_id"])
    assert got["event_id"] == orig["event_id"]
    assert got["payload"] == orig["payload"]
    assert got["superseded_by"] is None


def test_get_resolve_head_walks_chain(client: RoosterClient):
    orig = _ingest(client, user_id="u-get2", dedupe="g-2",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    c1 = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    c2 = client.correct_sleep_event(
        event_id=c1["event_id"],
        sleep_start="2025-04-12T23:30:00", sleep_end="2025-04-13T07:30:00",
        timezone="UTC",
    )
    head = client.get_sleep_event(event_id=orig["event_id"], resolve_head=True)
    assert head["event_id"] == c2["event_id"]
    assert head["payload"]["sleep_start"] == "2025-04-12T23:30:00"


def test_get_resolve_head_stops_at_retracted(client: RoosterClient):
    orig = _ingest(client, user_id="u-get3", dedupe="g-3",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    c1 = client.correct_sleep_event(
        event_id=orig["event_id"],
        sleep_start="2025-04-12T23:00:00", sleep_end="2025-04-13T07:00:00",
        timezone="UTC",
    )
    client.retract_sleep_event(event_id=c1["event_id"], reason="nope")
    head = client.get_sleep_event(event_id=orig["event_id"], resolve_head=True)
    # chain walk stops at the first retracted event and returns it
    assert head["event_id"] == c1["event_id"]
    assert head["retracted"] is True


def test_get_event_not_found(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.get_sleep_event(event_id=str(uuid.uuid4()))
    assert "EVENT_NOT_FOUND" in _err_text(ei.value)


def test_get_empty_event_id_invalid(client: RoosterClient):
    with pytest.raises(Exception) as ei:
        client.get_sleep_event(event_id="")
    assert "INVALID_INPUT" in _err_text(ei.value)


def test_get_resolve_head_non_boolean_invalid(client: RoosterClient):
    orig = _ingest(client, user_id="u-get4", dedupe="g-4",
                   sleep_start="2025-04-12T22:00:00", sleep_end="2025-04-13T06:00:00",
                   timezone="UTC", source="manual")
    with pytest.raises(Exception) as ei:
        client.get_sleep_event(event_id=orig["event_id"], resolve_head="yes")
    assert "INVALID_INPUT" in _err_text(ei.value)


# ---------------------------------------------------------------------------
# HTTP surface: /timeline/sleep/events
# ---------------------------------------------------------------------------


def test_http_ingest_200_unified_event(http_env):
    rc, tc = http_env
    resp = tc.post("/timeline/sleep/events", json={
        "user_id": "u-http1", "dedupe_key": "h1",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-13T06:00:00",
        "timezone": "America/New_York", "source": "manual",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["track"] == "sleep"
    assert body["title"] == "sleep"
    assert body["status"] == "occurred"
    assert body["user_id"] == "u-http1"
    assert body["occurred_at_utc"] == "2025-04-13T02:00:00Z"
    assert body["local_time"] == "2025-04-12T22:00:00"
    assert body["timezone"] == "America/New_York"
    assert body["source"] == "user"
    assert body["confirmation"] == "accepted"
    assert body["realization"] == "actual"
    assert body["payload"] == {"sleep_start": "2025-04-12T22:00:00",
                               "sleep_end": "2025-04-13T06:00:00"}
    uuid.UUID(body["event_id"])


def test_http_ingest_idempotent_http(http_env):
    rc, tc = http_env
    payload = {
        "user_id": "u-http2", "dedupe_key": "h2",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-13T06:00:00",
        "timezone": "UTC", "source": "manual",
    }
    first = tc.post("/timeline/sleep/events", json=payload)
    second = tc.post("/timeline/sleep/events", json=payload)
    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["event_id"] == first.json()["event_id"]


def test_http_ingest_400_invalid_input(http_env):
    rc, tc = http_env
    resp = tc.post("/timeline/sleep/events", json={
        "user_id": "u-http3", "dedupe_key": "h3",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-12T22:00:00",
        "timezone": "UTC", "source": "manual",
    })
    assert resp.status_code == 400
    assert "INVALID_INPUT" in json.dumps(resp.json())


def test_http_ingest_400_unknown_timezone(http_env):
    rc, tc = http_env
    resp = tc.post("/timeline/sleep/events", json={
        "user_id": "u-http4", "dedupe_key": "h4",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-13T06:00:00",
        "timezone": "Not/AZone", "source": "manual",
    })
    assert resp.status_code == 400
    assert "UNKNOWN_TIMEZONE" in json.dumps(resp.json())


def test_http_ingest_400_dst_gap(http_env):
    rc, tc = http_env
    resp = tc.post("/timeline/sleep/events", json={
        "user_id": "u-http5", "dedupe_key": "h5",
        "sleep_start": "2025-03-09T02:30:00", "sleep_end": "2025-03-09T06:00:00",
        "timezone": "America/New_York", "source": "manual",
    })
    assert resp.status_code == 400
    assert "UNKNOWN_TIMEZONE" in json.dumps(resp.json())


def test_http_list_get_correct_retract_flow(http_env):
    rc, tc = http_env
    user = "u-http6"
    # ingest
    ev = tc.post("/timeline/sleep/events", json={
        "user_id": user, "dedupe_key": "h6",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-13T06:00:00",
        "timezone": "UTC", "source": "manual",
    }).json()
    ev_id = ev["event_id"]
    # get-one
    one = tc.get(f"/timeline/sleep/events/{ev_id}")
    assert one.status_code == 200
    assert one.json()["event_id"] == ev_id
    # list
    lst = tc.get("/timeline/sleep/events", params={
        "user_id": user,
        "window_start": "2025-04-01T00:00:00Z",
        "window_end": "2025-05-01T00:00:00Z",
    })
    assert lst.status_code == 200
    assert [e["event_id"] for e in lst.json()] == [ev_id]
    # correct
    corr = tc.post(f"/timeline/sleep/events/{ev_id}/correct", json={
        "sleep_start": "2025-04-12T23:00:00", "sleep_end": "2025-04-13T07:00:00",
        "timezone": "UTC",
    })
    assert corr.status_code == 200
    corr_body = corr.json()
    assert corr_body["event_id"] != ev_id
    assert corr_body["supersedes"] == ev_id
    # old event now superseded (visible via get)
    old_after = tc.get(f"/timeline/sleep/events/{ev_id}")
    assert old_after.json()["superseded_by"] == corr_body["event_id"]
    # resolve_head
    resolved = tc.get(f"/timeline/sleep/events/{ev_id}", params={"resolve_head": "true"})
    assert resolved.status_code == 200
    assert resolved.json()["event_id"] == corr_body["event_id"]
    # retract
    ret = tc.post(f"/timeline/sleep/events/{corr_body['event_id']}/retract",
                  json={"reason": "entered wrong"})
    assert ret.status_code == 200
    assert ret.json()["retracted"] is True
    assert ret.json()["retracted_reason"] == "entered wrong"
    # second retract idempotent
    ret2 = tc.post(f"/timeline/sleep/events/{corr_body['event_id']}/retract",
                   json={"reason": "again"})
    assert ret2.status_code == 200
    assert ret2.json()["retracted_at"] == ret.json()["retracted_at"]
    # history query includes retracted
    lsth = tc.get("/timeline/sleep/events", params={
        "user_id": user,
        "window_start": "2025-04-01T00:00:00Z",
        "window_end": "2025-05-01T00:00:00Z",
        "include_history": "true",
    })
    assert any(e["event_id"] == corr_body["event_id"] for e in lsth.json())
    # default list hides retracted head + superseded
    lst_default = tc.get("/timeline/sleep/events", params={
        "user_id": user,
        "window_start": "2025-04-01T00:00:00Z",
        "window_end": "2025-05-01T00:00:00Z",
    })
    assert lst_default.json() == []


def test_http_get_404_event_not_found(http_env):
    rc, tc = http_env
    resp = tc.get("/timeline/sleep/events/does-not-exist-xyz")
    assert resp.status_code == 404
    assert "EVENT_NOT_FOUND" in json.dumps(resp.json())


def test_http_correct_404_event_not_found(http_env):
    rc, tc = http_env
    resp = tc.post("/timeline/sleep/events/does-not-exist/correct", json={
        "sleep_start": "2025-04-12T23:00:00", "sleep_end": "2025-04-13T07:00:00",
        "timezone": "UTC",
    })
    assert resp.status_code == 404
    assert "EVENT_NOT_FOUND" in json.dumps(resp.json())


def test_http_correct_409_already_superseded(http_env):
    rc, tc = http_env
    ev = tc.post("/timeline/sleep/events", json={
        "user_id": "u-http7", "dedupe_key": "h7",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-13T06:00:00",
        "timezone": "UTC", "source": "manual",
    }).json()
    ev_id = ev["event_id"]
    c1 = tc.post(f"/timeline/sleep/events/{ev_id}/correct", json={
        "sleep_start": "2025-04-12T23:00:00", "sleep_end": "2025-04-13T07:00:00",
        "timezone": "UTC",
    }).json()
    resp = tc.post(f"/timeline/sleep/events/{ev_id}/correct", json={
        "sleep_start": "2025-04-12T23:30:00", "sleep_end": "2025-04-13T07:30:00",
        "timezone": "UTC",
    })
    assert resp.status_code == 409
    body = json.dumps(resp.json())
    assert "EVENT_ALREADY_SUPERSEDED" in body
    assert c1["event_id"] in body  # superseded_by present


def test_http_correct_409_event_retracted(http_env):
    rc, tc = http_env
    ev = tc.post("/timeline/sleep/events", json={
        "user_id": "u-http8", "dedupe_key": "h8",
        "sleep_start": "2025-04-12T22:00:00", "sleep_end": "2025-04-13T06:00:00",
        "timezone": "UTC", "source": "manual",
    }).json()
    tc.post(f"/timeline/sleep/events/{ev['event_id']}/retract", json={"reason": "x"})
    resp = tc.post(f"/timeline/sleep/events/{ev['event_id']}/correct", json={
        "sleep_start": "2025-04-12T23:00:00", "sleep_end": "2025-04-13T07:00:00",
        "timezone": "UTC",
    })
    assert resp.status_code == 409
    assert "EVENT_RETRACTED" in json.dumps(resp.json())


def test_http_retract_404_event_not_found(http_env):
    rc, tc = http_env
    resp = tc.post("/timeline/sleep/events/does-not-exist/retract", json={"reason": "x"})
    assert resp.status_code == 404
    assert "EVENT_NOT_FOUND" in json.dumps(resp.json())


def test_http_list_400_invalid_window(http_env):
    rc, tc = http_env
    resp = tc.get("/timeline/sleep/events", params={
        "user_id": "u-http9",
        "window_start": "2025-05-01T00:00:00Z",
        "window_end": "2025-04-01T00:00:00Z",
    })
    assert resp.status_code == 400
    assert "INVALID_INPUT" in json.dumps(resp.json())

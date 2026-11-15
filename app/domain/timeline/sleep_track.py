"""Sleep track — the second registered track of the timeline kernel.

All behavior here matches the frozen SLEEP-001 contract data flows for
``sleep_event_ingest``, ``sleep_event_correct``, ``sleep_event_retract``,
``sleep_events_query``, and ``sleep_event_get``.

The sleep track reuses the same ``ScheduleEvent`` unified model and the
same ``EventStore`` as the schedule track (the store filters by track
name), so no schema change is needed. Sleep-specific behavior:

* status is hard-coded to ``'occurred'`` (a reported sleep is always
  retrospective);
* title is hard-coded to ``'sleep'``;
* the payload carries the two naive local boundary strings
  ``{sleep_start, sleep_end}``;
* the event-level ``timezone`` holds the IANA zone the user reported the
  sleep in; ``occurred_at_utc`` / ``local_time`` are anchored on
  ``sleep_start``;
* both boundaries must be naive local ISO-8601 strings WITHOUT a zone
  designator (a trailing ``Z`` or an explicit offset is rejected as
  INVALID_INPUT — the zone travels in the event-level field);
* ``sleep_end`` must resolve to a UTC instant strictly after
  ``sleep_start``'s.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.timeline.errors import (
    EventAlreadySupersededError,
    EventNotFoundError,
    EventRetractedError,
    InvalidInputError,
    UnknownTimezoneError,
)
from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.schedule_track import _parse_query_ts, _require_str
from app.domain.timeline.store import EventStore, _utcnow, new_event_id


TRACK_NAME = "sleep"
TITLE = "sleep"
STATUS = "occurred"


def _resolve_sleep_local(name: str, local_iso: Any, tz_name: str) -> datetime:
    """Convert a naive local ISO-8601 timestamp + IANA tz into UTC.

    Sleep boundaries must be naive local wall-clock times: any zone
    designator (a trailing ``Z``/``z`` or an explicit offset) is rejected
    as ``InvalidInputError`` because the IANA zone travels in the
    event-level ``timezone`` field.

    Raises ``UnknownTimezoneError`` when the timezone is not a valid IANA
    name, or when the local time falls in a DST gap (does not exist in
    that zone). Ambiguous local times (DST fall-back) are resolved by
    choosing the earlier UTC instant (fold=0), mirroring the schedule
    track's resolution rule.
    """
    if not isinstance(local_iso, str) or not local_iso:
        raise InvalidInputError(
            f"{name} must be a non-empty ISO-8601 local timestamp"
        )
    if local_iso.endswith("Z") or local_iso.endswith("z"):
        raise InvalidInputError(
            f"{name} must be a naive local timestamp without a zone "
            "designator; the IANA zone goes in the timezone field"
        )
    try:
        naive = datetime.fromisoformat(local_iso)
    except ValueError as e:
        raise InvalidInputError(
            f"{name} is not a valid ISO-8601 local timestamp: {e}"
        ) from e
    if naive.tzinfo is not None:
        raise InvalidInputError(
            f"{name} must be a naive local timestamp without a zone "
            "designator; the IANA zone goes in the timezone field"
        )
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as e:
        raise UnknownTimezoneError(f"unknown timezone: {tz_name!r}") from e

    # DST handling (same fold-0/1 rule as the schedule track):
    #   - unambiguous: both folds give the same UTC
    #   - ambiguous (fall-back): folds differ -> pick the earlier (fold=0)
    #   - gap (spring-forward): folds differ AND round-trip does not
    #     preserve the wall time -> reject as UnknownTimezoneError
    c0_utc = naive.replace(tzinfo=tz, fold=0).astimezone(timezone.utc)
    c1_utc = naive.replace(tzinfo=tz, fold=1).astimezone(timezone.utc)
    if c0_utc != c1_utc:
        back = c0_utc.astimezone(tz).replace(tzinfo=None)
        if back != naive:
            raise UnknownTimezoneError(
                f"local time {local_iso} does not exist in {tz_name} "
                "(DST spring-forward gap)"
            )
        return c0_utc
    return c0_utc


def _resolve_pair(
    sleep_start: Any, sleep_end: Any, timezone: str
) -> tuple[datetime, datetime]:
    """Resolve both boundaries to UTC and enforce strict ordering."""
    start_utc = _resolve_sleep_local("sleep_start", sleep_start, timezone)
    end_utc = _resolve_sleep_local("sleep_end", sleep_end, timezone)
    if end_utc <= start_utc:
        raise InvalidInputError(
            "sleep_end must be strictly after sleep_start "
            "(zero-length or inverted interval)"
        )
    return start_utc, end_utc


class SleepTrack:
    """The sleep track — ingest/correct/retract/query/get."""

    name = TRACK_NAME

    def __init__(self, store: EventStore):
        self.store = store

    # ---------- ingest ----------

    def ingest(
        self,
        user_id: str,
        dedupe_key: str,
        sleep_start: str,
        sleep_end: str,
        timezone: str,
        source: str,
    ) -> ScheduleEvent:
        user_id = _require_str("user_id", user_id)
        dedupe_key = _require_str("dedupe_key", dedupe_key)
        timezone = _require_str("timezone", timezone)
        source = _require_str("source", source)

        # Idempotency: if (track, user_id, dedupe_key) exists, return it
        # verbatim — even if the new request body differs.
        existing = self.store.get_by_dedupe_key(user_id, self.name, dedupe_key)
        if existing is not None:
            return existing

        start_utc, end_utc = _resolve_pair(sleep_start, sleep_end, timezone)
        now = _utcnow()
        event = ScheduleEvent(
            event_id=new_event_id(),
            track=self.name,
            user_id=user_id,
            dedupe_key=dedupe_key,
            occurred_at_utc=start_utc,
            local_time=sleep_start,
            timezone=timezone,
            source=source,
            status=STATUS,
            title=TITLE,
            payload={"sleep_start": sleep_start, "sleep_end": sleep_end},
            supersedes=None,
            superseded_by=None,
            retracted=False,
            retracted_at=None,
            retracted_reason=None,
            created_at=now,
            updated_at=now,
        )
        self.store.insert(event)
        return event

    # ---------- correct ----------

    def correct(
        self,
        event_id: str,
        sleep_start: str,
        sleep_end: str,
        timezone: str,
    ) -> ScheduleEvent:
        event_id = _require_str("event_id", event_id)
        timezone = _require_str("timezone", timezone)

        target = self.store.get_by_id(event_id)
        if target is None:
            raise EventNotFoundError(f"event {event_id} does not exist")

        start_utc, end_utc = _resolve_pair(sleep_start, sleep_end, timezone)
        payload = {"sleep_start": sleep_start, "sleep_end": sleep_end}

        # Content-based idempotency FIRST: if an identical successor for
        # this target already exists, return it unchanged (same ordering
        # as the schedule track's Lead ruling).
        existing_correction = self.store.get_existing_correction(
            target_event_id=event_id,
            occurred_at_utc=start_utc,
            status=STATUS,
            title=TITLE,
            payload=payload,
        )
        if existing_correction is not None:
            return existing_correction

        if target.superseded_by is not None:
            raise EventAlreadySupersededError(
                f"event {event_id} has a newer version; use "
                f"{target.superseded_by} as the next event_id",
                superseded_by=target.superseded_by,
            )
        if target.retracted:
            raise EventRetractedError(
                f"event {event_id} is retracted; corrections to retracted "
                "events are not allowed"
            )

        now = _utcnow()
        new_id = new_event_id()
        new_dedupe_key = "auto:" + new_id
        new_event = ScheduleEvent(
            event_id=new_id,
            track=target.track,
            user_id=target.user_id,
            dedupe_key=new_dedupe_key,
            occurred_at_utc=start_utc,
            local_time=sleep_start,
            timezone=timezone,
            # Source is inherited from the target (sleep correct requests
            # do not carry a source field).
            source=target.source,
            status=STATUS,
            title=TITLE,
            payload=payload,
            supersedes=event_id,
            superseded_by=None,
            retracted=False,
            retracted_at=None,
            retracted_reason=None,
            created_at=now,
            updated_at=now,
        )
        self.store.insert(new_event)
        self.store.update_superseded_by(event_id, new_id)
        return new_event

    # ---------- retract ----------

    def retract(
        self, event_id: str, reason: Optional[str] = None
    ) -> ScheduleEvent:
        event_id = _require_str("event_id", event_id)
        if reason is not None and not isinstance(reason, str):
            raise InvalidInputError("reason must be a string when provided")

        target = self.store.get_by_id(event_id)
        if target is None:
            raise EventNotFoundError(f"event {event_id} does not exist")

        if target.retracted:
            # Idempotent: do not update retracted_at or retracted_reason.
            return target

        now = _utcnow()
        self.store.set_retracted(event_id, now, reason)
        return self.store.get_by_id(event_id)  # type: ignore[return-value]

    # ---------- query ----------

    def query(
        self,
        user_id: str,
        window_start: str,
        window_end: str,
        include_history: bool = False,
    ) -> List[ScheduleEvent]:
        user_id = _require_str("user_id", user_id)
        window_start = _require_str("window_start", window_start)
        window_end = _require_str("window_end", window_end)
        if not isinstance(include_history, bool):
            raise InvalidInputError("include_history must be a boolean")

        start_dt = _parse_query_ts("window_start", window_start)
        end_dt = _parse_query_ts("window_end", window_end)
        if end_dt <= start_dt:
            raise InvalidInputError("window_end must be strictly after window_start")

        return self.store.list_for_user_window(
            user_id=user_id,
            track=self.name,
            window_start_utc=start_dt,
            window_end_utc=end_dt,
            include_history=include_history,
        )

    # ---------- get ----------

    def get(self, event_id: str, resolve_head: bool = False) -> ScheduleEvent:
        event_id = _require_str("event_id", event_id)
        if not isinstance(resolve_head, bool):
            raise InvalidInputError("resolve_head must be a boolean")

        current = self.store.get_by_id(event_id)
        if current is None:
            raise EventNotFoundError(f"event {event_id} does not exist")

        if not resolve_head:
            return current

        # Walk superseded_by forward until reaching the head (or a
        # retracted event). Cycle-protected like the schedule track.
        seen: set[str] = set()
        node = current
        while node.superseded_by is not None:
            if node.superseded_by in seen:
                break
            seen.add(node.event_id)
            nxt = self.store.get_by_id(node.superseded_by)
            if nxt is None:
                break
            node = nxt
            if node.retracted:
                return node
        return node

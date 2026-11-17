"""Schedule track — the first registered track of the timeline kernel.

All behavior here matches the frozen contract data flows for
``schedule_event_ingest``, ``schedule_event_correct``,
``schedule_event_retract``, ``schedule_events_query``, and
``schedule_event_get``.
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
from app.domain.timeline.models import (
    VALID_STATUSES,
    ScheduleEvent,
)
from app.domain.timeline import models as _models
from app.domain.timeline.store import EventStore, _iso_z, _utcnow, new_event_id


TRACK_NAME = "schedule"


def _require_str(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise InvalidInputError(f"{name} must be a string")
    if not allow_empty and value == "":
        raise InvalidInputError(f"{name} must be a non-empty string")
    return value


def _require_object(name: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise InvalidInputError(f"{name} must be a JSON object")


def _resolve_occurred_at(local_iso: str, tz_name: str) -> datetime:
    """Convert a local ISO-8601 timestamp + IANA tz into a UTC datetime.

    Raises UnknownTimezoneError when the timezone is not a valid IANA
    name, or when the local time falls in a DST gap (does not exist in
    that zone). Ambiguous local times (DST fall-back) are resolved by
    choosing the earlier UTC instant.
    """
    if not isinstance(local_iso, str) or not local_iso:
        raise InvalidInputError("occurred_at must be a non-empty ISO-8601 string")
    # Strip any trailing 'Z' (treat as local) just in case the caller is sloppy.
    if local_iso.endswith("Z"):
        local_iso = local_iso[:-1]
    try:
        naive = datetime.fromisoformat(local_iso)
    except ValueError as e:
        raise InvalidInputError(
            f"occurred_at is not a valid ISO-8601 local timestamp: {e}"
        ) from e
    if naive.tzinfo is not None:
        # Caller passed an offset or zone — reject so we don't silently
        # re-interpret the timezone. They should pass local + IANA name.
        raise InvalidInputError(
            "occurred_at must be a naive local timestamp; the zone "
            "goes in the timezone field"
        )
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as e:
        raise UnknownTimezoneError(f"unknown timezone: {tz_name!r}") from e

    # DST handling. Compute both folds:
    #   - unambiguous: fold=0 and fold=1 give the same UTC
    #   - ambiguous (fall-back): folds differ -> pick the earlier (fold=0)
    #   - gap (spring-forward): folds differ AND round-trip does not
    #     preserve the wall time -> reject as UnknownTimezoneError
    c0_utc = naive.replace(tzinfo=tz, fold=0).astimezone(timezone.utc)
    c1_utc = naive.replace(tzinfo=tz, fold=1).astimezone(timezone.utc)
    if c0_utc != c1_utc:
        # Ambiguous or gap. Round-trip to disambiguate.
        back = c0_utc.astimezone(tz).replace(tzinfo=None)
        if back != naive:
            raise UnknownTimezoneError(
                f"local time {local_iso} does not exist in {tz_name} "
                "(DST spring-forward gap)"
            )
        # Ambiguous fall-back: pick the earlier UTC instant (fold=0).
        return c0_utc
    # fold=0 == fold=1: unambiguous.
    return c0_utc


class ScheduleTrack:
    """The schedule track — ingest/correct/retract/query/get."""

    name = TRACK_NAME

    def __init__(self, store: EventStore):
        self.store = store

    # ---------- ingest ----------

    def ingest(
        self,
        user_id: str,
        dedupe_key: str,
        occurred_at: str,
        timezone: str,
        source: str,
        status: str,
        title: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> ScheduleEvent:
        user_id = _require_str("user_id", user_id)
        dedupe_key = _require_str("dedupe_key", dedupe_key)
        source = _require_str("source", source)
        title = _require_str("title", title)
        timezone = _require_str("timezone", timezone)
        if status not in VALID_STATUSES:
            raise InvalidInputError(
                f"status must be one of {sorted(VALID_STATUSES)}"
            )
        payload = payload or {}
        payload = _require_object("payload", payload)

        # Idempotency: if (track, user_id, dedupe_key) exists, return it.
        existing = self.store.get_by_dedupe_key(user_id, self.name, dedupe_key)
        if existing is not None:
            return existing

        occurred_at_utc = _resolve_occurred_at(occurred_at, timezone)
        now = _utcnow()
        # Auto-generated dedupe_key is stable; for ingest we just echo
        # the client-supplied one. For corrections the track applies its
        # own derivation (see correct()).
        event = ScheduleEvent(
            event_id=new_event_id(),
            track=self.name,
            user_id=user_id,
            dedupe_key=dedupe_key,
            occurred_at_utc=occurred_at_utc,
            local_time=occurred_at,
            timezone=timezone,
            source=source,
            status=status,
            realization='actual' if status == 'occurred' else 'planned',
            title=title,
            payload=payload,
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
        occurred_at: str,
        timezone: str,
        status: str,
        title: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> ScheduleEvent:
        event_id = _require_str("event_id", event_id)
        title = _require_str("title", title)
        timezone = _require_str("timezone", timezone)
        if status not in VALID_STATUSES:
            raise InvalidInputError(
                f"status must be one of {sorted(VALID_STATUSES)}"
            )
        payload = payload or {}
        payload = _require_object("payload", payload)

        target = self.store.get_by_id(event_id)
        if target is None or target.track != self.name or not self.store.is_visible_event(target):
            raise EventNotFoundError(f"event {event_id} does not exist")

        # Per Lead ruling: content-based idempotency is evaluated BEFORE
        # superseded/retracted state checks. If an identical successor
        # for this target already exists, return it unchanged.
        # Need occurred_at_utc to compare; resolve now.
        occurred_at_utc = _resolve_occurred_at(occurred_at, timezone)
        existing_correction = self.store.get_existing_correction(
            target_event_id=event_id,
            occurred_at_utc=occurred_at_utc,
            status=status,
            title=title,
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
        # Auto-generated dedupe_key derived deterministically from the
        # new event_id so the value is reproducible.
        new_id = new_event_id()
        new_dedupe_key = "auto:" + new_id
        new_event = ScheduleEvent(
            event_id=new_id,
            track=target.track,
            user_id=target.user_id,
            dedupe_key=new_dedupe_key,
            occurred_at_utc=occurred_at_utc,
            local_time=occurred_at,
            timezone=timezone,
            # Per Lead ruling: source is inherited from the target, not
            # from the client input.
            source=target.source,
            status=status,
            confirmation=target.confirmation,
            realization='actual' if status == 'occurred' else 'planned',
            title=title,
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
        # Link the target to its successor.
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
        if target is None or target.track != self.name or not self.store.is_visible_event(target):
            raise EventNotFoundError(f"event {event_id} does not exist")

        if target.retracted:
            # Idempotent: do not update retracted_at or retracted_reason.
            return target

        now = _utcnow()
        self.store.set_retracted(event_id, now, reason)
        # Re-read so the caller gets the updated record.
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
        if current is None or current.track != self.name or not self.store.is_visible_event(current):
            raise EventNotFoundError(f"event {event_id} does not exist")

        if not resolve_head:
            return current

        # Walk superseded_by forward until we reach the head.
        # The chain walk is bounded; in practice chains are short.
        seen: set[str] = set()
        node = current
        while node.superseded_by is not None:
            if node.superseded_by in seen:
                # Cycle protection (should not happen in well-formed data).
                break
            seen.add(node.event_id)
            nxt = self.store.get_by_id(node.superseded_by)
            if nxt is None:
                break
            node = nxt
            if node.retracted:
                # Return the retracted event so the caller can see the
                # terminal state.
                return node
        return node


def _parse_query_ts(name: str, value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise InvalidInputError(f"{name} must be a non-empty ISO-8601 string")
    s = value
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise InvalidInputError(
            f"{name} is not a valid ISO-8601 timestamp: {e}"
        ) from e
    if dt.tzinfo is None:
        # Default to UTC for query bounds; document this in the error.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

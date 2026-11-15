"""Timeline kernel — track-pluggable event store coordinator.

The kernel owns the storage and the registered tracks. Each track
declares its ingest/correct/retract/query/get shape. The kernel registers
both ``ScheduleTrack`` and ``SleepTrack``.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from app.domain.timeline.errors import EventAlreadySupersededError, EventNotFoundError, EventRetractedError, InvalidInputError, UnknownTimezoneError
from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.schedule_track import ScheduleTrack
from app.domain.timeline.sleep_track import SleepTrack
from app.domain.timeline.store import EventStore

class TimelineKernel:
    """Persistence-backed, track-pluggable personal timeline kernel."""

    def __init__(self, store: EventStore):
        self.store = store
        self._tracks: Dict[str, Any] = {}
        self.register(ScheduleTrack(store))
        self.register(SleepTrack(store))

    def register(self, track: Any) -> None:
        if not hasattr(track, 'name'):
            raise InvalidInputError("track must have a 'name' attribute")
        if track.name in self._tracks:
            raise InvalidInputError(f'track {track.name!r} already registered')
        self._tracks[track.name] = track

    def get_track(self, name: str) -> Any:
        if name not in self._tracks:
            raise InvalidInputError(f'track {name!r} is not registered')
        return self._tracks[name]

    @property
    def registered_tracks(self) -> List[str]:
        return list(self._tracks.keys())

    def ingest_schedule_event(self, user_id: str, dedupe_key: str, occurred_at: str, timezone: str, source: str, status: str, title: str, payload: Optional[Dict[str, Any]]=None) -> ScheduleEvent:
        return self.get_track('schedule').ingest(user_id=user_id, dedupe_key=dedupe_key, occurred_at=occurred_at, timezone=timezone, source=source, status=status, title=title, payload=payload)

    def correct_schedule_event(self, event_id: str, occurred_at: str, timezone: str, status: str, title: str, payload: Optional[Dict[str, Any]]=None) -> ScheduleEvent:
        return self.get_track('schedule').correct(event_id=event_id, occurred_at=occurred_at, timezone=timezone, status=status, title=title, payload=payload)

    def retract_schedule_event(self, event_id: str, reason: Optional[str]=None) -> ScheduleEvent:
        return self.get_track('schedule').retract(event_id=event_id, reason=reason)

    def list_schedule_events(self, user_id: str, window_start: str, window_end: str, include_history: bool=False) -> List[ScheduleEvent]:
        return self.get_track('schedule').query(user_id=user_id, window_start=window_start, window_end=window_end, include_history=include_history)

    def get_schedule_event(self, event_id: str, resolve_head: bool=False) -> ScheduleEvent:
        return self.get_track('schedule').get(event_id=event_id, resolve_head=resolve_head)

    def ingest_sleep_event(self, user_id: str, dedupe_key: str, sleep_start: str, sleep_end: str, timezone: str, source: str) -> ScheduleEvent:
        return self.get_track('sleep').ingest(user_id=user_id, dedupe_key=dedupe_key, sleep_start=sleep_start, sleep_end=sleep_end, timezone=timezone, source=source)

    def correct_sleep_event(self, event_id: str, sleep_start: str, sleep_end: str, timezone: str) -> ScheduleEvent:
        return self.get_track('sleep').correct(event_id=event_id, sleep_start=sleep_start, sleep_end=sleep_end, timezone=timezone)

    def retract_sleep_event(self, event_id: str, reason: Optional[str]=None) -> ScheduleEvent:
        return self.get_track('sleep').retract(event_id=event_id, reason=reason)

    def list_sleep_events(self, user_id: str, window_start: str, window_end: str, include_history: bool=False) -> List[ScheduleEvent]:
        return self.get_track('sleep').query(user_id=user_id, window_start=window_start, window_end=window_end, include_history=include_history)

    def get_sleep_event(self, event_id: str, resolve_head: bool=False) -> ScheduleEvent:
        return self.get_track('sleep').get(event_id=event_id, resolve_head=resolve_head)

    def close(self) -> None:
        self.store.close()

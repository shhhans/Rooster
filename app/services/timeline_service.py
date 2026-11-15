"""Service layer exposing the 5 frozen public interfaces of the timeline
kernel.

This module is the public Python API surface. The FastAPI layer in
``app/api/timeline_routes.py`` delegates to this module; CLI or batch
importers can do the same.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.domain.timeline import TimelineKernel
from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.store import EventStore
DEFAULT_DB_ENV = 'ROOSTER_TIMELINE_DB'

def default_db_path() -> str:
    env = os.environ.get(DEFAULT_DB_ENV)
    if env:
        return env
    p = Path.cwd() / '.timeline_kens' / 'kernel.db'
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

class TimelineService:
    """Thin service wrapper over the kernel for the 5 public interfaces."""

    def __init__(self, db_path: Optional[str]=None):
        path = db_path or default_db_path()
        self._store = EventStore(path)
        self._kernel = TimelineKernel(self._store)

    @property
    def store(self) -> EventStore:
        """The underlying event store (shared connection; used by the
        incremental scheduling stores to join one transaction)."""
        return self._store

    @property
    def kernel(self) -> TimelineKernel:
        """The timeline kernel (track-registered event coordinator)."""
        return self._kernel

    def ingest_schedule_event(self, user_id: str, dedupe_key: str, occurred_at: str, timezone: str, source: str, status: str, title: str, payload: Optional[Dict[str, Any]]=None) -> ScheduleEvent:
        return self._kernel.ingest_schedule_event(user_id=user_id, dedupe_key=dedupe_key, occurred_at=occurred_at, timezone=timezone, source=source, status=status, title=title, payload=payload)

    def correct_schedule_event(self, event_id: str, occurred_at: str, timezone: str, status: str, title: str, payload: Optional[Dict[str, Any]]=None) -> ScheduleEvent:
        return self._kernel.correct_schedule_event(event_id=event_id, occurred_at=occurred_at, timezone=timezone, status=status, title=title, payload=payload)

    def retract_schedule_event(self, event_id: str, reason: Optional[str]=None) -> ScheduleEvent:
        return self._kernel.retract_schedule_event(event_id=event_id, reason=reason)

    def list_schedule_events(self, user_id: str, window_start: str, window_end: str, include_history: bool=False) -> List[ScheduleEvent]:
        return self._kernel.list_schedule_events(user_id=user_id, window_start=window_start, window_end=window_end, include_history=include_history)

    def get_schedule_event(self, event_id: str, resolve_head: bool=False) -> ScheduleEvent:
        return self._kernel.get_schedule_event(event_id=event_id, resolve_head=resolve_head)

    def ingest_sleep_event(self, user_id: str, dedupe_key: str, sleep_start: str, sleep_end: str, timezone: str, source: str) -> ScheduleEvent:
        return self._kernel.ingest_sleep_event(user_id=user_id, dedupe_key=dedupe_key, sleep_start=sleep_start, sleep_end=sleep_end, timezone=timezone, source=source)

    def correct_sleep_event(self, event_id: str, sleep_start: str, sleep_end: str, timezone: str) -> ScheduleEvent:
        return self._kernel.correct_sleep_event(event_id=event_id, sleep_start=sleep_start, sleep_end=sleep_end, timezone=timezone)

    def retract_sleep_event(self, event_id: str, reason: Optional[str]=None) -> ScheduleEvent:
        return self._kernel.retract_sleep_event(event_id=event_id, reason=reason)

    def list_sleep_events(self, user_id: str, window_start: str, window_end: str, include_history: bool=False) -> List[ScheduleEvent]:
        return self._kernel.list_sleep_events(user_id=user_id, window_start=window_start, window_end=window_end, include_history=include_history)

    def get_sleep_event(self, event_id: str, resolve_head: bool=False) -> ScheduleEvent:
        return self._kernel.get_sleep_event(event_id=event_id, resolve_head=resolve_head)

    def close(self) -> None:
        self._kernel.close()
_service: Optional[TimelineService] = None

def get_default_service() -> TimelineService:
    global _service
    if _service is None:
        _service = TimelineService()
    return _service

def reset_default_service() -> None:
    global _service
    if _service is not None:
        _service.close()
    _service = None

"""Public Python invocation surface for the timeline kernel + schedule track.

This module is the canonical entry point for black-box validation. It
exposes the five frozen public interfaces as module-level callables
returning JSON-compatible ``dict`` / ``list`` results (matching the
contract output shape).

Configuration
-------------
- ``ROOSTER_TIMELINE_DB``: path to the SQLite database file. When set,
  every call uses an isolated database pointed at this path. When
  unset, the default ``<cwd>/.timeline_kens/kernel.db`` is used.

Reset hook
----------
- ``reset_isolated_db()``: closes the current service, deletes the
  database file (and SQLite ``-wal`` / ``-shm`` / ``-journal``
  companions), and reopens a fresh service. Designed for pytest
  fixtures and any other black-box validator that wants a clean slate.

Lifecycle
---------
- The service is lazily initialized on first call. The ``close()``
  helper releases the underlying SQLite connection.
- ``reset_isolated_db()`` resets state to a brand-new database at the
  current ``ROOSTER_TIMELINE_DB`` path.

The implementation here is a thin wrapper over
``app.services.timeline_service.TimelineService``; every callable
delegates to the same domain code that the FastAPI layer uses.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from app.domain.timeline.models import ScheduleEvent
from app.services.timeline_service import TimelineService, default_db_path
_service: Optional[TimelineService] = None

def _get_service() -> TimelineService:
    """Return the module-level service, initializing on first call."""
    global _service
    if _service is None:
        _service = TimelineService()
    return _service

def reset_isolated_db() -> None:
    """Reset the isolated database.

    Closes any open service, deletes the SQLite file (and its
    ``-wal`` / ``-shm`` / ``-journal`` companions) at the path given
    by ``ROOSTER_TIMELINE_DB`` (or the default location), and
    initializes a fresh service. Safe to call multiple times; safe to
    call before any other call has been made.

    Returns ``None``. Intended for use in pytest fixtures, black-box
    validators, and CLI bootstrap scripts.
    """
    global _service
    if _service is not None:
        try:
            _service.close()
        except Exception:
            pass
        _service = None
    db_path = os.environ.get('ROOSTER_TIMELINE_DB') or default_db_path()
    import time
    for suffix in ('', '-wal', '-shm', '-journal'):
        p = db_path + suffix if suffix else db_path
        if os.path.exists(p):
            for _ in range(3):
                try:
                    os.remove(p)
                    break
                except OSError:
                    time.sleep(0.05)
    _service = TimelineService(db_path=db_path)

def close() -> None:
    """Release the underlying SQLite connection. Idempotent."""
    global _service
    if _service is not None:
        try:
            _service.close()
        finally:
            _service = None

def ingest_schedule_event(user_id: str, dedupe_key: str, occurred_at: str, timezone: str, source: str, status: str, title: str, payload: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Idempotent schedule event ingest.

    See ``data_flow.schedule_event_ingest`` in the frozen contract.
    Returns the unified event_shape as a JSON-compatible ``dict``.
    """
    e: ScheduleEvent = _get_service().ingest_schedule_event(user_id=user_id, dedupe_key=dedupe_key, occurred_at=occurred_at, timezone=timezone, source=source, status=status, title=title, payload=payload)
    return e.to_dict()

def correct_schedule_event(event_id: str, occurred_at: str, timezone: str, status: str, title: str, payload: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Submit a correction that supersedes a stored event.

    See ``data_flow.schedule_event_correct`` in the frozen contract.
    Returns the unified event_shape as a JSON-compatible ``dict``.
    Raises ``EventAlreadySupersededError``, ``EventRetractedError``,
    ``EventNotFoundError``, ``InvalidInputError``, or
    ``UnknownTimezoneError`` per the contract.
    """
    e: ScheduleEvent = _get_service().correct_schedule_event(event_id=event_id, occurred_at=occurred_at, timezone=timezone, status=status, title=title, payload=payload)
    return e.to_dict()

def retract_schedule_event(event_id: str, reason: Optional[str]=None) -> Dict[str, Any]:
    """Retract an event without producing a successor.

    See ``data_flow.schedule_event_retract`` in the frozen contract.
    Returns the unified event_shape as a JSON-compatible ``dict``.
    Raises ``EventNotFoundError`` or ``InvalidInputError`` per the
    contract.
    """
    e: ScheduleEvent = _get_service().retract_schedule_event(event_id=event_id, reason=reason)
    return e.to_dict()

def list_schedule_events(user_id: str, window_start: str, window_end: str, include_history: bool=False) -> List[Dict[str, Any]]:
    """Return one user's schedule events chronologically.

    See ``data_flow.schedule_events_query`` in the frozen contract.
    Returns a list of unified event_shape records as JSON-compatible
    dicts, sorted by ``occurred_at_utc`` ascending (tiebreak
    ``created_at`` ascending, then ``event_id`` ascending).
    Raises ``InvalidInputError`` per the contract.
    """
    events = _get_service().list_schedule_events(user_id=user_id, window_start=window_start, window_end=window_end, include_history=include_history)
    return [e.to_dict() for e in events]

def get_schedule_event(event_id: str, resolve_head: bool=False) -> Dict[str, Any]:
    """Fetch a single event by id, optionally resolving the chain head.

    See ``data_flow.schedule_event_get`` in the frozen contract.
    Returns the unified event_shape as a JSON-compatible ``dict``.
    Raises ``EventNotFoundError`` or ``InvalidInputError`` per the
    contract.
    """
    e: ScheduleEvent = _get_service().get_schedule_event(event_id=event_id, resolve_head=resolve_head)
    return e.to_dict()

def ingest_sleep_event(user_id: str, dedupe_key: str, sleep_start: str, sleep_end: str, timezone: str, source: str) -> Dict[str, Any]:
    """Idempotent sleep-event ingest.

    See ``data_flow.sleep_event_ingest`` in the frozen contract.
    Returns the unified event shape as a JSON-compatible ``dict``.
    Raises ``InvalidInputError`` or ``UnknownTimezoneError``.
    """
    e: ScheduleEvent = _get_service().ingest_sleep_event(user_id=user_id, dedupe_key=dedupe_key, sleep_start=sleep_start, sleep_end=sleep_end, timezone=timezone, source=source)
    return e.to_dict()

def correct_sleep_event(event_id: str, sleep_start: str, sleep_end: str, timezone: str) -> Dict[str, Any]:
    """Submit a correction that supersedes a stored sleep event.

    See ``data_flow.sleep_event_correct`` in the frozen contract.
    Returns the new event as a JSON-compatible ``dict``. Raises
    ``InvalidInputError``, ``UnknownTimezoneError``,
    ``EventNotFoundError``, ``EventAlreadySupersededError``, or
    ``EventRetractedError`` per the contract.
    """
    e: ScheduleEvent = _get_service().correct_sleep_event(event_id=event_id, sleep_start=sleep_start, sleep_end=sleep_end, timezone=timezone)
    return e.to_dict()

def retract_sleep_event(event_id: str, reason: Optional[str]=None) -> Dict[str, Any]:
    """Retract a stored sleep event.

    See ``data_flow.sleep_event_retract`` in the frozen contract.
    Returns the updated event as a JSON-compatible ``dict``. Raises
    ``InvalidInputError`` or ``EventNotFoundError`` per the contract.
    """
    e: ScheduleEvent = _get_service().retract_sleep_event(event_id=event_id, reason=reason)
    return e.to_dict()

def list_sleep_events(user_id: str, window_start: str, window_end: str, include_history: bool=False) -> List[Dict[str, Any]]:
    """Return one user's sleep events chronologically.

    See ``data_flow.sleep_events_query`` in the frozen contract.
    Returns a list of unified event_shape records as JSON-compatible
    dicts, sorted by ``occurred_at_utc`` ascending (tiebreak
    ``created_at`` ascending, then ``event_id`` ascending).
    Raises ``InvalidInputError`` per the contract.
    """
    events = _get_service().list_sleep_events(user_id=user_id, window_start=window_start, window_end=window_end, include_history=include_history)
    return [e.to_dict() for e in events]

def get_sleep_event(event_id: str, resolve_head: bool=False) -> Dict[str, Any]:
    """Fetch a single sleep event by id, optionally resolving the chain head.

    See ``data_flow.sleep_event_get`` in the frozen contract.
    Returns the unified event_shape as a JSON-compatible ``dict``.
    Raises ``InvalidInputError`` or ``EventNotFoundError`` per the
    contract.
    """
    e: ScheduleEvent = _get_service().get_sleep_event(event_id=event_id, resolve_head=resolve_head)
    return e.to_dict()

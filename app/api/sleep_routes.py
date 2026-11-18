"""FastAPI router for the sleep track (SLEEP-001).

The router delegates to ``app.services.timeline_service.TimelineService``
which in turn uses the domain layer. Error envelopes mirror the schedule
track exactly:

* ``InvalidInputError`` -> 400 ``INVALID_INPUT``
* ``UnknownTimezoneError`` -> 400 ``UNKNOWN_TIMEZONE``
* ``EventNotFoundError`` -> 404 ``EVENT_NOT_FOUND``
* ``EventAlreadySupersededError`` -> 409 ``EVENT_ALREADY_SUPERSEDED``
  (with ``superseded_by`` in the body)
* ``EventRetractedError`` -> 409 ``EVENT_RETRACTED``

Implementation note: errors are returned as ``JSONResponse`` directly
rather than raised as ``HTTPException``. ``app.main`` registers an
app-wide ``HTTPException`` handler (owned by the schedule-state module)
that rewrites any detail dict without ``error_code`` into the
schedule-state envelope; returning ``JSONResponse`` bypasses that
rewrite and guarantees the contract's ``{error, message}`` envelope
reaches the client verbatim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.domain.timeline.errors import (
    EventAlreadySupersededError,
    EventNotFoundError,
    EventRetractedError,
    InvalidInputError,
    UnknownTimezoneError,
)
from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.sleep_track import _resolve_sleep_local
from app.services.timeline_service import TimelineService
from app.services.reactive_trigger import queue_input_review


router = APIRouter(prefix="/timeline/sleep", tags=["timeline-sleep"])


def _get_service(request: Request) -> TimelineService:
    """Return the app-lifespan timeline service, or lazily create one.

    When the app runs under uvicorn (or a ``with TestClient(app)`` block),
    the lifespan sets ``app.state.timeline_service`` and we use it. For a
    bare ``TestClient(app)`` without lifespan, we create a default service
    once and cache it on app.state so subsequent requests reuse it (same
    pattern as ``app.api.schedule_state._resolve_timeline_service``).
    """
    svc = getattr(request.app.state, "timeline_service", None)
    if svc is None:
        svc = TimelineService()
        request.app.state.timeline_service = svc
    return svc


# ---------- request models ----------
# Fields are Optional[Any] so that missing / null / wrong-typed inputs
# flow through to the track layer, which raises InvalidInputError ->
# 400 INVALID_INPUT (the contract's error envelope), instead of a
# FastAPI 422 validation error.


class SleepIngestRequest(BaseModel):
    user_id: Optional[Any] = None
    dedupe_key: Optional[Any] = None
    sleep_start: Optional[Any] = None
    sleep_end: Optional[Any] = None
    timezone: Optional[Any] = None
    source: Optional[Any] = None


class SleepCorrectRequest(BaseModel):
    sleep_start: Optional[Any] = None
    sleep_end: Optional[Any] = None
    timezone: Optional[Any] = None


class SleepRetractRequest(BaseModel):
    reason: Optional[Any] = None



# ---------- response shape ----------


def _event_dict(e: ScheduleEvent) -> Dict[str, Any]:
    return e.to_dict()


def _error_response(e: Exception) -> JSONResponse:
    """Return the contract's HTTP error envelope as a JSONResponse."""
    if isinstance(e, EventNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "EVENT_NOT_FOUND", "message": str(e)},
        )
    if isinstance(e, EventAlreadySupersededError):
        return JSONResponse(
            status_code=409,
            content={
                "error": "EVENT_ALREADY_SUPERSEDED",
                "message": str(e),
                "superseded_by": e.superseded_by,
            },
        )
    if isinstance(e, EventRetractedError):
        return JSONResponse(
            status_code=409,
            content={"error": "EVENT_RETRACTED", "message": str(e)},
        )
    if isinstance(e, UnknownTimezoneError):
        return JSONResponse(
            status_code=400,
            content={"error": "UNKNOWN_TIMEZONE", "message": str(e)},
        )
    if isinstance(e, InvalidInputError):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT", "message": str(e)},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL", "message": str(e)},
    )


# ---------- endpoints ----------


@router.post("/events", response_model=Dict[str, Any])
def ingest_event(req: SleepIngestRequest, background_tasks: BackgroundTasks, svc: TimelineService = Depends(_get_service)):
    try:
        with svc.store.transaction():
            revision = svc.store.get_input_revision(req.user_id) if isinstance(req.user_id, str) else None
            e = svc.ingest_sleep_event(
                user_id=req.user_id, dedupe_key=req.dedupe_key, sleep_start=req.sleep_start,
                sleep_end=req.sleep_end, timezone=req.timezone, source='user')
            if revision != svc.store.get_input_revision(e.user_id):
                queue_input_review(svc.store, background_tasks, [e], 'actual_sleep_updated')
    except (
        InvalidInputError,
        UnknownTimezoneError,
        EventNotFoundError,
        EventAlreadySupersededError,
        EventRetractedError,
    ) as exc:
        return _error_response(exc)
    return _event_dict(e)


@router.post("/events/{event_id}/correct", response_model=Dict[str, Any])
def correct_event(
    event_id: str, req: SleepCorrectRequest, background_tasks: BackgroundTasks, svc: TimelineService = Depends(_get_service)
):
    try:
        with svc.store.transaction():
            old = svc.store.get_by_id(event_id)
            revision = svc.store.get_input_revision(old.user_id) if old else None
            e = svc.correct_sleep_event(event_id=event_id, sleep_start=req.sleep_start,
                                        sleep_end=req.sleep_end, timezone=req.timezone)
            if revision != svc.store.get_input_revision(e.user_id):
                queue_input_review(svc.store, background_tasks, [old, e], 'actual_sleep_updated')
    except (
        InvalidInputError,
        UnknownTimezoneError,
        EventNotFoundError,
        EventAlreadySupersededError,
        EventRetractedError,
    ) as exc:
        return _error_response(exc)
    return _event_dict(e)


@router.post("/events/{event_id}/retract", response_model=Dict[str, Any])
def retract_event(
    event_id: str, req: SleepRetractRequest, background_tasks: BackgroundTasks, svc: TimelineService = Depends(_get_service)
):
    try:
        with svc.store.transaction():
            old = svc.store.get_by_id(event_id)
            revision = svc.store.get_input_revision(old.user_id) if old else None
            e = svc.retract_sleep_event(event_id=event_id, reason=req.reason)
            if revision != svc.store.get_input_revision(e.user_id):
                queue_input_review(svc.store, background_tasks, [old], 'actual_sleep_updated')
    except (InvalidInputError, EventNotFoundError) as exc:
        return _error_response(exc)
    return _event_dict(e)


@router.get("/events", response_model=List[Dict[str, Any]])
def list_events(
    user_id: Optional[str] = Query(None),
    window_start: Optional[str] = Query(None),
    window_end: Optional[str] = Query(None),
    include_history: bool = Query(False),
    include_end_utc: bool = Query(False),
    svc: TimelineService = Depends(_get_service),
):
    try:
        events = svc.list_sleep_events(
            user_id=user_id,
            window_start=window_start,
            window_end=window_end,
            include_history=include_history,
        )
    except InvalidInputError as exc:
        return _error_response(exc)
    result = [_event_dict(e) for e in events]
    if include_end_utc:
        for item, event in zip(result, events):
            item['sleep_end_utc'] = _resolve_sleep_local('sleep_end', event.payload['sleep_end'], event.timezone).isoformat().replace('+00:00', 'Z')
    return result


@router.get("/events/{event_id}", response_model=Dict[str, Any])
def get_event(
    event_id: str,
    resolve_head: bool = Query(False),
    svc: TimelineService = Depends(_get_service),
):
    try:
        e = svc.get_sleep_event(event_id=event_id, resolve_head=resolve_head)
    except (InvalidInputError, EventNotFoundError) as exc:
        return _error_response(exc)
    return _event_dict(e)

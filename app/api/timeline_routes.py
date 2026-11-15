"""FastAPI router for the timeline kernel + schedule track.

The router delegates to ``app.services.timeline_service.TimelineService``
which in turn uses the domain layer. Errors are mapped to HTTP responses:

* ``InvalidInputError`` -> 400
* ``EventNotFoundError`` -> 404
* ``EventAlreadySupersededError`` -> 409 (with superseded_by in the body)
* ``EventRetractedError`` -> 409
* ``UnknownTimezoneError`` -> 400
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.domain.timeline.errors import (
    EventAlreadySupersededError,
    EventNotFoundError,
    EventRetractedError,
    InvalidInputError,
    UnknownTimezoneError,
)
from app.domain.timeline.models import ScheduleEvent
from app.services.timeline_service import TimelineService


router = APIRouter(prefix="/timeline/schedule", tags=["timeline-schedule"])


def _get_service(request: Request) -> TimelineService:
    return request.app.state.timeline_service


# ---------- request models ----------


class IngestRequest(BaseModel):
    user_id: str
    dedupe_key: str
    occurred_at: str
    timezone: str
    source: str
    status: str
    title: str
    payload: Optional[Dict[str, Any]] = None


class CorrectRequest(BaseModel):
    occurred_at: str
    timezone: str
    status: str
    title: str
    payload: Optional[Dict[str, Any]] = None


class RetractRequest(BaseModel):
    reason: Optional[str] = None


# ---------- response shape ----------


def _event_dict(e: ScheduleEvent) -> Dict[str, Any]:
    return e.to_dict()


def _map_error(e: Exception) -> HTTPException:
    if isinstance(e, EventNotFoundError):
        return HTTPException(status_code=404, detail={"error": "EVENT_NOT_FOUND", "message": str(e)})
    if isinstance(e, EventAlreadySupersededError):
        return HTTPException(
            status_code=409,
            detail={
                "error": "EVENT_ALREADY_SUPERSEDED",
                "message": str(e),
                "superseded_by": e.superseded_by,
            },
        )
    if isinstance(e, EventRetractedError):
        return HTTPException(
            status_code=409, detail={"error": "EVENT_RETRACTED", "message": str(e)}
        )
    if isinstance(e, UnknownTimezoneError):
        return HTTPException(
            status_code=400, detail={"error": "UNKNOWN_TIMEZONE", "message": str(e)}
        )
    if isinstance(e, InvalidInputError):
        return HTTPException(
            status_code=400, detail={"error": "INVALID_INPUT", "message": str(e)}
        )
    return HTTPException(status_code=500, detail={"error": "INTERNAL", "message": str(e)})


# ---------- endpoints ----------


@router.post("/events", response_model=Dict[str, Any])
def ingest_event(req: IngestRequest, svc: TimelineService = Depends(_get_service)):
    try:
        e = svc.ingest_schedule_event(
            user_id=req.user_id,
            dedupe_key=req.dedupe_key,
            occurred_at=req.occurred_at,
            timezone=req.timezone,
            source=req.source,
            status=req.status,
            title=req.title,
            payload=req.payload,
        )
    except (
        InvalidInputError,
        UnknownTimezoneError,
        EventNotFoundError,
        EventAlreadySupersededError,
        EventRetractedError,
    ) as exc:
        raise _map_error(exc) from exc
    return _event_dict(e)


@router.post("/events/{event_id}/correct", response_model=Dict[str, Any])
def correct_event(
    event_id: str, req: CorrectRequest, svc: TimelineService = Depends(_get_service)
):
    try:
        e = svc.correct_schedule_event(
            event_id=event_id,
            occurred_at=req.occurred_at,
            timezone=req.timezone,
            status=req.status,
            title=req.title,
            payload=req.payload,
        )
    except (
        InvalidInputError,
        UnknownTimezoneError,
        EventNotFoundError,
        EventAlreadySupersededError,
        EventRetractedError,
    ) as exc:
        raise _map_error(exc) from exc
    return _event_dict(e)


@router.post("/events/{event_id}/retract", response_model=Dict[str, Any])
def retract_event(
    event_id: str, req: RetractRequest, svc: TimelineService = Depends(_get_service)
):
    try:
        e = svc.retract_schedule_event(event_id=event_id, reason=req.reason)
    except (
        InvalidInputError,
        EventNotFoundError,
    ) as exc:
        raise _map_error(exc) from exc
    return _event_dict(e)


@router.get("/events", response_model=List[Dict[str, Any]])
def list_events(
    user_id: str = Query(...),
    window_start: str = Query(...),
    window_end: str = Query(...),
    include_history: bool = Query(False),
    svc: TimelineService = Depends(_get_service),
):
    try:
        events = svc.list_schedule_events(
            user_id=user_id,
            window_start=window_start,
            window_end=window_end,
            include_history=include_history,
        )
    except InvalidInputError as exc:
        raise _map_error(exc) from exc
    return [_event_dict(e) for e in events]


@router.get("/events/{event_id}", response_model=Dict[str, Any])
def get_event(
    event_id: str,
    resolve_head: bool = Query(False),
    svc: TimelineService = Depends(_get_service),
):
    try:
        e = svc.get_schedule_event(event_id=event_id, resolve_head=resolve_head)
    except (InvalidInputError, EventNotFoundError) as exc:
        raise _map_error(exc) from exc
    return _event_dict(e)
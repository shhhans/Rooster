"""On-demand generation of a rolling baseline sleep window."""
import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.timeline.baseline import generate_baseline
from app.services.reactive_timeline import refresh_issues
import logging

router = APIRouter(prefix='/api/v1/baseline')


def _refresh_baseline_issues(store, user_id, start, end):
    try:
        refresh_issues(store, user_id, start - timedelta(days=1), end + timedelta(days=1))
    except Exception:
        logging.getLogger(__name__).exception('Failed to refresh baseline issues for %s', user_id)


class BaselineRequest(BaseModel):
    user_id: str = Field(min_length=1)
    start_date: date
    profile: dict[str, Any] | None = None


class MonthBaselineRequest(BaseModel):
    user_id: str = Field(min_length=1)
    month: str = Field(pattern=r'^\d{4}-(0[1-9]|1[0-2])$')
    profile: dict[str, Any] | None = None


@router.post('/roster-month')
def generate_roster_month(body: MonthBaselineRequest, request: Request, background_tasks: BackgroundTasks):
    store = request.app.state.timeline_service.store
    with store.lock:
        found = store.connection.execute('SELECT 1 FROM roster_confirmations WHERE user_id=? AND coverage_month=? LIMIT 1', (body.user_id, body.month)).fetchone()
    if not found:
        return {'status': 'no_roster', 'created': 0}
    year, month = map(int, body.month.split('-'))
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    try:
        result = generate_baseline(store, body.user_id, first, last, body.profile)
        if result['created'] or result.get('profile_changed'):
            zone = ZoneInfo(body.profile['home_timezone'])
            start = datetime.combine(first, time(), zone).astimezone(timezone.utc)
            end = datetime.combine(last + timedelta(days=1), time(), zone).astimezone(timezone.utc)
            background_tasks.add_task(_refresh_baseline_issues, store, body.user_id, start, end)
        return result
    except (ValueError, TypeError, ZoneInfoNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/generate')
def generate(body: BaselineRequest, request: Request, background_tasks: BackgroundTasks):
    try:
        store = request.app.state.timeline_service.store
        result = generate_baseline(store, body.user_id,
                                 body.start_date, body.start_date + timedelta(days=13), body.profile)
        if result['created'] or result.get('profile_changed'):
            zone = ZoneInfo(body.profile['home_timezone'])
            start = datetime.combine(body.start_date, time(), zone).astimezone(timezone.utc)
            end = datetime.combine(body.start_date + timedelta(days=14), time(), zone).astimezone(timezone.utc)
            background_tasks.add_task(_refresh_baseline_issues, store, body.user_id, start, end)
        return result
    except (ValueError, TypeError, ZoneInfoNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

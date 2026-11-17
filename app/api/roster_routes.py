"""Roster preview API — POST /api/v1/roster/preview.

Preview only: uploaded PDF/PNG/JPEG rosters are parsed into proposed
schedule events with provenance and unresolved items. Nothing is persisted,
no timeline storage is touched, and failed parses always return an explicit
error envelope (never a silent empty success).
"""
from __future__ import annotations
import hashlib
import json
import uuid
import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from app.roster_parser import ROSTER_CONTRACT_VERSION, ROSTER_PARSER_VERSION
from app.roster_parser.errors import InvalidTimezoneContextError, RosterError
from app.roster_parser.normalize import build_user_text, normalize_preview, parse_model_json, SYSTEM_PROMPT
from app.roster_parser.provider import MiniMaxClient, ProviderConfig
from app.roster_parser.rendering import CONTENT_TYPE_JPEG, CONTENT_TYPE_PDF, CONTENT_TYPE_PNG, RosterLimits, enforce_upload_limit, render_pdf_pages, sniff_content_type
from app.domain.timeline.baseline import generate_baseline
from app.services.reactive_timeline import queue_review, run_queued_review
router = APIRouter()

def _default_provider_factory() -> MiniMaxClient:
    return MiniMaxClient(ProviderConfig.from_env())
_provider_factory = _default_provider_factory

class RosterConfirmation(BaseModel):
    user_id: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r'^[a-fA-F0-9]{64}$')
    coverage_month: str = Field(pattern=r'^\d{4}-(0[1-9]|1[0-2])$')
    proposal: dict[str, Any]
    profile: dict[str, Any] | None = None

def _generate_month_baseline(store, body):
    year, month = map(int, body.coverage_month.split('-'))
    try:
        return generate_baseline(store, body.user_id, date(year, month, 1),
                                 date(year, month, calendar.monthrange(year, month)[1]), body.profile)
    except (ValueError, TypeError, KeyError):
        return {'status': 'profile_required', 'created': 0}

@router.post('/api/v1/roster/confirm')
def confirm_roster(body: RosterConfirmation, request: Request, background_tasks: BackgroundTasks):
    """Persist the complete confirmed preview and its complete timeline events."""
    service = request.app.state.timeline_service
    proposal = body.proposal
    items = proposal.get('proposed_events')
    if not body.user_id.strip() or not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        raise HTTPException(status_code=400, detail={'error_code': 'invalid_proposal', 'message': 'Invalid user or proposed events'})
    encoded = json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    file_hash = body.file_sha256.lower()
    with service.store.transaction() as conn:
        existing = conn.execute('SELECT proposal_json, result_json FROM roster_confirmations WHERE user_id=? AND file_sha256=? AND coverage_month=?', (body.user_id, file_hash, body.coverage_month)).fetchone()
        if existing:
            if existing['proposal_json'] != encoded:
                raise HTTPException(status_code=409, detail={'error_code': 'confirmation_conflict', 'message': 'This file and month were already confirmed with different results'})
            result = json.loads(existing['result_json'])
            if body.profile:
                result['baseline'] = _generate_month_baseline(service.store, body)
                if result['baseline']['created'] or result['baseline'].get('profile_changed'):
                    _queue_month_review(background_tasks, service.store, body)
            return result
        confirmation_id = str(uuid.uuid4())
        imported, pending = [], []
        for index, item in enumerate(items):
            kind = item.get('event_type')
            data, times = item.get('schedule_payload'), item.get('timeline')
            if kind not in ('flight', 'duty') or not isinstance(data, dict) or not isinstance(times, dict):
                pending.append({'seq': index, 'reason': 'invalid_event', 'fields': []})
                continue
            missing = data.get('pending', [])
            if not isinstance(missing, list):
                raise HTTPException(status_code=400, detail={'error_code': 'invalid_proposal', 'message': 'Invalid pending fields'})
            required = ('departure', 'arrival', 'origin_tz', 'destination_tz') if kind == 'flight' else ('duty_start', 'duty_end')
            missing = sorted(set(missing) | {field for field in required if not data.get(field)})
            local_key, zone_key = ('departure_local', 'departure_timezone') if kind == 'flight' else ('duty_start_local', 'duty_timezone')
            if missing or not times.get(local_key) or not times.get(zone_key):
                pending.append({'seq': index, 'reason': 'incomplete_event', 'fields': missing or [local_key, zone_key]})
                continue
            event = service.ingest_schedule_event(user_id=body.user_id, dedupe_key=f'roster:{confirmation_id}:{index}', occurred_at=times[local_key], timezone=times[zone_key], source='system', status='planned', title=kind, payload={**data, 'roster_confirmation_id': confirmation_id, 'roster_item': item})
            imported.append({'event_id': event.event_id, 'title': event.title, 'occurred_at_utc': event.to_dict()['occurred_at_utc']})
        result = {'import_id': confirmation_id, 'coverage_month': body.coverage_month, 'imported': imported, 'pending_items': pending, 'duplicates': [], 'unresolved_items': proposal.get('unresolved_items', []), 'blank_days': []}
        result['baseline'] = _generate_month_baseline(service.store, body)
        if result['baseline']['status'] == 'generated':
            _queue_month_review(background_tasks, service.store, body)
        from datetime import datetime, timezone
        conn.execute('INSERT INTO roster_confirmations VALUES (?, ?, ?, ?, ?, ?, ?)', (confirmation_id, body.user_id, file_hash, body.coverage_month, encoded, json.dumps(result, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return result

def _queue_month_review(tasks, store, body):
    year, month = map(int, body.coverage_month.split('-'))
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1]) + timedelta(days=1)
    zone = ZoneInfo(body.profile['home_timezone'])
    job_id = queue_review(store, body.user_id,
                          datetime.combine(first, time(), zone).astimezone(timezone.utc),
                          datetime.combine(last, time(), zone).astimezone(timezone.utc))
    tasks.add_task(run_queued_review, store, job_id)

@router.get('/api/v1/roster/confirmations')
def list_confirmations(request: Request, user_id: str):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail={'error_code': 'invalid_user', 'message': 'User ID is required'})
    store = request.app.state.timeline_service.store
    with store.lock:
        rows = store.connection.execute('SELECT confirmation_id, file_sha256, coverage_month, proposal_json, result_json, created_at FROM roster_confirmations WHERE user_id=? ORDER BY created_at DESC', (user_id,)).fetchall()
        return [{'confirmation_id': row['confirmation_id'], 'file_sha256': row['file_sha256'], 'coverage_month': row['coverage_month'], 'proposal': json.loads(row['proposal_json']), 'result': json.loads(row['result_json']), 'created_at': row['created_at']} for row in rows]

@router.post('/api/v1/roster/preview')
def preview_roster(file: UploadFile=File(...), home_timezone: str | None=Form(None), airport_timezone_map: str | None=Form(None), roster_year: int | None=Form(None), roster_month: int | None=Form(None)) -> dict[str, Any]:
    """Parse an uploaded roster file into a confirmation-ready preview."""
    request_id = str(uuid.uuid4())
    try:
        limits = RosterLimits.from_env()
        data = file.file.read(limits.max_upload_bytes + 1)
        for event in _preview_steps(data, limits, home_timezone, airport_timezone_map, roster_year, roster_month):
            if event['type'] == 'complete':
                return event['result']
        raise RuntimeError('preview ended without a result')
    except RosterError as exc:
        raise HTTPException(status_code=exc.http_status, detail={'error_code': exc.error_code, 'message': exc.message, 'field': None, 'request_id': request_id}) from exc

@router.post('/api/v1/roster/preview/stream')
def preview_roster_stream(file: UploadFile=File(...), home_timezone: str | None=Form(None), airport_timezone_map: str | None=Form(None), roster_year: int | None=Form(None), roster_month: int | None=Form(None)):
    """Stream actual pipeline boundaries; no invented percentages or model tokens."""
    limits = RosterLimits.from_env()
    data = file.file.read(limits.max_upload_bytes + 1)
    request_id = str(uuid.uuid4())

    def events():
        try:
            for event in _preview_steps(data, limits, home_timezone, airport_timezone_map, roster_year, roster_month):
                yield ('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n')
        except (RosterError, HTTPException) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else {'error_code': exc.error_code, 'message': exc.message}
            if not isinstance(detail, dict):
                detail = {'message': str(detail)}
            yield ('data: ' + json.dumps({'type': 'error', 'error': detail.get('error_code', 'invalid_request'), 'message': detail.get('message', '班表解析失败'), 'request_id': request_id}, ensure_ascii=False) + '\n\n')
        except Exception:
            yield ('data: ' + json.dumps({'type': 'error', 'error': 'parser_error', 'message': '班表解析发生错误，请重试', 'request_id': request_id}, ensure_ascii=False) + '\n\n')
    return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

def _preview_steps(data, limits, home_timezone, airport_timezone_map, roster_year, roster_month):
    yield {'type': 'progress', 'stage': 'validating'}
    airport_map = _validated_airport_map(airport_timezone_map)
    home_tz = _validated_home_timezone(home_timezone)
    _validated_roster_month(roster_year, roster_month)
    enforce_upload_limit(data, limits)
    content_type = sniff_content_type(data)
    pdf_page_count: int | None = None
    if content_type == CONTENT_TYPE_PDF:
        pdf_page_count = _pdf_page_count(data, limits)
        yield {'type': 'progress', 'stage': 'rendering', 'pages': pdf_page_count}
        images = [(CONTENT_TYPE_PNG, page) for page in render_pdf_pages(data, limits)]
    else:
        images = [(content_type, data)]
    page_count = len(images)
    provider = _provider_factory()
    yield {'type': 'progress', 'stage': 'recognizing', 'pages': page_count}
    result = provider.complete_vision(SYSTEM_PROMPT, images, build_user_text(page_count, roster_year, roster_month))
    yield {'type': 'progress', 'stage': 'normalizing'}
    model_json = parse_model_json(result.content)
    normalized = normalize_preview(model_json, page_count=page_count, home_timezone=home_tz, airport_timezone_map=airport_map, roster_year=roster_year, roster_month=roster_month)
    normalized['meta'] = {'model': result.model, 'file_sha256': hashlib.sha256(data).hexdigest(), 'file_size_bytes': len(data), 'content_type': content_type, 'render': {'dpi': limits.render_dpi if content_type == CONTENT_TYPE_PDF else None, 'pages_rendered': page_count, 'pdf_page_count': pdf_page_count}, 'usage': result.usage, 'http_attempts': result.http_attempts, 'streamed': result.streamed, 'elapsed_seconds': round(result.elapsed_seconds, 3), 'roster_parser_version': ROSTER_PARSER_VERSION, 'contract_version': ROSTER_CONTRACT_VERSION, 'limits': {'max_upload_bytes': limits.max_upload_bytes, 'max_pages': limits.max_pages, 'render_dpi': limits.render_dpi}}
    yield {'type': 'complete', 'result': normalized}

def _validated_home_timezone(home_timezone: str | None) -> str | None:
    if home_timezone is None:
        return None
    try:
        ZoneInfo(home_timezone)
    except Exception as exc:
        raise InvalidTimezoneContextError(f'home_timezone {home_timezone!r} is not a valid IANA timezone') from exc
    return home_timezone

def _validated_airport_map(airport_timezone_map: str | None) -> dict[str, str]:
    if airport_timezone_map is None or airport_timezone_map.strip() == '':
        return {}
    try:
        parsed = json.loads(airport_timezone_map)
    except ValueError as exc:
        raise InvalidTimezoneContextError('airport_timezone_map must be a JSON object of IATA -> IANA') from exc
    if not isinstance(parsed, dict):
        raise InvalidTimezoneContextError('airport_timezone_map must be a JSON object of IATA -> IANA')
    validated: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str) or len(key) != 3 or (not key.isalpha()) or (not key.isupper()):
            raise InvalidTimezoneContextError(f'airport_timezone_map key {key!r} must be a 3-letter uppercase IATA code')
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise InvalidTimezoneContextError(f'airport_timezone_map value {value!r} is not a valid IANA timezone') from exc
        validated[key] = value
    return validated

def _validated_roster_month(year: int | None, month: int | None) -> None:
    if year is not None and (not 1 <= year <= 9999):
        raise HTTPException(status_code=400, detail={'error_code': 'invalid_roster_context', 'message': 'roster_year must be between 1 and 9999', 'field': 'roster_year', 'request_id': str(uuid.uuid4())})
    if month is not None and (not 1 <= month <= 12):
        raise HTTPException(status_code=400, detail={'error_code': 'invalid_roster_context', 'message': 'roster_month must be between 1 and 12', 'field': 'roster_month', 'request_id': str(uuid.uuid4())})

def _pdf_page_count(data: bytes, limits: RosterLimits) -> int:
    import pypdfium2 as pdfium
    try:
        document = pdfium.PdfDocument(data)
    except Exception as exc:
        from app.roster_parser.errors import PdfRenderError
        raise PdfRenderError(f'invalid or unreadable PDF: {exc}') from exc
    try:
        return len(document)
    finally:
        document.close()

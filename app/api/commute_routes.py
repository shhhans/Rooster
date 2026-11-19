"""Homeport settings and generated travel time."""
import json
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from app.roster_parser.normalize import AIRPORT_TIMEZONES
from app.services.commute import sync_commutes
from app.services.reactive_timeline import refresh_issues

router = APIRouter(prefix='/api/v1/commute')

class Settings(BaseModel):
    user_id: str
    homeport: str
    commute_minutes: float | None = None

def refresh(store, user_id):
    now = datetime.now(timezone.utc)
    profile_row = store.connection.execute('SELECT profile_json FROM sleep_profiles WHERE user_id=?', (user_id,)).fetchone()
    profile = json.loads(profile_row[0]) if profile_row else {}
    if profile.get('preferred_sleep_start_hour') is None or profile.get('sleep_need_hours') is None:
        return
    rows = store.connection.execute("SELECT MAX(occurred_at_utc) FROM events WHERE user_id=? AND track='schedule' AND title='flight' AND retracted=0", (user_id,)).fetchone()
    if rows[0]:
        end = max(now + timedelta(days=1), datetime.fromisoformat(rows[0].replace('Z','+00:00')) + timedelta(days=3))
        refresh_issues(store, user_id, now, end)

@router.post('/settings')
def settings(body: Settings, request: Request, background_tasks: BackgroundTasks):
    home = body.homeport.strip().upper()
    if not body.user_id.strip() or home not in AIRPORT_TIMEZONES:
        raise HTTPException(status_code=400, detail='请输入已支持的基地机场代码')
    if body.commute_minutes is not None and not 0 <= body.commute_minutes <= 1440:
        raise HTTPException(status_code=400, detail='通勤时间必须在 0 至 1440 分钟之间')
    store = request.app.state.timeline_service.store
    with store.transaction() as conn:
        row = conn.execute('SELECT profile_json FROM sleep_profiles WHERE user_id=?', (body.user_id,)).fetchone()
        profile = json.loads(row[0]) if row else {}
        changed = (profile.get('homeport'), profile.get('commute_minutes'), profile.get('home_timezone')) != (home, body.commute_minutes, AIRPORT_TIMEZONES[home])
        profile.update(homeport=home, commute_minutes=body.commute_minutes, home_timezone=AIRPORT_TIMEZONES[home])
        conn.execute('INSERT INTO sleep_profiles VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at', (body.user_id, json.dumps(profile), datetime.now(timezone.utc).isoformat()))
        if changed: store._bump_input_revision(body.user_id, datetime.now(timezone.utc).isoformat())
        result = sync_commutes(store, body.user_id, profile)
    background_tasks.add_task(refresh, store, body.user_id)
    return {**result, 'home_timezone': profile['home_timezone']}

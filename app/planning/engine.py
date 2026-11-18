"""Deterministic sleep planning primitives.

This is an engineering model for exploring schedules, not a legal or clinical
assessment. All intervals are timezone-aware and calculations use UTC instants.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import random
from zoneinfo import ZoneInfo
from app.sleep_model.contracts import VERSION as MODEL_VERSION


LEGACY_MODEL_VERSION = "sleep-s-c-engineering-v1"
POLICY_VERSION = "schedule-constraints-v3"
GRID_MINUTES = 15
MIN_MAIN_HOURS = 2.0
MAX_WAKE_HOURS = 20.0


def _dt(value: str | datetime) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware instant required")
    return result.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _events(snapshot: dict) -> list[dict]:
    return deepcopy(snapshot.get("events", []))


def _interval(item: dict) -> tuple[datetime, datetime]:
    start, end = _dt(item["start_utc"]), _dt(item["end_utc"])
    if end <= start:
        raise ValueError(f"invalid interval: {item.get('id', '<unknown>')}")
    return start, end


def _overlap(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _merge(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[list[datetime]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _timezone_for(snapshot: dict) -> str:
    return (snapshot.get("profile") or {}).get("home_timezone", "UTC")


def _confirmed_environment(snapshot: dict) -> list[tuple[datetime, datetime, str, str]]:
    available = []
    for event in _events(snapshot):
        value = event.get("availability")
        confirmed = event.get("confirmed") is True or value in {True, "confirmed"}
        if (event.get("kind") == "environment" and confirmed
                and value not in {False, "unavailable", "blocked", "conditional", "unknown", "location_only"}
                and event.get("location") in {"home", "hotel", "aircraft"}):
            a, b = _interval(event)
            available.append((a, b, event.get("location"), event.get("timezone", _timezone_for(snapshot))))
    return available


def _aircraft_rest_windows(snapshot: dict, event_id: str) -> list[tuple[datetime, datetime]]:
    return [_interval(event) for event in _events(snapshot)
            if event.get('kind') == 'crew_rest' and event.get('authorized') is True
            and event_id in event.get('authorized_event_ids', [])]


def _blocked(snapshot: dict) -> list[tuple[datetime, datetime]]:
    now, horizon = _dt(snapshot["now_utc"]), _dt(snapshot["horizon_end_utc"])
    blocked = []
    for event in _events(snapshot):
        kind = event.get("kind")
        blocked_kind = kind in {"duty", "flight", "sleep_actual", "wake_actual", "commute", "preparation"}
        blocked_kind = blocked_kind or (kind == "commitment" and event.get("locked", True))
        blocked_kind = blocked_kind or (kind == "environment" and event.get("availability") in {False, "unavailable", "blocked"})
        if blocked_kind:
            start, end = _interval(event)
            aircraft_windows = _aircraft_rest_windows(snapshot, event.get('id'))
            if kind == "duty":
                before, after = (start - timedelta(minutes=90), start), (end, end + timedelta(minutes=75))
                blocked.extend((max(a, now), min(b, horizon)) for a, b in (before, after) if max(a, now) < min(b, horizon))
                duty_piece = (max(start, now), min(end, horizon))
                permitted = _merge([(max(duty_piece[0], a), min(duty_piece[1], b)) for a, b in aircraft_windows
                                    if _overlap(duty_piece, (a, b))])
                blocked.extend(_subtract(duty_piece, permitted))
            elif kind == "flight":
                piece = (max(start, now), min(end, horizon))
                permitted = _merge([(max(piece[0], a), min(piece[1], b)) for a, b in aircraft_windows
                                    if _overlap(piece, (a, b))])
                blocked.extend(_subtract(piece, permitted))
            else:
                start, end = max(start, now), min(end, horizon)
                if end > start:
                    blocked.append((start, end))
    return _merge(blocked)


def available_windows(snapshot: dict) -> list[dict]:
    """Return windows not occupied by known commitments inside the horizon.

    Sleep suggestions are provisional plans: absence of a location fact does
    not make an otherwise free interval unavailable.
    """
    snap = deepcopy(snapshot)
    now, horizon = _dt(snap["now_utc"]), _dt(snap["horizon_end_utc"])
    free = _subtract((now, horizon), _blocked(snap))
    result = []
    tzname = _timezone_for(snap)
    for start, end in free:
        result.append({"start_utc": _stamp(start), "end_utc": _stamp(end), "timezone": tzname})
    return result


def _aircraft_rest_exception(snapshot: dict, block: dict, event: dict,
                             overlap: tuple[datetime, datetime]) -> bool:
    rest_id = block.get('crew_rest_id')
    return any(rest.get('id') == rest_id and rest.get('kind') == 'crew_rest'
               and rest.get('authorized') is True
               and event.get('id') in rest.get('authorized_event_ids', [])
               and _interval(rest)[0] <= overlap[0] and _interval(rest)[1] >= overlap[1]
               for rest in _events(snapshot))


def check_plan(snapshot: dict, blocks: list[dict]) -> list[dict]:
    """Evaluate hard scheduling constraints; legal rules remain not_evaluated."""
    snap = deepcopy(snapshot)
    now, horizon = _dt(snap["now_utc"]), _dt(snap["horizon_end_utc"])
    checks: list[dict] = []
    parsed = []
    for block in deepcopy(blocks):
        try:
            start, end = _interval(block)
            parsed.append((block, start, end))
            if start < now or end > horizon:
                checks.append({"id": "block_horizon", "status": "fail", "block_id": block.get("id"), "message": "sleep block outside planning horizon"})
            minimum = (snap.get('scheduling_policy') or {}).get('minimum_sleep_minutes')
            if minimum is not None and (end-start).total_seconds() < minimum*60:
                checks.append({'id':'configured_minimum_sleep','status':'fail','block_id':block.get('id')})
            if block.get('crew_rest_id'):
                rest = next((e for e in _events(snap) if e.get('id') == block['crew_rest_id']
                             and e.get('kind') == 'crew_rest' and e.get('authorized') is True), None)
                if rest is None or not (_interval(rest)[0] <= start and end <= _interval(rest)[1]):
                    checks.append({'id': 'crew_rest_authorization', 'status': 'fail',
                                   'block_id': block.get('id'), 'event_id': block['crew_rest_id']})
        except (KeyError, TypeError, ValueError) as exc:
            checks.append({"id": "block_interval", "status": "fail", "block_id": block.get("id"), "message": str(exc)})
    for i, (block, start, end) in enumerate(parsed):
        for other, a, b in parsed[i + 1:]:
            if _overlap((start, end), (a, b)):
                checks.append({"id": "sleep_overlap", "status": "fail", "block_id": block.get("id"), "other_id": other.get("id")})
        for event in _events(snap):
            blocked_kind = event.get("kind") in {"duty", "flight", "sleep_actual", "wake_actual", "sleep", "exercise", "meal", "commute", "preparation"}
            blocked_kind = blocked_kind or (event.get("kind") == "commitment" and event.get("locked", True))
            blocked_kind = blocked_kind or (event.get("kind") == "environment" and event.get("availability") in {False, "unavailable", "blocked"})
            if blocked_kind:
                a, b = _interval(event)
                if event.get("kind") == "duty":
                    buffer_overlap = _overlap((start, end), (a - timedelta(minutes=90), a)) or _overlap((start, end), (b, b + timedelta(minutes=75)))
                    actual_overlap = _overlap((start, end), (a, b))
                    if buffer_overlap or (actual_overlap and not _aircraft_rest_exception(snap, block, event,
                                                                                           (max(start, a), min(end, b)))):
                        checks.append({"id": "sleep_event_overlap", "status": "fail", "block_id": block.get("id"), "event_id": event.get("id")})
                elif _overlap((start, end), (a, b)) and not (event.get("kind") == "flight" and
                        _aircraft_rest_exception(snap, block, event, (max(start, a), min(end, b)))):
                    checks.append({"id": "sleep_event_overlap", "status": "fail", "block_id": block.get("id"), "event_id": event.get("id")})
    events = _events(snap)
    duties = [(e, *_interval(e)) for e in events if e.get("kind") == "duty"]
    for i, (left, a, b) in enumerate(duties):
        for right, c, d in duties[i + 1:]:
            overlap = (max(a, c), min(b, d))
            if overlap[1] > overlap[0]:
                checks.append({"id": "duty_overlap", "status": "fail", "event_id": left.get("id"), "other_id": right.get("id")})
    for event in events:
        if event.get("kind") != "protected_rest":
            continue
        start, end = _interval(event)
        if end <= now:
            continue  # historical conflicts remain in execution audit, not future feasibility
        minimum = float(event.get("minimum_hours", (end - start).total_seconds() / 3600))
        if minimum < 0 or minimum * 3600 > (end - start).total_seconds():
            checks.append({"id": "protected_rest_definition", "status": "fail", "event_id": event.get("id"), "message": "minimum_hours exceeds protected interval"})
            continue
        obstacles = [(e, _interval(e)) for e in events
                     if e.get("kind") in {"duty", "commitment", "commute"}]
        occupied = _merge([(max(start, a), min(end, b)) for _, (a, b) in obstacles
                           if _overlap((start, end), (a, b))])
        free_hours = max(((b - a).total_seconds() / 3600 for a, b in _subtract((start, end), occupied)), default=0.0)
        if free_hours + 1e-9 < minimum:
            checks.append({"id": "protected_rest_preserved", "status": "fail", "event_id": event.get("id"),
                           "required_hours": minimum, "available_hours": round(free_hours, 2)})
    wake_limit = (snap.get('scheduling_policy') or {}).get('maximum_wake_hours')
    if wake_limit is not None and _future_longest_wake(snap, blocks) > wake_limit:
        checks.append({"id": "maximum_continuous_wake", "status": "fail",
                       "longest_wake_hours": round(_future_longest_wake(snap, blocks), 2), "limit_hours": wake_limit})
    if not any(c["status"] == "fail" for c in checks):
        checks.append({"id": "schedule_constraints", "status": "pass"})
    checks.append({"id": "aviation_legal_compliance", "status": "not_evaluated", "message": "no jurisdiction-specific legal rules are encoded"})
    return checks


def _subtract(interval: tuple[datetime, datetime], occupied: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    pieces, cursor = [], interval[0]
    for start, end in occupied:
        if start > cursor:
            pieces.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < interval[1]:
        pieces.append((cursor, interval[1]))
    return pieces


def _sleep_intervals(snapshot: dict, blocks: list[dict] | None) -> list[tuple[datetime, datetime]]:
    events = _events(snapshot)
    intervals = []
    for e in events:
        if e.get("kind") == "sleep_actual" and _dt(e["start_utc"]) < _dt(snapshot["now_utc"]):
            a, b = _interval(e)
            intervals.append((a, min(b, _dt(snapshot["now_utc"]))))
    initial = snapshot.get("initial_state") or {}
    if initial.get("last_sleep_start_utc") and initial.get("last_sleep_end_utc"):
        a, b = _dt(initial["last_sleep_start_utc"]), min(_dt(initial["last_sleep_end_utc"]), _dt(snapshot["now_utc"]))
        if b > a:
            intervals.append((a, b))
    if blocks is not None:
        intervals.extend(_interval(b) for b in deepcopy(blocks))
    return _merge([(a, b) for a, b in intervals if b > a])


def _wake_anchor(snapshot: dict, now: datetime) -> datetime | None:
    """Latest known wake instant from actual history or explicit initial state."""
    initial = snapshot.get("initial_state") or {}
    anchors = [_dt(initial["last_wake_utc"])] if initial.get("last_wake_utc") else []
    for event in _events(snapshot):
        if event.get("kind") == "sleep_actual" and _dt(event["end_utc"]) <= now:
            anchors.append(_dt(event["end_utc"]))
    if initial.get("last_sleep_end_utc"):
        end = _dt(initial["last_sleep_end_utc"])
        if end <= now:
            anchors.append(end)
    return max(anchors) if anchors else None


def _future_longest_wake(snapshot: dict, blocks: list[dict]) -> float:
    """Longest projected wake interval, allowing recovery from an already overdue state."""
    now, horizon = _dt(snapshot["now_utc"]), _dt(snapshot["horizon_end_utc"])
    anchor = _wake_anchor(snapshot, now)
    # Report the real known waking interval even if already beyond an optional policy limit.
    cursor = anchor if anchor is not None else now
    longest = 0.0
    for start, end in sorted((_interval(block) for block in blocks)):
        if end <= now:
            continue
        start = max(start, now)
        longest = max(longest, (start - cursor).total_seconds() / 3600)
        cursor = max(cursor, end)
    longest = max(longest, (horizon - cursor).total_seconds() / 3600)
    return longest


def _wake_stats(sleeps: list[tuple[datetime, datetime]], now: datetime, horizon: datetime,
                initial_state: dict | None = None) -> tuple[float, float]:
    # Merge the currently-awake interval across now so history plus forecast is
    # measured as one gap. Historical completed gaps are left to separate audits.
    anchor = None
    last_wake = (initial_state or {}).get("last_wake_utc")
    if last_wake:
        anchor = _dt(last_wake)
    for a, b in sleeps:
        if b <= now:
            anchor = max(anchor or b, b)
    cursor = anchor or now
    longest = 0.0
    for a, b in sleeps:
        if a < now:
            continue
        longest = max(longest, (a - cursor).total_seconds() / 3600)
        cursor = max(cursor, b)
    tail = max(0.0, (horizon - cursor).total_seconds() / 3600)
    return max(longest, tail), tail


def _process_metrics(snapshot: dict, sleeps: list[tuple[datetime, datetime]]) -> tuple[float, float, float]:
    """Simulate exponential Process S and fixed-phase circadian Process C at 15m steps."""
    now, horizon = _dt(snapshot["now_utc"]), _dt(snapshot["horizon_end_utc"])
    profile = snapshot.get("profile") or {}
    initial = snapshot.get("initial_state") or {}
    tau_wake = max(1.0, float(profile.get("tau_wake_hours", 18.2)))
    tau_sleep = max(0.5, float(profile.get("tau_sleep_hours", 4.2)))
    s_value = min(1.0, max(0.0, float(initial.get("process_s", 0.5))))
    tz = ZoneInfo(profile.get("home_timezone", "UTC"))
    phase = float(profile.get("circadian_phase_hour", 3.0))
    duties = [(a, b) for e in _events(snapshot) if e.get("kind") == "duty" for a, b in [_interval(e)]]
    start = now - timedelta(hours=72)
    step = timedelta(minutes=15)
    time_cursor, integral_s, integral_c_sleep, sleep_duration = start, 0.0, 0.0, 0.0
    burden_values = []
    while time_cursor < horizon:
        step_end = min(horizon, time_cursor + step)
        duration = (step_end - time_cursor).total_seconds() / 3600
        asleep = any(a < step_end and b > time_cursor for a, b in sleeps)
        if asleep:
            s_value *= math.exp(-duration / tau_sleep)
            sleep_duration += duration
        else:
            s_value = 1.0 - (1.0 - s_value) * math.exp(-duration / tau_wake)
        local = time_cursor.astimezone(tz)
        biohour = (phase + local.hour + local.minute / 60) % 24
        c_sleep = (1.0 + math.cos(2.0 * math.pi * biohour / 24.0)) / 2.0
        integral_s += s_value * duration
        if asleep:
            integral_c_sleep += c_sleep * duration
        if step_end > now and any(a < step_end and b > max(time_cursor, now) for a, b in duties):
            burden_values.append(min(1.0, max(0.0, 0.55 * s_value + 0.45 * c_sleep)))
        time_cursor = step_end
    total_hours = max((horizon - start).total_seconds() / 3600, 0.25)
    process_s_mean = integral_s / total_hours
    circadian_fit = integral_c_sleep / max(sleep_duration, 0.25)
    duty_burden = sum(burden_values) / len(burden_values) if burden_values else 0.0
    return process_s_mean, circadian_fit, duty_burden


def score_timeline_v1(snapshot: dict, sleep_blocks: list[dict] | None = None) -> dict:
    """Compute deterministic two-process (S/C) engineering metrics."""
    snap = deepcopy(snapshot)
    now, horizon = _dt(snap["now_utc"]), _dt(snap["horizon_end_utc"])
    sleeps = _sleep_intervals(snap, sleep_blocks)
    span_start = now - timedelta(hours=72)
    clipped = [(max(a, span_start), min(b, horizon)) for a, b in sleeps if b > span_start and a < horizon]
    sleep_hours = sum((b - a).total_seconds() / 3600 for a, b in clipped)
    rolling = {}
    for hours in (24, 48, 72):
        left = now - timedelta(hours=hours)
        rolling[str(hours)] = sum(max(0.0, (min(b, now) - max(a, left)).total_seconds() / 3600)
                                  for a, b in sleeps if b > left and a < now)
    longest_wake, _ = _wake_stats(sleeps, now, horizon, snap.get("initial_state"))
    fragmentation = max(0, len(clipped) - 1)
    process_s_mean, circadian_fit, duty_burden = _process_metrics(snap, sleeps)
    profile = snap.get("profile") or {}
    target = float(profile.get("sleep_need_hours", 8.0))
    horizon_hours = max(0.0, (horizon - now).total_seconds() / 3600)
    target_hours = target * min(3.0, horizon_hours / 24.0)
    # Process S follows exponential rise while awake and decay while asleep.
    # Process C is a fixed local-time phase used for duty burden and sleep fit.
    projected_sleep = rolling["72"] + sum(max(0.0, (min(b, now + timedelta(hours=72)) - max(a, now)).total_seconds() / 3600)
                                            for a, b in sleeps if b > now and a < now + timedelta(hours=72))
    circadian_penalty = min(20.0, 20.0 * (1.0 - circadian_fit))
    fragmentation_penalty = min(15.0, fragmentation * 2.0)
    burden_penalty = min(15.0, duty_burden * 15.0)
    adequacy = min(1.0, projected_sleep / max(target_hours, 0.25))
    s_penalty = min(40.0, (1.0 - adequacy) * 30.0 + max(0.0, longest_wake - 16.0) * 2.0 + process_s_mean * 10.0)
    utility = max(0.0, min(100.0, 100.0 - s_penalty - circadian_penalty - fragmentation_penalty - burden_penalty))
    return {"utility": round(utility, 2), "longest_wake_hours": round(longest_wake, 2),
            "rolling_sleep_hours_24": round(rolling["24"], 2), "rolling_sleep_hours_48": round(rolling["48"], 2),
            "rolling_sleep_hours_72": round(rolling["72"], 2), "sleep_hours": round(sleep_hours, 2),
            "fragmentation": fragmentation, "duty_burden": round(duty_burden, 2),
            "process_s_mean": round(process_s_mean, 4), "circadian_sleep_fit": round(circadian_fit, 4),
            "model_version": LEGACY_MODEL_VERSION, "target_sleep_hours": round(target_hours, 2)}


def score_timeline(snapshot: dict, sleep_blocks: list[dict] | None = None, *, model_version=None,
                   sleep_count=None) -> dict:
    """Explicitly versioned adapter. Frozen reference_v1 never calls this function.

    ``sleep_count`` is the explicit sleep-count compensation switch forwarded to
    the scoring path (``None`` keeps the legacy, enabled behaviour).
    """
    version = model_version or snapshot.get('sleep_model_version', MODEL_VERSION)
    if version == LEGACY_MODEL_VERSION:
        return score_timeline_v1(snapshot, sleep_blocks)
    if version != MODEL_VERSION:
        raise ValueError('unknown sleep model version: ' + version)
    from .adapter import score
    return score(snapshot, sleep_blocks, sleep_count=sleep_count)


def _candidate_blocks(snapshot: dict, requests: dict | None = None) -> list[list[dict]]:
    """Build full-horizon schedules, exploring circadian offsets and recovery naps."""
    windows = available_windows(snapshot)
    now, horizon = _dt(snapshot["now_utc"]), _dt(snapshot["horizon_end_utc"])
    profile = snapshot.get("profile") or {}
    options = requests if isinstance(requests, dict) else {}
    target = float(options.get("sleep_hours", profile.get("sleep_need_hours", 8.0)))
    target = max(MIN_MAIN_HOURS, min(12.0, target))
    offsets = (0.0, -1.5, 1.5, -3.0, 3.0)
    grid = timedelta(minutes=GRID_MINUTES)
    candidates = []
    tz = ZoneInfo(_timezone_for(snapshot))
    anchor = _wake_anchor(snapshot, now)
    habits = profile.get("sleep_start_distribution") or [profile.get("preferred_sleep_start_hour", 22.0)]
    preferences = sorted({(float(habit) + offset) % 24 for habit in habits for offset in offsets})
    for preferred in preferences:
        blocks, wake_cursor = [], anchor or now
        # Schedule a main opportunity near the local preferred time each day.
        day = now.astimezone(tz).date()
        while wake_cursor < horizon:
            earliest = max(now, wake_cursor + timedelta(hours=16))
            latest = max(now, wake_cursor + timedelta(hours=MAX_WAKE_HOURS))
            candidates_for_day = []
            for day_delta in range(-1, 4):
                local_day = day + timedelta(days=day_delta)
                hour = preferred
                local_target = datetime.combine(local_day, datetime.min.time(), tz).replace(
                    hour=int(hour), minute=int((hour % 1) * 60))
                candidates_for_day.append(local_target.astimezone(timezone.utc))
            candidates_for_day.sort(key=lambda t: (t < earliest, abs((t - earliest).total_seconds())))
            selected = None
            for preferred_start in candidates_for_day:
                desired = max(earliest, preferred_start)
                for window in windows:
                    a, b = _dt(window["start_utc"]), _dt(window["end_utc"])
                    if b - a < timedelta(hours=MIN_MAIN_HOURS):
                        continue
                    start = max(a, desired)
                    # Round upward to the 15-minute grid.
                    seconds = (start - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
                    start = datetime(1970, 1, 1, tzinfo=timezone.utc) + grid * math.ceil(seconds / grid.total_seconds())
                    if start > latest or start + timedelta(hours=MIN_MAIN_HOURS) > b:
                        continue
                    duration = min(target, (b - start).total_seconds() / 3600)
                    role = "main" if duration >= MIN_MAIN_HOURS else "nap"
                    if duration < MIN_MAIN_HOURS:
                        duration = min(2.0, (b - start).total_seconds() / 3600)
                    if duration < MIN_MAIN_HOURS:
                        continue
                    selected = {"id": f"sleep-{len(blocks) + 1}", "start_utc": _stamp(start),
                                "end_utc": _stamp(start + timedelta(hours=duration)),
                                "timezone": window["timezone"], "role": role}
                    break
                if selected:
                    break
            if not selected:
                # A nap can bridge a long daytime gap when no main window fits.
                for window in windows:
                    a, b = _dt(window["start_utc"]), _dt(window["end_utc"])
                    start = max(a, earliest)
                    if start <= latest and start + timedelta(minutes=30) <= b:
                        selected = {"id": f"nap-{len(blocks) + 1}", "start_utc": _stamp(start),
                                    "end_utc": _stamp(start + timedelta(minutes=30)),
                                    "timezone": window["timezone"], "role": "nap"}
                        break
            if not selected:
                break
            blocks.append(selected)
            wake_cursor = _dt(selected["end_utc"])
            day = wake_cursor.astimezone(tz).date()
        if blocks:
            candidates.append(blocks)
    return candidates


def plan(snapshot: dict, requests: dict | None = None, *, sleep_count=None) -> dict:
    """Generate and deterministically rank feasible candidate sleep schedules.

    ``sleep_count`` is the explicit sleep-count compensation switch: ``None``
    (default) keeps the legacy behaviour, ``False`` ablates the compensation in
    the beam objective and in the candidate scoring. The switch never changes
    the candidate generation or the hard constraints.
    """
    snap = deepcopy(snapshot)
    request_options = deepcopy(requests or {})
    requests = deepcopy(request_options)
    candidates = []
    requests = requests if isinstance(requests, list) else (requests or {}).get("requests", [])
    from .search import generate, VERSION as SEARCH_VERSION
    from .evaluation import evaluate
    initialization = evaluate(snap, [], scenarios=False, sleep_count=sleep_count)
    if initialization.get('utility') is None:
        return {'status':'no_candidate_found','candidates':[],'selected':None,
                'checks':check_plan(snap,[]),'model_version':MODEL_VERSION,'policy_version':POLICY_VERSION,
                'revision':snap.get('revision'),'search_version':SEARCH_VERSION,
                'search':{'status':'invalid_model_input','error':initialization.get('error')}}
    schedules, search_info = generate(snap, available_windows(snap), initialization['initial_state']['S'],
                                      reference_trace=initialization['trace'], sleep_count=sleep_count)
    # Retain legacy seeds for coverage during migration; they no longer define search coverage.
    schedules += _candidate_blocks(snap, request_options)
    unique = {}; refined = []
    for blocks in schedules:
        key=tuple((b['start_utc'],b['end_utc']) for b in blocks)
        unique.setdefault(key,blocks)
    schedules=list(unique.values())
    # Bounded 15-minute local translations around the leading beam schedules.
    for blocks in schedules[:2]:
        for minutes in (-15,15):
            shifted=deepcopy(blocks)
            for block in shifted:
                for key in ('start_utc','end_utc'):
                    block[key]=_stamp(_dt(block[key])+timedelta(minutes=minutes))
            refined.append(shifted)
    schedules += refined
    for blocks in schedules:
        checks = check_plan(snap, blocks)
        if any(c.get("status") == "fail" for c in checks):
            continue
        # Translate the already-supported intents; this does not add an LLM intent schema.
        score_snapshot = deepcopy(snap)
        from app.contracts.time import local_instant
        structured = list(score_snapshot.get('preferences', []))
        for index, request in enumerate(requests):
            if request.get('type') == 'free_window':
                left = local_instant(request['date'], request['start'], request['timezone'])
                right = local_instant(request['date'], request['end'], request['timezone'])
                if right <= left:
                    right += timedelta(days=1)
                structured.append({'id':f'intent-{index}','kind':'free_window','weight':1,
                    'scope_start_utc':_stamp(left),'scope_end_utc':_stamp(right)})
            elif request.get('type') == 'start_by':
                deadline=local_instant(request['date'],request['start'],request['timezone'])
                midpoint=(deadline-timedelta(minutes=90)).astimezone(ZoneInfo(request['timezone']))
                structured.append({'id':f'intent-{index}','kind':'bedtime','weight':1,
                    'target':midpoint.strftime('%H:%M'),'tolerance':90,'timezone':request['timezone'],
                    'scope_start_utc':_stamp(max(_dt(snap['now_utc']),deadline-timedelta(hours=12))),
                    'scope_end_utc':_stamp(max(_dt(snap['now_utc'])+timedelta(minutes=1),deadline+timedelta(hours=12)))})
            elif request.get('type') == 'reduce_changes':
                structured.append({'id':f'intent-{index}','kind':'change_cost','weight':1,
                    'scope_start_utc':snap['now_utc'],'scope_end_utc':snap['horizon_end_utc']})
        score_snapshot['preferences'] = structured
        metrics = score_timeline(score_snapshot, blocks, sleep_count=sleep_count)
        if metrics.get('utility') is None:
            continue
        preference_bonus = 0.0
        for request in requests:
            try:
                zone = ZoneInfo(request.get("timezone", _timezone_for(snap)))
                day = datetime.fromisoformat(request["date"]).date()
                start_local = datetime.combine(day, datetime.strptime(request["start"], "%H:%M").time(), zone)
                end_local = (datetime.combine(day, datetime.strptime(request.get("end", request["start"]), "%H:%M").time(), zone)
                             if request.get("type") == "free_window" else start_local + timedelta(minutes=1))
                if end_local <= start_local:
                    end_local += timedelta(days=1)
                req_interval = (start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc))
                block_intervals = [(block, _interval(block)) for block in blocks]
                if request.get("type") == "free_window" and not any(_overlap(interval, req_interval) for _, interval in block_intervals):
                    preference_bonus += 1.0
                elif request.get("type") == "start_by" and any(
                        block.get("role", "main") == "main" and req_interval[0] - timedelta(hours=3) <= interval[0] <= req_interval[0]
                        for block, interval in block_intervals):
                    preference_bonus += 1.0
            except (KeyError, TypeError, ValueError):
                continue
        # Legacy free-window bonuses must not be mixed into the v2 physiological score.
        if metrics.get('model_version') == LEGACY_MODEL_VERSION:
            metrics["utility"] = round(min(100.0, metrics["utility"] + preference_bonus), 2)
        candidates.append({"id": f"candidate-{len(candidates) + 1:03d}", "sleep_blocks": blocks,
                           "metrics": metrics, "checks": checks})
    candidates.sort(key=lambda c: (-c['metrics']['utility'], c["sleep_blocks"][0]["start_utc"]))
    diverse=[];seen=set()
    for candidate in candidates:
        signature=tuple((b['start_utc'],b['end_utc']) for b in candidate['sleep_blocks'])
        if signature in seen:continue
        seen.add(signature)
        # Suppress almost identical schedules, while allowing genuinely different short alternatives.
        def distance(other):
            a,b=candidate['sleep_blocks'],other['sleep_blocks']
            if len(a)!=len(b):return float('inf')
            return sum(abs((_dt(x[k])-_dt(y[k])).total_seconds()) for x,y in zip(a,b) for k in ('start_utc','end_utc'))/3600
        if any(distance(other)<1 for other in diverse):continue
        diverse.append(candidate)
        if len(diverse)==3:break
    ranking = ('sleep model utility including rolling sleep-count cost; external heuristics excluded'
               if sleep_count is None or bool(sleep_count) else
               'sleep model utility with the sleep-count compensation ablated; external heuristics excluded')
    search_info.update(evaluated_candidates=len(schedules),feasible_candidates=len(candidates),returned_candidates=len(diverse),
                       ranking=ranking,
                       ranking_version='search-ranking-v2',approximate=True)
    candidates=diverse
    # Stable IDs are assigned after ranking so repeated inputs produce identical output.
    for index, candidate in enumerate(candidates, 1):
        candidate["id"] = f"candidate-{index:03d}"
    selected = deepcopy(candidates[0]) if candidates else None
    return {"status": "ready" if selected else "no_candidate_found", "candidates": candidates,
            "selected": selected, "checks": selected["checks"] if selected else check_plan(snap, []),
            "model_version": MODEL_VERSION, "policy_version": POLICY_VERSION,
            "revision": snap.get("revision"),'search_version':SEARCH_VERSION,'search':search_info}


def randomized_strategy(snapshot: dict, seed: int) -> dict:
    """Return an unscored seeded baseline from physically feasible windows."""
    rng = random.Random(seed)
    schedules = [blocks for blocks in _candidate_blocks(snapshot)
                 if not any(check.get("status") == "fail" for check in check_plan(snapshot, blocks))]
    if not schedules:
        return {"status": "impossible", "sleep_blocks": [], "seed": seed}
    return {"status": "ready", "sleep_blocks": rng.choice(schedules), "seed": seed}

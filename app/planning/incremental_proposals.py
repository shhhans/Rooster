"""Deterministic proposal generation and tool-evaluation composition for the
incremental scheduling loop (INCREMENTAL-PLANNING-01).

Design invariants pinned by the plan review and `interface-agreement-08/09`:

- **F1:** model evaluation always runs on the caller's original snapshot
  (``now_utc`` = the real now) over the requested window (default
  ``[now, horizon_end]``) with the *accumulated* candidate blocks (all
  retained blocks of all segments). Segment-cloned snapshots are used **only** to seed
  ``search.generate``; they are never handed to the model.
- **F2/F3:** ``evaluation.evaluate`` answers with status dicts; only
  ``status == 'complete'`` with a non-None utility is scoreable. Unscoreable
  candidates are never retained.
- **Model-face rule** (agreements 06/07/09 C3): the event list handed to the
  model or the engine contains only interval-bearing events, and the
  planning-owned kinds ``rest_day`` / ``sleep_plan`` are stripped. Their
  semantics (no-duty facts, protection by reference) are resolved on the raw
  snapshot by planning, never by the model.
- Candidate/recommended arrangements travel only as ``compute``'s second
  parameter (``sleep_blocks``) and are never written into ``snapshot['events']``.
- The per-segment entry ``S`` extracted from a trace is a **planning-owned
  search seed**, not a model checkpoint and not resumable physiology.
"""
from copy import deepcopy
from datetime import timedelta
from zoneinfo import ZoneInfo

from app.contracts.time import dt, overlap, stamp

from .engine import MIN_MAIN_HOURS, available_windows, check_plan
from .evaluation import evaluate

GRID_MINUTES = 15
MOVE_OFFSETS_MINUTES = (15, 30, 60)
RESIZE_OFFSETS_MINUTES = (30,)
HABIT_OFFSETS_HOURS = (0.0, -1.5, 1.5, -3.0, 3.0)
CAP_PER_SOURCE = 4

PROTECTED_KINDS = frozenset({'duty', 'flight', 'sleep_actual', 'commute', 'preparation', 'protected_rest'})
PROTECTION_MARKERS = frozenset({'fixed_external', 'committed', 'actual'})
PLANNING_EVENT_KINDS = frozenset({'rest_day', 'sleep_plan'})


def fact_view(snapshot):
    """Model/engine-facing view of a snapshot.

    The model normalizes every visible event through ``interval(e)`` and the
    checkpoint identity hashes every event that starts before the knowledge
    cutoff, so the model face carries only interval-bearing events and never
    the planning-owned kinds ``rest_day``/``sleep_plan`` (agreement-06 §2,
    agreement-07 §1, agreement-09 C3). Everything else is passed through
    verbatim, including ``task_markers`` (agreement-09 C5: neither added nor
    stripped, a documented input property).
    """
    view = deepcopy(snapshot)
    events = []
    for event in view.get('events', []):
        if event.get('kind') in PLANNING_EVENT_KINDS:
            continue
        start, end = event.get('start_utc'), event.get('end_utc')
        if start is None or end is None:
            continue
        try:
            dt(start), dt(end)
        except (ValueError, TypeError, KeyError):
            continue
        events.append(event)
    view['events'] = events
    return view


def event_protected(event):
    """Frozen event-level protection union (interface-agreement.md §1)."""
    if event.get('protection') in PROTECTION_MARKERS:
        return True
    if event.get('kind') in PROTECTED_KINDS:
        return True
    return event.get('kind') == 'commitment' and event.get('locked', True)


def protected_event_ids(snapshot):
    return {e['id'] for e in snapshot.get('events', []) if event_protected(e)}


def commitment_protected_block_ids(snapshot):
    """Block ids protected by reference through commitment records.

    A commitment (``kind='commitment'``, ``locked`` default true, or
    ``protection='committed'``) references the plan block it commits to via
    ``target_event_id`` (defensively also ``plan_block_id``); the referenced
    block is protected as a consequence of that commitment
    (interface-agreement-05 §3, agreement-08 §4). Unmatched targets are
    reported separately (``unmatched_commitment_notes``), never ignored.
    """
    ids = set()
    for event in snapshot.get('events', []):
        if event.get('kind') != 'commitment':
            continue
        if not (event.get('locked', True) or event.get('protection') == 'committed'):
            continue
        for key in ('target_event_id', 'plan_block_id'):
            value = event.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def unmatched_commitment_notes(snapshot, plan_block_ids):
    """Truthful notes for commitments that reference no plan block and no event."""
    notes = []
    event_ids = {e.get('id') for e in snapshot.get('events', [])}
    for event in snapshot.get('events', []):
        if event.get('kind') != 'commitment' or not event.get('locked', True):
            continue
        target = event.get('target_event_id')
        if isinstance(target, str) and target and target not in plan_block_ids and target not in event_ids:
            notes.append('commitment ' + str(event.get('id')) + ' target_event_id ' + target +
                         ' matches no plan block or event; kept as protected context without a linked block')
    return notes


def ongoing_session_spans(snapshot):
    """Spans of sleep sessions already in execution (``actual_end_utc`` None)."""
    spans = []
    for session in (snapshot.get('sessions') or {}).values():
        if session.get('actual_end_utc') is not None:
            continue
        start = dt(session['actual_start_utc'])
        end = dt(session['expected_end_utc'])
        spans.append((session['id'], start, max(start, end), session.get('planned_block_id')))
    return spans


def block_protected(block, snapshot):
    """A plan block is protected when it carries an explicit convenience flag,
    is marker/kind-protected via a matching event id, is referenced by an
    active commitment (protection by reference — the authoritative path,
    agreement-08 §4), or is the planned block of an ongoing session."""
    if block.get('accepted') or block.get('committed'):
        return True
    if block.get('protection') in PROTECTION_MARKERS:
        return True
    if block.get('id') in protected_event_ids(snapshot):
        return True
    if block.get('id') in commitment_protected_block_ids(snapshot):
        return True
    for _, _, _, planned_block_id in ongoing_session_spans(snapshot):
        if planned_block_id is not None and planned_block_id == block.get('id'):
            return True
    return False


def segment_windows(facts, segment):
    """Free windows clipped to the segment bounds."""
    from .engine import available_windows
    left, right = dt(segment['start_utc']), dt(segment['end_utc'])
    out = []
    for window in available_windows(facts):
        a, b = dt(window['start_utc']), dt(window['end_utc'])
        start, end = max(left, a), min(right, b)
        if start < end:
            out.append({'start_utc': stamp(start), 'end_utc': stamp(end), 'timezone': window['timezone']})
    return out


def blocks_owned_by(blocks, segment):
    """Blocks whose start lies inside the segment (segment ownership rule)."""
    left, right = dt(segment['start_utc']), dt(segment['end_utc'])
    return [b for b in blocks if left <= dt(b['start_utc']) < right]


def segment_covers(blocks, segment):
    left, right = dt(segment['start_utc']), dt(segment['end_utc'])
    return any(left <= dt(b['start_utc']) < right for b in blocks)


def segment_clone(facts, segment):
    """Bounds-only clone of the *fact view* used exclusively to seed
    ``search.generate``.

    The caller's original snapshot is never evaluated on a clone: a cloned
    ``now_utc`` would misstate the knowledge cutoff for the model. The clone
    drops any ``model_checkpoint`` because the checkpoint identity belongs to
    the real ``now``; segment seeds re-initialize from visible history (a
    documented search-owned approximation — retention is always decided by a
    full evaluation on the original snapshot).
    """
    clone = deepcopy(facts)
    clone['now_utc'] = segment['start_utc']
    clone['horizon_end_utc'] = segment['end_utc']
    clone.pop('model_checkpoint', None)
    return clone


def _next_local_instant(anchor, hour, timezone_name):
    from app.contracts.time import local_instant
    zone = ZoneInfo(timezone_name)
    local = anchor.astimezone(zone)
    minutes = int(round((hour - int(hour)) * 60))
    hh, mm = int(hour) + minutes // 60, minutes % 60
    for day in (local.date(), local.date() + timedelta(days=1)):
        try:
            candidate = local_instant(day.isoformat(), '%02d:%02d' % (hh, mm), timezone_name)
        except ValueError:
            continue
        if candidate >= anchor:
            return candidate
    return None


def _overlaps_omitted(block, omitted_spans):
    span = (dt(block['start_utc']), dt(block['end_utc']))
    return any(overlap(span, (dt(o['start_utc']), dt(o['end_utc']))) for o in omitted_spans)


def habitual_proposals(facts, segment, omitted_spans=()):
    """Coverage-driven main-sleep/nap placements for an uncovered segment.

    Placements overlapping an omitted span (a feedback omission or a rejected
    replacement timing) are never proposed within the same run.
    """
    profile = facts.get('profile') or {}
    need = float(profile.get('sleep_need_hours', 8.0))
    base = float(profile['preferred_sleep_start_hour'])
    timezone_name = profile.get('home_timezone', 'UTC')
    hours = sorted({round((base + offset) % 24, 4) for offset in HABIT_OFFSETS_HOURS})
    proposals = []
    seen = set()
    for window in segment_windows(facts, segment):
        wa, wb = dt(window['start_utc']), dt(window['end_utc'])
        for hour in hours:
            start = _next_local_instant(wa, hour, timezone_name)
            if start is None or start >= wb:
                continue
            available = (wb - start).total_seconds() / 3600
            if available < 0.5:
                continue
            duration = min(need, available)
            block = {'id': None, 'start_utc': stamp(start),
                     'end_utc': stamp(start + timedelta(hours=duration)),
                     'timezone': window['timezone'],
                     'role': 'main' if duration >= MIN_MAIN_HOURS else 'nap'}
            if _overlaps_omitted(block, omitted_spans):
                continue
            key = (block['start_utc'], block['end_utc'], block['timezone'])
            if key in seen:
                continue
            seen.add(key)
            proposals.append({'kind': 'add', 'source': 'habitual', 'blocks': [block]})
            if len(proposals) >= CAP_PER_SOURCE:
                return proposals
    return proposals


def move_proposals(facts, blocks, segment):
    """Bounded move/resize proposals for unconfirmed blocks owned by the segment."""
    proposals = []
    for block in blocks_owned_by(blocks, segment):
        if block_protected(block, facts):
            continue
        start, end = dt(block['start_utc']), dt(block['end_utc'])
        for minutes in MOVE_OFFSETS_MINUTES:
            for sign in (1, -1):
                delta = timedelta(minutes=sign * minutes)
                after = dict(block, start_utc=stamp(start + delta), end_utc=stamp(end + delta))
                proposals.append({'kind': 'move', 'source': 'move', 'event_id': block['id'],
                                  'before': deepcopy(block), 'after': after})
        for minutes in RESIZE_OFFSETS_MINUTES:
            for sign in (1, -1):
                delta = timedelta(minutes=sign * minutes)
                if end + delta <= start:
                    continue
                after = dict(block, end_utc=stamp(end + delta))
                proposals.append({'kind': 'move', 'source': 'resize', 'event_id': block['id'],
                                  'before': deepcopy(block), 'after': after})
        if len(proposals) >= CAP_PER_SOURCE:
            break
    return proposals[:CAP_PER_SOURCE]


def build_proposals(facts, draft, segment):
    """Ordered deterministic proposal list for one step on one segment.

    Order: habitual coverage adds for uncovered segments, then bounded moves
    of unconfirmed blocks; the segment-reused search seed is materialized
    lazily by the caller once the cheap queue is exhausted. Already-tried
    signatures (recorded in the draft) are filtered by the caller.
    """
    blocks = draft['plan']['sleep_blocks']
    omitted_spans = draft.get('omitted_spans') or []
    proposals = []
    if not segment_covers(blocks, segment):
        proposals.extend(habitual_proposals(facts, segment, omitted_spans))
    proposals.extend(move_proposals(facts, blocks, segment))
    return proposals


def search_seed_proposals(facts, draft, segment, sleep_count):
    """Return generated search schedules that fit the Agent's edit batch."""
    from .search import generate
    clone = segment_clone(facts, segment)
    entry = (segment.get('entry_state') or {}).get('S')
    schedules, info = generate(clone, available_windows(clone), entry if entry is not None else .5,
                               reference_trace=(), sleep_count=sleep_count)
    omitted_spans = draft.get('omitted_spans') or []
    proposals = []
    for schedule in schedules:
        blocks = [dict(block, id=None) for block in schedule
                  if not _overlaps_omitted(block, omitted_spans)]
        if not blocks or len(blocks) > 8:
            continue
        proposals.append({'kind': 'add', 'source': 'search', 'blocks': blocks,
                          'search_status': info.get('status')})
    return proposals


def search_seed_proposal(facts, draft, segment, sleep_count):
    """Backward-compatible selector seed for the deterministic diagnostic path."""
    return next(iter(search_seed_proposals(facts, draft, segment, sleep_count)), None)


def proposal_key(proposal):
    """Stable signature for the draft's tried-list (no ids assigned yet)."""
    from .state import canonical
    material = {k: v for k, v in proposal.items() if k != 'requires_confirmation'}
    return canonical(material)


def candidate_blocks(blocks, proposal, id_assigner=None):
    """Apply a proposal to a block list without mutating it.

    New blocks get a real id before any model call — the evaluation chain
    string-concatenates block ids, so an anonymous ``id=None`` block would
    fail as ``invalid_input``. When the proposal pre-assigned ids
    (confirmation cards) they are kept; otherwise ``id_assigner`` supplies
    them, so the evaluated candidate and the committed diff carry identical
    identities. A reverted trial has consumed its id — deterministic, because
    the proposal order and the tried-list are deterministic.
    Returns ``(candidate, diff_specs)`` where ``diff_specs`` are
    ``(op, event_id_or_None, before, after)`` tuples.
    """
    kind = proposal['kind']
    if kind == 'add':
        new_blocks = []
        for block in proposal['blocks']:
            item = deepcopy(block)
            if item.get('id') is None and id_assigner is not None:
                item['id'] = id_assigner()
            new_blocks.append(item)
        return blocks + new_blocks, [('add', None, None, b) for b in new_blocks]
    if kind == 'move':
        event_id = proposal['event_id']
        candidate = [dict(proposal['after'], id=event_id) if b.get('id') == event_id else deepcopy(b)
                     for b in blocks]
        return candidate, [('move', event_id, proposal['before'], dict(proposal['after'], id=event_id))]
    if kind == 'remove':
        event_id = proposal['event_id']
        candidate = [deepcopy(b) for b in blocks if b.get('id') != event_id]
        return candidate, [('remove', event_id, proposal['before'], None)]
    if kind == 'replace':
        event_id = proposal['event_id']
        new_blocks = []
        for block in proposal['blocks']:
            item = deepcopy(block)
            if item.get('id') is None and id_assigner is not None:
                item['id'] = id_assigner()
            new_blocks.append(item)
        candidate = [deepcopy(b) for b in blocks if b.get('id') != event_id] + new_blocks
        return candidate, [('remove', event_id, proposal['before'], None)] + \
                          [('add', None, None, b) for b in new_blocks]
    raise ValueError('unknown proposal kind: ' + str(kind))


def hard_validate(facts, blocks):
    """Cheap feasibility gate before any model call.

    Historical blocks (ending at/before ``now``) and in-progress blocks
    (started before ``now``) are execution context, not future feasibility:
    they are skipped here and excluded from ``check_plan`` by the caller.
    The gate mirrors the model's input rules (no overlapping blocks) and the
    engine's window semantics (free of known occupied time).
    """
    now = dt(facts['now_utc'])
    reasons = []
    future = []
    for block in blocks:
        try:
            start, end = dt(block['start_utc']), dt(block['end_utc'])
        except (ValueError, TypeError, KeyError) as exc:
            return False, ['invalid block interval: ' + str(exc)]
        if end <= start:
            return False, ['invalid block interval: ' + str(block.get('id'))]
        if end <= now or start < now:
            continue
        future.append((block, start, end))
    spans = ongoing_session_spans(facts)
    for block, start, end in future:
        for session_id, a, b, _ in spans:
            if overlap((start, end), (a, b)):
                reasons.append('block ' + str(block.get('id')) + ' overlaps ongoing session ' + session_id)
    for i, (block, start, end) in enumerate(future):
        for other, a, b in future[i + 1:]:
            if overlap((start, end), (a, b)):
                reasons.append('blocks ' + str(block.get('id')) + ' and ' + str(other.get('id')) + ' overlap')
    windows = [(dt(w['start_utc']), dt(w['end_utc'])) for w in segment_windows(facts, {
        'start_utc': facts['now_utc'], 'end_utc': facts['horizon_end_utc']})]
    for block, start, end in future:
        if not any(a <= start and end <= b for a, b in windows):
            reasons.append('block ' + str(block.get('id')) + ' overlaps known occupied time or lies outside the planning horizon')
    return (not reasons), reasons


def constraint_report(facts, blocks):
    """Authoritative constraint report over the plan's future blocks."""
    from .engine import check_plan
    now = dt(facts['now_utc'])
    future = [b for b in blocks if dt(b['start_utc']) >= now]
    return check_plan(facts, future)


def summarize_constraints(report):
    return {'failed': [c for c in report if c.get('status') == 'fail'],
            'passed': [c for c in report if c.get('status') == 'pass'],
            'not_evaluated': [c for c in report if c.get('status') == 'not_evaluated']}


def run_evaluation(facts, blocks, *, sleep_count, start_utc=None, end_utc=None):
    """Tool evaluation through the public model composition (F1: the original
    snapshot's fact view, caller-selected window or default ``[now, horizon_end]``,
    accumulated candidate blocks).

    ``evaluate`` answers with status dicts instead of raising; the loop must
    inspect them: only ``status == 'complete'`` with a non-None utility is
    scoreable. Unscoreable candidates are never retained (plan review
    findings 2 and 3).
    """
    result = evaluate(facts, blocks, mode='forecast', scenarios=False, sleep_count=sleep_count,
                      start_utc=start_utc, end_utc=end_utc)
    scoreable = result.get('status') == 'complete' and result.get('utility') is not None
    return result, scoreable

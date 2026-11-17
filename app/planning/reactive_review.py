"""Compare a local repair against the effective plan on the same facts."""
from datetime import timedelta

from app.contracts.time import dt, interval, overlap, stamp
from app.planning.engine import check_plan
from app.planning.sleep_deficit import check_sleep_deficit
from app.planning.sleep_feasibility import check_sleep_feasibility, pressure_thresholds
from app.sleep_model.api import compute
from app.sleep_model.contracts import VERSION as MODEL_VERSION

PRESSURE_ISSUES = {'sleep_onset_pressure_low', 'sleep_maintenance_pressure_low'}


def review(snapshot, blocks, window):
    start, end = map(dt, window)
    events = [e for e in snapshot['events'] if e.get('end_utc')]
    future = [b for b in blocks if dt(b['end_utc']) > dt(snapshot['now_utc'])]
    facts = {e['id']: e for e in events + future}
    issues = []
    # Keep the whole context for constraint checks, then select local evidence.
    for check in check_plan(dict(snapshot, events=events), future):
        if check['status'] != 'fail' or check['id'] == 'block_horizon':
            continue
        ids = [check[k] for k in ('block_id', 'event_id', 'other_id') if check.get(k)]
        spans = [interval(facts[i]) for i in ids if i in facts]
        if check['id'] == 'sleep_event_overlap' and facts[check['event_id']]['kind'] == 'duty':
            a, b = spans[-1]
            spans[-1] = a - timedelta(minutes=90), b + timedelta(minutes=75)
        left = max(a for a, _ in spans) if spans else start
        right = min(b for _, b in spans) if spans else end
        if check['id'] == 'crew_rest_authorization' and check.get('block_id') in facts:
            left, right = interval(facts[check['block_id']])
        if left >= right or not overlap((left, right), (start, end)):
            continue
        issues.append({'type': 'hard_overlap' if check['id'].endswith('overlap') else 'hard_constraint',
                       'rule': check['id'], 'event_ids': ids, 'start_utc': stamp(left),
                       'end_utc': stamp(right), 'evidence': check})
    result = {'issues': issues, 'feasible': not issues,
              'start_utc': stamp(start), 'end_utc': stamp(end), 'physiological_loss': None}
    if any(i['rule'] == 'sleep_overlap' for i in issues):
        result['model_unavailable'] = 'overlapping sleep opportunities'
        return result
    try:
        model = compute(snapshot, future, start_utc=stamp(start), end_utc=stamp(end))
        thresholds, threshold_status = pressure_thresholds(model['provenance']['configuration'])
        anchor = dt(snapshot['now_utc'])
        proposals = sorted(future, key=lambda b: dt(b['start_utc']))
        pressure = check_sleep_feasibility(
            [dict(b, start_utc=stamp(max(anchor, dt(b['start_utc'])))) for b in proposals],
            anchor_utc=anchor, initial_s=model['initial_state']['S'],
            pressure_config=model['provenance']['configuration'], **thresholds,
        )
        for issue in pressure['issues']:
            block = facts[issue['proposal_id']]
            if not overlap(interval(block), (start, end)):
                continue
            if issue['type'] == 'sleep_onset_pressure_low' and dt(block['start_utc']) < anchor:
                continue
            ids = [issue['proposal_id']]
            if issue.get('previous_proposal_id'):
                ids.append(issue['previous_proposal_id'])
            issues.append({
                'type': issue['type'], 'rule': issue['type'], 'event_ids': ids,
                'start_utc': block['start_utc'], 'end_utc': block['end_utc'], 'evidence': issue,
            })
        result.update(feasible=not issues, sleep_pressure={
            'anchor_utc': stamp(anchor), 'initial_s': model['initial_state']['S'],
            'assumption': 'planned sleep occurs in full after the forecast cutoff',
            'threshold_status': threshold_status, 'model_version': MODEL_VERSION,
            'checks': [dict(check, status='not_evaluated') if
                       check['type'] == 'sleep_onset_pressure_low' and
                       dt(facts[check['proposal_id']]['start_utc']) < anchor else check
                       for check in pressure['checks']
                       if overlap(interval(facts[check['proposal_id']]), (start, end))],
            **thresholds,
        })
        duties = [e for e in model['normalized']['snapshot']['events'] if e['kind'] == 'duty']
        issues.extend(check_sleep_deficit(model['actual_sleep'], future, duties=duties,
                                         cutoff_utc=snapshot['now_utc'], start_utc=stamp(start),
                                         end_utc=stamp(end),
                                         config=model['provenance']['configuration'])['issues'])
        result.update(physiological_loss=model['physiological_loss'] if model['status'] == 'complete' else None,
                      assumptions=model['assumptions'],
                      task_burden=model['physiological_metrics']['task_markers'])
    except ValueError as exc:
        result['model_unavailable'] = str(exc)
        # A successful hard-conflict check alone cannot establish that sleep
        # pressure is sufficient. Preserve any confirmed hard failures.
        if result['feasible']:
            result['feasible'] = None
    return result


def matches(a, b):
    if a['type'] != b['type']:
        return False
    if a['type'] == 'sleep_deficit':
        return overlap(interval(a), interval(b))
    if a['type'] in PRESSURE_ISSUES:
        return a['evidence']['proposal_id'] == b['evidence']['proposal_id']
    return a['rule'] == b['rule'] and set(a['event_ids']) == set(b['event_ids'])


def compare(effective, candidate):
    before, after = effective['issues'], candidate['issues']
    previous, current = effective['physiological_loss'], candidate['physiological_loss']
    return {
        'effective': effective, 'candidate': candidate,
        'added': [b for b in after if not any(matches(a, b) for a in before)],
        'resolved_in_candidate': [a for a in before if not any(matches(a, b) for b in after)
                                  and (a['type'] not in PRESSURE_ISSUES or 'sleep_pressure' in candidate)
                                  and (a['type'] != 'sleep_deficit' or 'assumptions' in candidate)],
        'remaining': [b for b in after if any(matches(a, b) for a in before)],
        'physiological_improvement': previous - current if None not in (previous, current) else None,
    }

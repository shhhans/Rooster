from app.planning.sleep_deficit import check_sleep_deficit


def record(identifier, start, end):
    return {'id': identifier, 'start_utc': start, 'end_utc': end}


def test_rolling_deficit_uses_partial_overlaps_and_exact_crossings():
    cutoff = '2026-11-15T08:00:00Z'
    actual = [
        record('actual', '2026-11-14T00:30:00+08:00', '2026-11-14T08:30:00+08:00'),
        record('duplicate', '2026-11-13T18:30:00Z', '2026-11-14T00:30:00Z'),
    ]
    # Sleep rolls out at 16:30 on the 14th, then predicted sleep starts at
    # 12:15 on the 15th and reaches eight hours at 20:15.
    planned = [record('plan', '2026-11-15T12:00:00Z', '2026-11-15T21:00:00Z')]
    duties = [record('duty', '2026-11-15T09:00:00Z', '2026-11-15T11:00:00Z')]
    issues = check_sleep_deficit(
        actual, planned, cutoff_utc=cutoff, start_utc='2026-11-14T12:00:00Z',
        end_utc='2026-11-15T21:00:00Z', duties=duties, config={'latency_minutes': 15.0},
    )['issues']
    assert len(issues) == 1
    issue = issues[0]
    assert issue['type'] == 'sleep_deficit'
    assert issue['start_utc'] == '2026-11-14T16:30:00Z'
    assert issue['end_utc'] == '2026-11-15T20:15:00Z'
    assert issue['max_deficit_hours'] == 8
    assert issue['duty_ids'] == ['duty']


def test_exact_target_separates_deficits():
    cutoff = '2026-11-15T00:00:00Z'
    planned = [record('plan', cutoff, '2026-11-15T08:15:00Z')]
    issues = check_sleep_deficit(
        [], planned, cutoff_utc=cutoff, start_utc=cutoff,
        end_utc='2026-11-16T01:15:00Z', config={'latency_minutes': 15.0},
    )['issues']
    assert len(issues) == 2
    assert issues[0]['end_utc'] == '2026-11-15T08:15:00Z'
    assert issues[1]['start_utc'] == '2026-11-16T00:15:00Z'
    assert issues[1]['max_deficit_hours'] == 1
    assert issues[1]['evidence']['sleep_opportunity_ids'] == ['plan']


def test_missing_sleep_produces_one_issue_without_imputation():
    start = '2026-11-15T08:00:00Z'
    end = '2026-11-15T10:00:00Z'
    issues = check_sleep_deficit([], [], cutoff_utc=start, start_utc=start, end_utc=end)['issues']
    assert [(i['start_utc'], i['end_utc'], i['max_deficit_hours']) for i in issues] == [
        (start, end, 8),
    ]


def test_deficit_samples_cover_26_hours_in_two_hour_steps():
    start = '2026-11-15T00:00:00Z'
    end = '2026-11-16T02:00:00Z'
    issue = check_sleep_deficit([], [], cutoff_utc=start, start_utc=start, end_utc=end)['issues'][0]
    assert len(issue['samples']) == 13
    assert issue['samples'][0] == {'at_utc': start, 'window_start_utc': '2026-11-14T00:00:00Z', 'deficit_hours': 8.0}
    assert issue['samples'][-1]['at_utc'] == '2026-11-16T00:00:00Z'


def test_eight_hour_daily_sleep_has_no_default_latency_deficit():
    planned = [record('first', '2027-01-02T15:00:00Z', '2027-01-02T23:00:00Z'),
               record('second', '2027-01-03T15:00:00Z', '2027-01-03T23:00:00Z')]
    result = check_sleep_deficit([], planned, cutoff_utc='2027-01-01T00:00:00Z',
                                 start_utc='2027-01-02T23:00:00Z', end_utc='2027-01-03T23:00:00Z')
    assert result['issues'] == []
def test_inflight_estimates_interpolate_and_actual_suppresses():
    from datetime import datetime, timedelta, timezone
    from app.planning.sleep_deficit import check_sleep_deficit
    start = datetime(2027, 1, 1, tzinfo=timezone.utc)
    for hours, expected in [(7, 0), (8, 1), (10, 2), (12, 3), (14, 3.5), (16, 4), (18, 4.5)]:
        flight = {'id': 'flight', 'start_utc': start.isoformat(), 'end_utc': (start + timedelta(hours=hours)).isoformat()}
        end = start + timedelta(hours=24)
        result = check_sleep_deficit([], [], flights=[flight], cutoff_utc=start.isoformat(), start_utc=end.isoformat(), end_utc=(end + timedelta(minutes=1)).isoformat())
        issue = result['issues'][-1]
        estimates = issue['evidence']['inflight_sleep_estimates']
        assert len(estimates) == (1 if expected else 0)
        if estimates:
            a, b = (datetime.fromisoformat(estimates[0][key].replace('Z', '+00:00')) for key in ('start_utc', 'end_utc'))
            assert (b - a).total_seconds() / 3600 == expected
            assert a + (b - a) / 2 == start + timedelta(hours=hours / 2)
    actual = [{'id': 'actual', 'start_utc': (start + timedelta(hours=2)).isoformat(), 'end_utc': (start + timedelta(hours=3)).isoformat()}]
    result = check_sleep_deficit(actual, [], flights=[flight], cutoff_utc=(start + timedelta(hours=3)).isoformat(), start_utc=(start + timedelta(hours=3)).isoformat(), end_utc=end.isoformat())
    assert all(not i['evidence']['inflight_sleep_estimates'] for i in result['issues'])

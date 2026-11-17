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


def test_eight_hour_daily_sleep_has_no_default_latency_deficit():
    planned = [record('first', '2027-01-02T15:00:00Z', '2027-01-02T23:00:00Z'),
               record('second', '2027-01-03T15:00:00Z', '2027-01-03T23:00:00Z')]
    result = check_sleep_deficit([], planned, cutoff_utc='2027-01-01T00:00:00Z',
                                 start_utc='2027-01-02T23:00:00Z', end_utc='2027-01-03T23:00:00Z')
    assert result['issues'] == []

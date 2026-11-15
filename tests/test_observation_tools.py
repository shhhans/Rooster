import pytest

from app.planning.observation import observe_event, observe_track


def fixture():
    return {'events': [
        {'id': 'flight', 'kind': 'duty', 'start_utc': '2026-11-12T14:50:00Z',
         'end_utc': '2026-11-13T07:20:00Z', 'timezone': 'Asia/Hong_Kong'},
        {'id': 'light', 'kind': 'light', 'start_utc': '2026-11-12T10:00:00Z',
         'end_utc': '2026-11-12T11:00:00Z', 'timezone': 'Asia/Hong_Kong'},
        {'id': 'coffee', 'kind': 'caffeine', 'start_utc': '2026-11-12T10:30:00Z',
         'timezone': 'Asia/Hong_Kong'},
    ]}, [
        {'id': 's1', 'start_utc': '2026-11-11T15:00:00Z',
         'end_utc': '2026-11-11T23:00:00Z', 'timezone': 'Asia/Hong_Kong', 'role': 'main'},
        {'id': 's2', 'start_utc': '2026-11-11T23:00:00Z',
         'end_utc': '2026-11-12T07:00:00Z', 'timezone': 'Asia/Hong_Kong', 'role': 'main'},
        {'id': 's3', 'start_utc': '2026-11-12T07:00:00Z',
         'end_utc': '2026-11-12T13:15:00Z', 'timezone': 'Asia/Hong_Kong', 'role': 'main'},
    ]


def test_track_includes_boundary_cut_and_contiguous_group():
    snapshot, blocks = fixture()
    result = observe_track(snapshot, blocks, ['activity'], '2026-11-12T00:00',
                           '2026-11-12T22:50', 'Asia/Hong_Kong')
    assert [item['id'] for item in result['events']] == ['s1', 's2', 's3']
    assert result['events'][0]['cut_by_window_start']
    assert result['events'][0]['start_local'] == '2026-11-11T23:00:00+08:00'
    assert result['contiguous_planned_sleep'][0]['minutes'] == 1335
    assert result['activity_overlaps'] == []


def test_event_context_gaps_and_concurrent_tracks():
    snapshot, blocks = fixture()
    result = observe_event(snapshot, blocks, 'flight', 6, 1,
                           ['activity', 'environment', 'intake'], 'Asia/Hong_Kong')
    assert result['nearest_by_track']['activity']['previous'] == [{'id': 's3', 'gap_minutes': 95.0}]
    assert result['nearest_by_track']['environment']['previous'] == [{'id': 'light', 'gap_minutes': 230.0}]
    assert result['nearest_by_track']['intake']['previous'] == [{'id': 'coffee', 'gap_minutes': 260.0}]
    assert result['windows']['before'] == {'start_utc': '2026-11-12T08:50:00Z',
        'end_utc': '2026-11-12T14:50:00Z', 'start_local': '2026-11-12T16:50:00+08:00',
        'end_local': '2026-11-12T22:50:00+08:00'}
    assert result['windows']['after'] == {'start_utc': '2026-11-13T07:20:00Z',
        'end_utc': '2026-11-13T08:20:00Z', 'start_local': '2026-11-13T15:20:00+08:00',
        'end_local': '2026-11-13T16:20:00+08:00'}


def test_half_open_boundaries_for_point_events():
    snapshot, blocks = fixture()
    result = observe_track(snapshot, blocks, ['intake'], '2026-11-12T18:30',
                           '2026-11-12T19:00', 'Asia/Hong_Kong')
    assert [event['id'] for event in result['events']] == ['coffee']
    result = observe_track(snapshot, blocks, ['intake'], '2026-11-12T18:00',
                           '2026-11-12T18:30', 'Asia/Hong_Kong')
    assert result['events'] == []


def test_requested_display_timezone_is_distinct_from_event_timezone():
    snapshot = {'events': [{'id': 'london', 'kind': 'sleep_actual',
                            'start_utc': '2026-11-13T22:00:00Z',
                            'end_utc': '2026-11-14T06:00:00Z',
                            'timezone': 'Europe/London'}]}
    result = observe_track(snapshot, [], ['activity'], '2026-11-13', '2026-11-15',
                           'Asia/Hong_Kong')
    event = result['events'][0]
    assert event['start_local'] == '2026-11-14T06:00:00+08:00'
    assert event['display_timezone'] == 'Asia/Hong_Kong'
    assert event['event_timezone'] == 'Europe/London'


def test_flight_duty_uses_one_display_timezone_and_keeps_destination_metadata():
    snapshot = {'events': [
        {'id': 'duty', 'kind': 'duty', 'start_utc': '2026-11-12T14:50:00Z',
         'end_utc': '2026-11-13T07:20:00Z', 'timezone': 'Asia/Hong_Kong'},
        {'id': 'flight', 'kind': 'flight', 'start_utc': '2026-11-12T16:15:00Z',
         'end_utc': '2026-11-13T06:50:00Z', 'timezone': 'Asia/Hong_Kong',
         'arrival_timezone': 'Europe/London', 'duty_id': 'duty'}]}
    result = observe_track(snapshot, [], ['duty'], '2026-11-12', '2026-11-14',
                           'Asia/Hong_Kong')
    duty = next(event for event in result['events'] if event['id'] == 'duty')
    assert duty['end_local'] == '2026-11-13T15:20:00+08:00'
    assert duty['destination_timezone'] == 'Europe/London'
    assert 'end_at_destination' not in duty


def test_primary_activity_overlap_is_reported_without_changing_records():
    snapshot, blocks = fixture()
    snapshot['events'].append({'id': 'exercise', 'kind': 'exercise',
        'start_utc': '2026-11-12T12:45:00Z', 'end_utc': '2026-11-12T13:45:00Z'})
    result = observe_track(snapshot, blocks, ['activity'], '2026-11-12T19:00',
                           '2026-11-12T22:30', 'Asia/Hong_Kong')
    overlap = result['activity_overlaps'][0]
    assert set(overlap['ids']) == {'s3', 'exercise'}
    assert overlap['minutes'] == 30
    assert blocks[-1]['end_utc'] == '2026-11-12T13:15:00Z'


def test_naive_queries_reject_dst_gap_and_ambiguous_fold():
    with pytest.raises(ValueError, match='nonexistent'):
        observe_track({'events': []}, [], ['activity'],
                      '2026-03-29T01:30', '2026-03-29T03:00', 'Europe/London')
    with pytest.raises(ValueError, match='ambiguous'):
        observe_track({'events': []}, [], ['activity'],
                      '2026-10-25T01:30', '2026-10-25T03:00', 'Europe/London')


def test_right_boundary_and_cross_track_overlap_keep_full_event():
    snapshot, blocks = fixture()
    snapshot['events'].append({'id': 'onboard-light', 'kind': 'light',
        'start_utc': '2026-11-12T15:00:00Z', 'end_utc': '2026-11-12T16:00:00Z'})
    result = observe_event(snapshot, blocks, 'flight', 1, 1,
                           ['duty', 'environment'], 'Asia/Hong_Kong')
    assert result['nearest_by_track']['environment']['overlapping_ids'] == ['onboard-light']
    track = observe_track(snapshot, blocks, ['environment'], '2026-11-12T23:00',
                          '2026-11-12T23:30', 'Asia/Hong_Kong')
    assert track['events'][0]['cut_by_window_end']
    assert track['events'][0]['end_local'] == '2026-11-13T00:00:00+08:00'


def test_time_precision_and_free_time_default():
    snapshot = {'events': [{'id': 'point', 'kind': 'caffeine',
                            'start_utc': '2026-11-12T10:30:27.123456Z',
                            'timezone': 'Asia/Hong_Kong'}]}
    result = observe_track(snapshot, [], ['intake'], '2026-11-12T18:00',
                           '2026-11-12T19:00', 'Asia/Hong_Kong')
    event = result['events'][0]
    assert event['start_utc'] == '2026-11-12T10:30:27.123456Z'
    assert event['start_local'] == '2026-11-12T18:30:27.123456+08:00'
    assert '默认可自由支配' in result['coverage']

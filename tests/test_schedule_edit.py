from copy import deepcopy

import pytest

from app.planning.schedule_edit import edit_schedule
from app.planning.observation import observe_event


def test_batch_edit_returns_cross_track_neighbors_and_does_not_mutate():
    snapshot = {'events': [
        {'id': 'duty', 'kind': 'duty', 'start_utc': '2026-11-13T00:00:00Z',
         'end_utc': '2026-11-13T11:20:00Z', 'timezone': 'Europe/London'},
        {'id': 'light', 'kind': 'light', 'start_utc': '2026-11-13T12:00:00Z',
         'end_utc': '2026-11-13T14:00:00Z', 'timezone': 'Europe/London'}]}
    blocks = [{'id': 'sleep', 'kind': 'sleep_plan', 'start_utc': '2026-11-13T08:35:00Z',
               'end_utc': '2026-11-13T11:00:00Z', 'timezone': 'Europe/London'}]
    old = deepcopy(blocks)
    result = edit_schedule(snapshot, blocks, [
        {'op': 'update', 'event_id': 'sleep', 'fields': {'start': '2026-11-13T12:30', 'end': '2026-11-13T16:00'}},
        {'op': 'create', 'fields': {'kind': 'melatonin', 'start': '2026-11-13T21:00',
                                    'end': '2026-11-13T21:30', 'timezone': 'Europe/London'}}], 'Europe/London')
    assert blocks == old
    assert result['changes'][0]['after']['start_local'] == '2026-11-13T12:30:00+00:00'



def test_conflict_is_returned_for_later_review_without_mutating_input():
    snapshot = {'events': [{'id': 'meal', 'kind': 'meal', 'start_utc': '2026-11-13T13:00:00Z',
                            'end_utc': '2026-11-13T14:00:00Z', 'timezone': 'Europe/London'}]}
    blocks = []
    operations = [{'op': 'create', 'fields': {'kind': 'sleep_plan', 'start': '2026-11-13T12:30',
                   'end': '2026-11-13T16:00', 'timezone': 'Europe/London'}}]
    result = edit_schedule(snapshot, blocks, operations, 'Europe/London')
    assert len(result['blocks']) == 1
    assert blocks == []


def test_updated_draft_is_visible_to_followup_observation():
    blocks = [{'id': 'old', 'start_utc': '2026-11-13T08:00:00Z',
               'end_utc': '2026-11-13T09:00:00Z', 'timezone': 'Europe/London'}]
    result = edit_schedule({'events': []}, blocks, [{'op': 'update', 'event_id': 'old',
        'fields': {'start': '2026-11-13T12:30', 'end': '2026-11-13T16:00'}}], 'Asia/Hong_Kong')
    observed = observe_event({'events': []}, result['blocks'], 'old', 1, 1, ['activity'], 'Asia/Hong_Kong')
    assert observed['event']['start_local'] == '2026-11-13T20:30:00+08:00'
    assert observed['event']['start_utc'] == '2026-11-13T12:30:00Z'
    assert blocks[0]['start_utc'] == '2026-11-13T08:00:00Z'

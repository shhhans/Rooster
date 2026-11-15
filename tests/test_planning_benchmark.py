from copy import deepcopy

from app.planning.engine import available_windows, check_plan, plan, randomized_strategy, score_timeline


def snapshot(now="2027-04-03T06:00:00Z", horizon="2027-04-06T06:00:00Z", events=None, **extra):
    events = list(events or [])
    events.append({"id": "confirmed-home", "kind": "environment", "start_utc": now,
                   "end_utc": horizon, "timezone": "Asia/Tokyo", "location": "home",
                   "availability": "available", "confirmed": True})
    value = {"now_utc": now, "horizon_end_utc": horizon, "revision": 12,
             "profile": {"home_timezone": "Asia/Tokyo", "sleep_need_hours": 8,
                         "preferred_sleep_start_hour": 22},
             "initial_state": {}, "events": events}
    value.update(extra)
    return value


def test_generalized_planning_uses_arbitrary_dates_and_local_timezone():
    data = snapshot(events=[{"id": "duty-1", "kind": "duty", "start_utc": "2027-04-04T01:00:00Z",
                             "end_utc": "2027-04-04T05:00:00Z", "timezone": "America/Los_Angeles",
                             "flight_ids": ["f-1"]}])
    result = plan(data)
    assert result["status"] == "ready"
    assert result["selected"]["sleep_blocks"]
    assert result["selected"]["sleep_blocks"][0]["timezone"] == "Asia/Tokyo"
    assert result["model_version"]
    assert result["policy_version"]
    assert all(c["status"] != "fail" for c in result["checks"])


def test_actual_history_is_counted_but_future_actual_is_not_added():
    data = snapshot(events=[
        {"id": "past", "kind": "sleep_actual", "start_utc": "2027-04-02T21:00:00Z",
         "end_utc": "2027-04-03T05:00:00Z", "timezone": "Asia/Tokyo"},
        {"id": "bad-future", "kind": "sleep_actual", "start_utc": "2027-04-05T01:00:00Z",
         "end_utc": "2027-04-05T08:00:00Z", "timezone": "Asia/Tokyo"},
    ])
    before = deepcopy(data)
    # Historical v1 keeps its original "ignore future actual" behavior; v2 rejects it.
    metrics = score_timeline(data, model_version='sleep-s-c-engineering-v1')
    assert metrics["rolling_sleep_hours_24"] == 8
    assert metrics["rolling_sleep_hours_48"] == 8
    assert data == before
    invalid = score_timeline(data)
    assert invalid['status'] == 'invalid_input'
    assert invalid['utility'] is None


def test_duty_buffers_and_locked_commitments_block_sleep():
    data = snapshot(events=[
        {"id": "d", "kind": "duty", "start_utc": "2027-04-03T12:00:00Z", "end_utc": "2027-04-03T16:00:00Z"},
        {"id": "c", "kind": "commitment", "locked": True,
         "start_utc": "2027-04-04T12:00:00Z", "end_utc": "2027-04-04T13:00:00Z"},
    ])
    windows = available_windows(data)
    assert not any(w["start_utc"] < "2027-04-03T10:30:00Z" < w["end_utc"] for w in windows)
    assert not any(w["start_utc"] < "2027-04-03T17:15:00Z" < w["end_utc"] for w in windows)
    sleep = [{"id": "b", "start_utc": "2027-04-03T11:00:00Z", "end_utc": "2027-04-03T19:00:00Z",
              "timezone": "UTC", "role": "main"}]
    assert any(c["id"] == "sleep_event_overlap" and c["status"] == "fail" for c in check_plan(data, sleep))


def test_protected_rest_requires_uninterrupted_availability_but_allows_sleep():
    rest = {"id": "rest", "kind": "protected_rest", "start_utc": "2027-04-03T08:00:00Z",
            "end_utc": "2027-04-03T20:00:00Z", "minimum_hours": 8, "source": "synthetic"}
    ok = snapshot(events=[rest])
    block = [{"id": "sleep", "start_utc": "2027-04-03T09:00:00Z", "end_utc": "2027-04-03T17:00:00Z",
              "timezone": "UTC", "role": "main"}]
    assert not any(c["id"] == "protected_rest_preserved" and c["status"] == "fail" for c in check_plan(ok, block))
    broken = snapshot(events=[rest, {"id": "duty", "kind": "duty", "start_utc": "2027-04-03T09:00:00Z",
                                    "end_utc": "2027-04-03T13:00:00Z"},
                              {"id": "c", "kind": "commitment", "locked": True,
                               "start_utc": "2027-04-03T14:00:00Z", "end_utc": "2027-04-03T18:00:00Z"}])
    checks = check_plan(broken, [])
    assert any(c["id"] == "protected_rest_preserved" and c["status"] == "fail" for c in checks)
    assert any(c["id"] == "aviation_legal_compliance" and c["status"] == "not_evaluated" for c in checks)


def test_impossible_short_horizon_and_seeded_strategy_reproducibility():
    tiny = snapshot(now="2027-04-03T06:00:00Z", horizon="2027-04-03T07:00:00Z")
    # Search now supports short opportunities rather than imposing a 2h main minimum.
    short = plan(tiny)
    assert short['status'] == 'ready'
    assert all(c['status'] != 'fail' for c in short['checks'])
    wide = snapshot()
    assert randomized_strategy(wide, 71) == randomized_strategy(wide, 71)
    assert randomized_strategy(wide, 71)["status"] == "ready"


def test_confirmed_aircraft_rest_is_the_only_flight_sleep_exception():
    data = snapshot(now="2027-04-03T00:00:00Z", horizon="2027-04-04T00:00:00Z", events=[
        {"id": "duty", "kind": "duty", "start_utc": "2027-04-03T01:00:00Z", "end_utc": "2027-04-03T10:00:00Z"},
        {"id": "flight", "kind": "flight", "start_utc": "2027-04-03T02:00:00Z", "end_utc": "2027-04-03T09:00:00Z"},
        {"id": "aircraft-rest", "kind": "environment", "start_utc": "2027-04-03T03:00:00Z",
         "end_utc": "2027-04-03T07:00:00Z", "location": "aircraft", "confirmed": True},
    ])
    block = [{"id": "air-sleep", "start_utc": "2027-04-03T03:30:00Z", "end_utc": "2027-04-03T05:30:00Z",
              "timezone": "UTC", "role": "main"}]
    checks = check_plan(data, block)
    assert any(c['id'] == 'sleep_event_overlap' for c in checks)
    data['events'].append({'id': 'rest-authority', 'kind': 'crew_rest',
        'start_utc': '2027-04-03T03:00:00Z', 'end_utc': '2027-04-03T07:00:00Z',
        'authorized': True, 'authorized_event_ids': ['duty', 'flight']})
    block[0]['crew_rest_id'] = 'rest-authority'
    checks = check_plan(data, block)
    assert not any(c["status"] == "fail" for c in checks)
    block[0]['end_utc'] = '2027-04-03T07:30:00Z'
    assert any(c['id'] == 'crew_rest_authorization' for c in check_plan(data, block))
    block[0]['end_utc'] = '2027-04-03T05:30:00Z'
    data['events'][-1]['authorized_event_ids'] = ['duty']
    assert any(c.get('event_id') == 'flight' and c['status'] == 'fail' for c in check_plan(data, block))
    data['events'][-1]['authorized_event_ids'] = ['duty', 'flight']
    # Aircraft rest can authorize rest within flight, but it never makes two duty records compatible.
    data["events"].append({"id": "overlapping-duty", "kind": "duty", "start_utc": "2027-04-03T04:00:00Z",
                           "end_utc": "2027-04-03T08:00:00Z"})
    checks = check_plan(data, block)
    assert any(c["id"] == "duty_overlap" and c["status"] == "fail" for c in checks)


def test_sleep_without_confirmed_environment_uses_known_free_time():
    data = snapshot(events=[])
    data["events"] = [e for e in data["events"] if e["kind"] != "environment"]
    data["events"].append({"id": "conditional-room", "kind": "environment", "start_utc": data["now_utc"],
                            "end_utc": data["horizon_end_utc"], "location": "hotel",
                            "availability": "conditional", "confirmed": True})
    assert available_windows(data)
    block = [{"id": "unknown-place", "start_utc": "2027-04-03T10:00:00Z", "end_utc": "2027-04-03T18:00:00Z",
              "timezone": "UTC", "role": "main"}]
    assert not any(c["status"] == "fail" for c in check_plan(data, block))


def test_continuous_wake_joins_history_to_first_planned_sleep():
    now = "2027-04-03T06:00:00Z"
    data = snapshot(now=now, horizon="2027-04-05T06:00:00Z")
    data["initial_state"] = {"last_wake_utc": "2027-04-02T20:00:00Z"}
    data['scheduling_policy'] = {'maximum_wake_hours':20}
    block = [{"id": "late", "start_utc": "2027-04-03T21:00:00Z", "end_utc": "2027-04-04T05:00:00Z",
              "timezone": "UTC", "role": "main"}]
    assert any(c["id"] == "maximum_continuous_wake" and c["status"] == "fail" for c in check_plan(data, block))


def test_replan_can_recover_after_historical_wake_limit_was_exceeded():
    data = snapshot(now="2027-04-03T06:00:00Z", horizon="2027-04-04T02:00:00Z")
    data["initial_state"] = {"last_wake_utc": "2027-04-02T05:00:00Z"}
    block = [{"id": "recovery", "start_utc": "2027-04-03T07:00:00Z", "end_utc": "2027-04-03T15:00:00Z",
              "timezone": "UTC", "role": "main"}]
    checks = check_plan(data, block)
    assert not any(c["id"] == "maximum_continuous_wake" and c["status"] == "fail" for c in checks)

"""Behavioral contract tests for the sleep-model v2 design.

The A-number comments map each test to the accepted decisions in
docs/sleep-model-design-decisions.md.  Tests use only synthetic UTC facts.
"""
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from app.sleep_model import circadian, homeostat


UTC = timezone.utc
NOW = datetime(2027, 1, 5, 12, tzinfo=UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def synthetic_snapshot(*, duty=True):
    """Three complete historical nights, a confirmed home, and an optional duty."""
    events = []
    for days_ago in (3, 2, 1):
        wake = NOW - timedelta(days=days_ago) + timedelta(hours=19)
        events.append({
            "id": f"sleep-{days_ago}", "kind": "sleep_actual",
            "start_utc": iso(wake - timedelta(hours=8)), "end_utc": iso(wake),
            "source": "synthetic_observation", "episode_id": f"episode-{days_ago}",
        })
    events.append({
        "id": "home", "kind": "environment", "location": "home",
        "availability": "available", "confirmed": True, "timezone": "UTC",
        "start_utc": iso(NOW - timedelta(days=4)),
        "end_utc": iso(NOW + timedelta(days=2)),
    })
    if duty:
        events.append({
            "id": "duty-1", "kind": "duty", "start_utc": iso(NOW + timedelta(hours=2)),
            "end_utc": iso(NOW + timedelta(hours=6)), "timezone": "UTC",
        })
    return {
        "snapshot_id": "synthetic-sleep-v2", "revision": 1,
        "now_utc": iso(NOW), "horizon_end_utc": iso(NOW + timedelta(days=2)),
        "profile": {"home_timezone": "UTC", "habit_bed_local": "23:00",
                    "habit_wake_local": "07:00", "sleep_need_hours": 8},
        "initial_state": {}, "events": events,
    }


def opportunity(start, end, *, role="main", identifier="sleep-next"):
    return {"id": identifier, "start_utc": iso(start), "end_utc": iso(end),
            "timezone": "UTC", "environment_id": "home", "role": role}


def evaluate(snapshot, sleep_blocks=None, **kwargs):
    # Import lazily so the pure state tests remain usable during implementation.
    from app.sleep_model import evaluate as evaluate_model
    return evaluate_model(snapshot, sleep_blocks, scenarios=False, **kwargs)


# A1, A15: normalized Process S follows the exact exponential wake/sleep updates.
def test_process_s_uses_exact_hour_scale_updates():
    assert homeostat.advance(.5, 18.2, asleep=False) == pytest.approx(1 - .5 / math.e, abs=1e-12)
    assert homeostat.advance(.5, 4.2, asleep=True) == pytest.approx(.5 / math.e, abs=1e-12)


# A1, A2, A5, A6, A7: Forger99 exposes the coupled oscillator RHS with continuous state.
def test_forger_rhs_is_finite_and_zero_light_has_no_photic_drive():
    state = np.array([.2, -.4, .3])
    actual = circadian.rhs(0.0, state, 0.0)
    expected = np.array([
        np.pi / 12 * state[1],
        np.pi / 12 * (.23 * (state[1] - 4 / 3 * state[1] ** 3)
                      - state[0] * (24 / (.99669 * 24.2)) ** 2),
        60 * (-.0075 * state[2]),
    ])
    assert actual.shape == (3,)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


# A20, A22, A25, A55, A56, A58, A59, A60, A63: state and prediction are explicit outputs.
def test_evaluation_returns_trace_predictions_and_provenance():
    snap = synthetic_snapshot()
    block = opportunity(NOW + timedelta(hours=10), NOW + timedelta(hours=19))
    result = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)))
    assert result["status"] in {"complete", "partial"}
    for key in ("utility", "losses", "physiological_metrics", "preference_metrics",
                "trace", "predicted_sleep", "assumptions", "constraint_report",
                "publishability", "provenance", "scenarios"):
        assert key in result
    assert result["trace"]
    for point in result["trace"]:
        assert {"time_utc", "S", "C", "inertia", "burden", "asleep", "x", "xc", "n"} <= point.keys()
    assert result["predicted_sleep"]
    assert all("predicted" in str(item).lower() for item in result["predicted_sleep"])
    assert result["provenance"]


# A25, A38, A58: an opportunity is a soft forecast with the declared 15-minute latency.
def test_sleep_opportunity_is_not_counted_as_immediate_or_observed_sleep():
    snap = synthetic_snapshot(duty=False)
    block = opportunity(NOW + timedelta(hours=8), NOW + timedelta(hours=16))
    result = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)))
    predicted = result["predicted_sleep"]
    assert len(predicted) == 1
    assert predicted[0]["start_utc"] == iso(NOW + timedelta(hours=8, minutes=15))
    assert predicted[0]["end_utc"] == block["end_utc"]
    assert predicted[0].get("source") != "observed"


# A20, A43, A51, A54: naps affect state consistently and wake inertia remains explicit.
def test_nap_contributes_predicted_sleep_without_role_specific_scoring():
    snap = synthetic_snapshot(duty=False)
    nap = opportunity(NOW + timedelta(hours=3), NOW + timedelta(hours=5), role="nap", identifier="nap-1")
    result = evaluate(snap, [nap], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=8)))
    assert result["predicted_sleep"]
    assert result["predicted_sleep"][0].get("role") == "nap"
    assert any(float(point["inertia"]) > 0 for point in result["trace"]
               if point["time_utc"] > result["predicted_sleep"][0]["end_utc"])


# A33, A45: hard availability conflicts cannot be compensated by a high score.
def test_sleep_overlapping_duty_is_not_publishable():
    snap = synthetic_snapshot(duty=True)
    conflicting = opportunity(NOW + timedelta(hours=3), NOW + timedelta(hours=8), identifier="overlap")
    result = evaluate(snap, [conflicting], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)))
    assert result["publishability"] == "invalid"
    assert result["constraint_report"]


# A38, A46: structured preferences are reported separately and shortfall stays soft.
def test_short_continuous_sleep_preference_reports_deviation_without_rejecting_simulation():
    snap = synthetic_snapshot(duty=False)
    block = opportunity(NOW + timedelta(hours=8), NOW + timedelta(hours=14))
    preference = {"id": "continuous-8h", "kind": "continuous_sleep", "target": 8,
                  "scope_start_utc": iso(NOW), "scope_end_utc": iso(NOW + timedelta(days=1)), "weight": 1}
    result = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)),
                      preferences=[preference])
    assert result["status"] in {"complete", "partial"}
    assert result["publishability"] != "invalid"
    assert result["preference_metrics"]


# A25, A55, A59, A60: missing duty leaves task-duty burden unknown rather than zero.
def test_no_duty_keeps_duty_burden_unknown():
    snap = synthetic_snapshot(duty=False)
    result = evaluate(snap, [], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=12)))
    metrics = result["physiological_metrics"]
    assert metrics.get("duty_burden") is None


# A2, A25, A63: future candidate changes cannot rewrite the committed now state.
def test_changing_future_opportunity_preserves_now_state_and_is_deterministic():
    snap = synthetic_snapshot(duty=False)
    first = evaluate(snap, [], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)))
    block = opportunity(NOW + timedelta(hours=10), NOW + timedelta(hours=18))
    second = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)))
    repeated = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(days=1)))
    initial_a = min(first["trace"], key=lambda p: abs((datetime.fromisoformat(p["time_utc"].replace("Z", "+00:00")) - NOW).total_seconds()))
    initial_b = min(second["trace"], key=lambda p: abs((datetime.fromisoformat(p["time_utc"].replace("Z", "+00:00")) - NOW).total_seconds()))
    for name in ("S", "x", "xc", "n", "inertia"):
        assert initial_a[name] == pytest.approx(initial_b[name], abs=1e-10)
    assert second == repeated


# A44: three normal nights with separate episode IDs are not mislabeled as fragmentation.
def test_separate_nightly_main_sleeps_do_not_become_fragmentation_count():
    result = evaluate(synthetic_snapshot(duty=False), [], start_utc=iso(NOW),
                      end_utc=iso(NOW + timedelta(days=1)))
    metrics = result["physiological_metrics"]
    assert metrics.get("fragmentation") in (None, 0, 0.0)


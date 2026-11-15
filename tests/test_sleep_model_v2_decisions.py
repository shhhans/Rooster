"""Decision-level behavioral coverage for accepted sleep-model decisions.

Each test names the A-number(s) it exercises.  Facts and histories are synthetic;
the assertions verify the declared computation, not individual accuracy.
"""
from datetime import timedelta
import math
import re

import pytest

from app.sleep_model import evaluate as evaluate_model
from app.sleep_model.contracts import configuration
from app.sleep_model.metrics import measure, penalty
from tests.test_sleep_model_v2 import NOW, evaluate, iso, opportunity, synthetic_snapshot


def test_process_s_state_is_bounded_and_piecewise_sleep_reduces_it():
    """A1, A15: actual segments update continuous Process S without resetting it."""
    snap = synthetic_snapshot(duty=False)
    block = opportunity(NOW + timedelta(hours=1), NOW + timedelta(hours=5))
    result = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=8)))
    before = next(p for p in result["trace"] if p["time_utc"] == block["start_utc"])
    asleep = next(p for p in result["trace"] if p["time_utc"] == result["predicted_sleep"][0]["start_utc"])
    after = next(p for p in result["trace"] if p["time_utc"] == result["predicted_sleep"][0]["end_utc"])
    assert 0 <= before["S"] <= 1
    assert after["S"] < asleep["S"]
    assert all(0 <= p["S"] <= 1 for p in result["trace"])


def test_circadian_state_moves_continuously_and_emits_cbtmin_dlmo_proxy():
    """A2, A5, A6: phase markers are model markers with a fixed seven-hour proxy gap."""
    result = evaluate(synthetic_snapshot(duty=False), [], start_utc=iso(NOW),
                      end_utc=iso(NOW + timedelta(days=2)))
    markers = result["physiological_metrics"]["phase_markers"]
    assert markers
    for marker in markers:
        from datetime import datetime
        cbt = datetime.fromisoformat(marker["cbtmin_utc"].replace("Z", "+00:00"))
        dlmo = datetime.fromisoformat(marker["dlmo_proxy_utc"].replace("Z", "+00:00"))
        assert (cbt - dlmo).total_seconds() == pytest.approx(7 * 3600)
        assert marker["method"]


def test_observed_light_changes_future_circadian_state():
    """A7: photopic light is a circadian input, not just descriptive metadata."""
    base = synthetic_snapshot(duty=False)
    start, end = NOW, NOW + timedelta(hours=12)
    dim = dict(base, events=base["events"] + [{
        "id": "light", "kind": "light", "metric": "photopic_lux", "lux": 0,
        "start_utc": iso(start), "end_utc": iso(end),
    }])
    bright = dict(base, events=base["events"] + [{
        "id": "light", "kind": "light", "metric": "photopic_lux", "lux": 1000,
        "start_utc": iso(start), "end_utc": iso(end),
    }])
    a = evaluate(dim, [], start_utc=iso(start), end_utc=iso(end))
    b = evaluate(bright, [], start_utc=iso(start), end_utc=iso(end))
    assert a["status"] in {"complete", "partial"} and b["status"] in {"complete", "partial"}
    assert abs(a["final_state"]["n"] - b["final_state"]["n"]) > 1e-5
    assert abs(a["final_state"]["x"] - b["final_state"]["x"]) > 1e-7


def test_retrospective_observed_sleep_is_counted_as_it_happened_without_latency():
    """A25: retrospective actual sleep covers its observed interval, without forecast latency."""
    snap = synthetic_snapshot(duty=False)
    start, end = NOW - timedelta(hours=12), NOW
    snap["events"] = [e for e in snap["events"] if e["kind"] != "sleep_actual"]
    actual = {"id": "actual-window", "kind": "sleep_actual",
              "start_utc": iso(NOW - timedelta(hours=10)),
              "end_utc": iso(NOW - timedelta(hours=4)), "episode_id": "actual-episode"}
    snap["events"].append(actual)
    result = evaluate_model(snap, [], start_utc=iso(start), end_utc=iso(end), mode="retrospective",
                            scenarios=False)
    assert result["status"] in {"complete", "partial"}
    assert result["predicted_sleep"] == []
    assert result["physiological_metrics"]["actual_sleep_hours"] == pytest.approx(6)


def test_key_task_markers_are_sampled_and_wake_to_task_margin_is_reported():
    """A22, A43, A54: task state and time from predicted wake remain inspectable."""
    snap = synthetic_snapshot(duty=True)
    wake = NOW + timedelta(hours=8)
    task = wake + timedelta(minutes=45)
    snap["task_markers"] = [{"id": "report", "time_utc": iso(task), "kind": "report", "source": "synthetic"}]
    block = opportunity(NOW + timedelta(hours=1), wake, identifier="main")
    result = evaluate(snap, [block], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=12)))
    metrics = result["physiological_metrics"]
    assert metrics["task_marker_coverage"] == 1
    assert metrics["task_markers"][0]["burden"] is not None
    margin = next(x for x in metrics["wake_task_margins"] if x["wake_utc"] == result["predicted_sleep"][0]["end_utc"])
    assert margin["next_task_minutes"] == pytest.approx(45)


def test_softplus_penalty_is_smooth_monotone_and_keeps_low_and_peak_metrics():
    """A40, A41: threshold penalty is smooth; summaries preserve mean, peak and area."""
    cfg = configuration()
    samples = [penalty(v, cfg) for v in (0.0, 0.59, 0.60, 0.61, 1.0)]
    assert all(0 <= value <= 1 for value in samples)
    assert samples == sorted(samples)
    # A softplus has a positive, smooth slope on both sides of the threshold.
    left_slope = (samples[2] - samples[1]) / .01
    right_slope = (samples[3] - samples[2]) / .01
    assert left_slope > 0 and right_slope > 0
    snap = synthetic_snapshot(duty=True)
    result = evaluate(snap, [], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=8)))
    metrics = result["physiological_metrics"]
    assert metrics["mean_burden"] is not None
    assert metrics["peak_burden"] >= metrics["mean_burden"]
    # Accepted A41 also calls for the low end of the burden curve; this exposes
    # a missing metric in the current implementation if it is not returned.
    assert metrics["minimum_burden"] is not None
    assert metrics["minimum_burden"] <= metrics["mean_burden"]
    assert metrics["above_threshold_hours"] is not None
    assert metrics["excess_area"] is not None


def test_conditional_plan_requires_authorization_and_short_sleep_remains_publishable():
    """A38, A45: short goals remain soft while proposed block changes stay conditional."""
    snap = synthetic_snapshot(duty=False)
    short = opportunity(NOW + timedelta(hours=2), NOW + timedelta(hours=8), identifier="short")
    short_result = evaluate(snap, [short], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=12)))
    assert short_result["publishability"] == "valid"
    assert (short_result["predicted_sleep"][0]["end_utc"] != short_result["predicted_sleep"][0]["start_utc"])
    conditional = dict(short, id="move-event", conditional=True)
    conditional_result = evaluate(snap, [conditional], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=12)))
    assert conditional_result["publishability"] == "conditional"


def test_law_compliance_is_not_evaluated_and_cannot_change_physiological_state():
    """A36: legal classification stays separate from physiological state evolution."""
    a = synthetic_snapshot(duty=False)
    b = synthetic_snapshot(duty=False)
    a["regulatory_adaptation"] = "adapted"
    b["regulatory_adaptation"] = "not_adapted"
    left = evaluate(a, [], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=6)))
    right = evaluate(b, [], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=6)))
    legal = next(row for row in left["constraint_report"] if row["id"] == "aviation_legal_compliance")
    assert legal["status"] == "not_evaluated"
    for key in ("S", "x", "xc", "n", "inertia"):
        assert left["final_state"][key] == pytest.approx(right["final_state"][key], abs=1e-12)


def test_no_history_degrades_with_zero_observation_coverage_and_assumptions():
    """A55, A56, A59: no wearable or history is runnable but explicitly assumed."""
    snap = synthetic_snapshot(duty=False)
    snap["events"] = [e for e in snap["events"] if e["kind"] != "sleep_actual"]
    result = evaluate(snap, [], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=6)))
    assert result["status"] in {"complete", "partial"}
    assert result["trace"]
    assert result["physiological_metrics"]["history_observation_coverage"] == 0
    assert any("prior" in str(value).lower() or "unobserved" in str(value).lower()
               for value in result["assumptions"])


def test_missing_task_duty_is_unknown_and_constraint_status_does_not_fake_legal_pass():
    """A33, A55, A60: missing domains remain null and legal status remains explicit."""
    result = evaluate(synthetic_snapshot(duty=False), [], start_utc=iso(NOW),
                      end_utc=iso(NOW + timedelta(hours=6)))
    assert result["physiological_metrics"]["duty_burden"] is None
    assert next(x for x in result["constraint_report"] if x["id"] == "aviation_legal_compliance")["status"] == "not_evaluated"


def test_version_units_and_hashes_are_explicit_and_repeatable():
    """A46, A60, A63: normalized losses, version and content hashes are auditable."""
    snap = synthetic_snapshot(duty=False)
    args = (snap, [],)
    kwargs = {"start_utc": iso(NOW), "end_utc": iso(NOW + timedelta(hours=6))}
    one = evaluate_model(*args, scenarios=False, **kwargs)
    two = evaluate_model(*args, scenarios=False, **kwargs)
    assert one["scorer_version"] == one["provenance"]["model_version"]
    assert one["losses"]["total"] == pytest.approx(1 - one["utility"] / 100)
    assert one["preference_metrics"]["loss"] >= 0
    assert 0 <= one["physiological_metrics"]["history_observation_coverage"] <= 1
    assert one["physiological_metrics"]["duty_hours"] == 0
    assert one["provenance"]["configuration_hash"] == two["provenance"]["configuration_hash"]
    assert one["provenance"]["trace_hash"] == two["provenance"]["trace_hash"]
    assert re.fullmatch(r"[0-9a-f]{64}", one["provenance"]["input_hash"])
    assert "configuration" in one["provenance"]
    assert one["provenance"]["equation_source"]


def test_nap_changes_later_process_s_but_is_not_given_special_fixed_reward():
    """A20, A51: a nap changes downstream S through the same sleep update as main sleep."""
    snap = synthetic_snapshot(duty=False)
    start, end = NOW + timedelta(hours=1), NOW + timedelta(hours=3)
    nap = opportunity(start, end, role="nap", identifier="nap")
    main = opportunity(start, end, role="main", identifier="main")
    a = evaluate(snap, [nap], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=6)))
    b = evaluate(snap, [main], start_utc=iso(NOW), end_utc=iso(NOW + timedelta(hours=6)))
    after = iso(end)
    pa = next(p for p in a["trace"] if p["time_utc"] == after)
    pb = next(p for p in b["trace"] if p["time_utc"] == after)
    assert pa["S"] == pytest.approx(pb["S"], abs=1e-10)
    assert a["losses"]["physiological"] == pytest.approx(b["losses"]["physiological"], abs=1e-10)


def test_fragmentation_counts_wake_gaps_only_within_one_episode():
    """A44: separate normal nights are not interruptions; a split episode is."""
    snap = synthetic_snapshot(duty=False)
    start, end = NOW - timedelta(days=3), NOW
    sleeps = [
        {"id": "part-1", "episode_id": "same-episode", "start_utc": iso(start),
         "end_utc": iso(start + timedelta(hours=3))},
        {"id": "part-2", "episode_id": "same-episode", "start_utc": iso(start + timedelta(hours=3, minutes=30)),
         "end_utc": iso(start + timedelta(hours=7))},
        {"id": "night-two", "episode_id": "other-episode", "start_utc": iso(start + timedelta(days=1)),
         "end_utc": iso(start + timedelta(days=1, hours=8))},
    ]
    # Directly exercise the metrics contract with trace rows across the metric window.
    trace = [{"time_utc": iso(start), "S": .5, "C": .5, "inertia": 0., "burden": .5,
              "asleep": False, "light_source": "assumed", "x": .5, "xc": .5, "n": .5},
             {"time_utc": iso(end), "S": .6, "C": .5, "inertia": 0., "burden": .6,
              "asleep": False, "light_source": "assumed", "x": .5, "xc": .5, "n": .5}]
    metrics, _ = measure(trace, {"events": [], "task_markers": [], "initial_state": {}},
                         sleeps, start, end, configuration(), {"S": .5})
    assert metrics["fragmentation_interruptions"] == 1


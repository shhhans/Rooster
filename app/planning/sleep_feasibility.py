"""Deterministic issue report for a complete planned sleep timeline."""
from math import isfinite
import hashlib
import json
from pathlib import Path

from app.contracts.time import dt, stamp
from app.sleep_model.api import advance_pressure
from app.sleep_model.contracts import VERSION as MODEL_VERSION, physiological_configuration

PRESSURE_THRESHOLDS = {'onset_threshold': .4, 'maintenance_threshold': .05}
CALIBRATION_DIR = Path(__file__).resolve().parents[2] / 'fixtures' / 'sleep-feasibility-review'


def pressure_thresholds(config):
    """Use frozen thresholds only after a matching holdout report passes."""
    try:
        frozen_raw = (CALIBRATION_DIR / 'frozen-calibration.json').read_bytes()
        frozen = json.loads(frozen_raw)
        report = json.loads((CALIBRATION_DIR / 'holdout-report.json').read_bytes())
        cases = (CALIBRATION_DIR / 'cases.jsonl').read_bytes()
        onset, maintenance = frozen['onset_threshold'], frozen['maintenance_threshold']
        if (report['matches_all'] is True and report['coverage_complete'] is True and report['observations']
                and all(row.get('match') is True for row in report['observations'])
                and report['frozen_sha256'] == hashlib.sha256(frozen_raw).hexdigest()
                and frozen['cases_sha256'] == hashlib.sha256(cases).hexdigest()
                and frozen['model_version'] == MODEL_VERSION
                and all(config[key] == frozen['pressure_config'][key] for key in ('tau_w', 'tau_s'))
                and isinstance(onset, (int, float)) and isinstance(maintenance, (int, float))
                and 0 <= maintenance < onset <= 1):
            return {'onset_threshold': onset, 'maintenance_threshold': maintenance}, 'human_calibrated'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return dict(PRESSURE_THRESHOLDS), 'v1'


def check_sleep_feasibility(proposals, *, anchor_utc, initial_s,
                            onset_threshold=None, maintenance_threshold=None, pressure_config=None):
    """Evaluate ordered proposals from a fixed, awake pressure anchor.

    Each proposal contains id, start_utc and end_utc. Planned sleep is
    assumed to occur in full, including when it produces an issue.
    """
    if onset_threshold is None and maintenance_threshold is None:
        thresholds, threshold_status = pressure_thresholds(physiological_configuration(pressure_config))
        onset_threshold = thresholds['onset_threshold']
        maintenance_threshold = thresholds['maintenance_threshold']
    else:
        threshold_status = 'provided'
    values = (initial_s, onset_threshold, maintenance_threshold)
    if any(not isinstance(v, (float, int)) or not isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('pressure and thresholds must be finite values in [0, 1]')
    if onset_threshold <= maintenance_threshold:
        raise ValueError('onset threshold must exceed maintenance threshold')

    cursor = dt(anchor_utc)
    pressure = initial_s
    previous = None
    seen = set()
    issues = []
    checks = []
    for proposal in proposals:
        identifier = proposal['id']
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('proposal IDs must be unique nonempty strings')
        seen.add(identifier)
        start, end = dt(proposal['start_utc']), dt(proposal['end_utc'])
        if start < cursor or end <= start:
            raise ValueError('proposals must be ordered, non-overlapping and after anchor')
        gap = (start - cursor).total_seconds() / 3600
        pressure = advance_pressure(pressure, gap, asleep=False, config=pressure_config)
        context = {'proposal_id': identifier}
        if previous is not None:
            context.update(previous_proposal_id=previous, awake_gap_hours=gap)
        onset = dict(context, type='sleep_onset_pressure_low', time_utc=stamp(start),
                     pressure=pressure, threshold=onset_threshold)
        checks.append(dict(onset, status='fail' if pressure < onset_threshold else 'pass'))
        if pressure < onset_threshold:
            issues.append(onset)
        pressure = advance_pressure(pressure, (end - start).total_seconds() / 3600, asleep=True, config=pressure_config)
        maintenance = dict(context, type='sleep_maintenance_pressure_low', time_utc=stamp(end),
                           pressure=pressure, threshold=maintenance_threshold)
        checks.append(dict(maintenance, status='fail' if pressure < maintenance_threshold else 'pass'))
        if pressure < maintenance_threshold:
            issues.append(maintenance)
        previous, cursor = identifier, end
    return {'issues': issues, 'checks': checks, 'threshold_status': threshold_status,
            'model_version': MODEL_VERSION, 'onset_threshold': onset_threshold,
            'maintenance_threshold': maintenance_threshold}

"""Phase-2 joint validation — MELATONIN-EVALUATION-01.

Validates the Lead-bound joint candidate (sleep-model's opt-in Breslow13
melatonin component + planning's decoupled consumption) against the
independent Phase-1 reference and the frozen decoupling baseline.

Identity discipline (M1/M3/M6 + Lead message 309):

* The Lead-bound manifest is the only source of the candidate path; the
  authoritative identity is the persisted ``task.verification_snapshot``
  capture, and the harness hard-fails on any mismatch (worktree, commit, tree).
* The candidate worktree may not be the evaluation worktree or live inside it.
* The candidate source is loaded from its absolute worktree path into this test
  process (never copied into the evaluation worktree, never shadowed by it);
  ``app*``/``rooster_bench*`` are purged from ``sys.modules`` before the first
  candidate import, and per-module ``__file__`` binding assertions hold at the
  moment the comparisons run.
* ``tests/test_decoupling_baseline.py`` and the frozen fixture are loaded from
  the evaluation worktree by absolute path (never through ``sys.path`` search,
  because the candidate ships its own ``tests`` package).
* The Phase-1 reference module is loaded from its absolute path strictly as the
  independent comparison side — it is never treated as (or named like) the
  candidate, and the joint module contains no import of it (AST-checked).

Case list (acceptance conditions 3–5): J1 Process L semantics; J2 piecewise
A(phi) with the Lead-corrected boundary/jump semantics and periodicity;
J3 reachable suppression-guard behaviour; J4 analytic clearance with a=0 through
the declared experiment channel; J5 nonnegativity/validity shape and total
integration failure; J6 units/UTC/grid/input validation; J7 checkpoint
continuity and identity rejection; J8 oral-zero and dose rejection; J9
opt-in/default-off against the frozen baseline; J10 planner diagnostic enabled;
J11 planner diagnostic disabled; J12 no duplicated physiology or policy matrix;
J13 no saliva, no DLMO, no quality claim.

Run (conda ``205agent``, repo root as cwd)::

    $env:MELATONIN_CANDIDATE_MANIFEST='<Lead manifest path>'
    $env:MELATONIN_CANDIDATE_SNAPSHOT='<persisted snapshot capture>'
    pytest tests/test_melatonin_joint.py -v
    python -m tests.test_melatonin_joint --verify-identity
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_WORKTREE = REPO_ROOT
FIXTURE_DIR = REPO_ROOT / 'tests' / 'fixtures' / 'melatonin'
DEFAULT_MANIFEST_PATH = FIXTURE_DIR / 'candidate-manifest.json'
DEFAULT_SNAPSHOT_PATH = FIXTURE_DIR / 'candidate-verification-snapshot.json'
MANIFEST_ENV_VAR = 'MELATONIN_CANDIDATE_MANIFEST'
SNAPSHOT_ENV_VAR = 'MELATONIN_CANDIDATE_SNAPSHOT'
MANIFEST_CLI_OPTION = '--candidate-manifest'
SNAPSHOT_CLI_OPTION = '--candidate-snapshot'
NO_MANIFEST_REASON = (
    'no Lead-bound candidate manifest: Lead has not late-bound the joint candidate '
    'commit/tree yet (Lead message 309), so the Phase-2 joint assertions cannot run '
    '(Phase 1 hold; see groups/evaluation/tasks/MELATONIN-EVALUATION-01/implementation.md)')

CANONICAL_FIELDS = ('worktree', 'snapshot_commit', 'snapshot_tree')
LEGACY_ALIASES = {'commit': 'snapshot_commit', 'tree': 'snapshot_tree'}

# (case_id, slug, acceptance condition, assertion plan)
JOINT_CASES = (
    ('J1', 'process_l_semantics', '3 (contract case 1)',
     'candidate dark/constant-light n(t) vs the closed forms; algebraic alpha and B'),
    ('J2', 'synthesis_phase_boundaries', '3 (contract case 2)',
     'candidate A(phi) at chosen initial-state phases vs synthesis_phase.json; inside '
     'limit at phi_on is 0; boundary takes the complementary constant; jump at phi_off; '
     'atan2(x, xc) phase convention; periodicity under phi +/- 2*pi'),
    ('J3', 'suppression_guard_reachable', '3 (contract case 2)',
     'S = 1 above the H1 < 0.001 pM threshold; S = 0 for H1 < 0.001 with m*B > 1; a '
     'negative initial H1 raises (no clamping); the guard rule matches the pinned rule '
     'on the whole boundary table including one-ulp probes'),
    ('J4', 'clearance_analytic_a_zero', '3 (contract case 3)',
     'candidate H1/H2 vs the two-compartment closed form through the declared '
     'experiment channel; a=0 rejected outside it; H3 identically zero; the coupled '
     'oscillator does not contaminate the endocrine states'),
    ('J5', 'nonnegative_state_and_validity', '3 (contract case 4)',
     'validity-block shape on clean and guard-activated runs; guard activations '
     'counted; no clamped value; total integration failure raises ArithmeticError'),
    ('J6', 'units_utc_grid_input_validation', '3 (case 5)',
     'rejection table for malformed grids/lux/config/state/summaries; UTC mapping; '
     'elapsed-hours mapping; minute-form analytic vs hour-form module comparison; '
     'interval boundary inclusivity'),
    ('J7', 'checkpoint_continuity_and_identity', '3 (case 6)',
     'chunked + checkpoint vs single window; stale version/config/source/at_utc/H3 and '
     'revised light prefix rejected; checkpoints carry the identity fields'),
    ('J8', 'oral_zero_and_dose_rejection', '3 (cases 3/6)',
     'H3 identically zero on every endogenous path; oral-dose events rejected; an '
     'external H3=0 checkpoint is accepted and the oral compartment dropped'),
    ('J9', 'opt_in_default_off_frozen_baseline', '4',
     'candidate legacy evaluate/adapter.score/plan vs the frozen decoupling-baseline '
     'expectations through the M1-bound harness; omitted/None selection resolves to '
     'the original five components and melatonin stays excluded'),
    ('J10', 'planner_diagnostic_enabled', '5',
     'an actual call to the candidate public model (counting wrapper); envelope with '
     'selection echo, versions, window fields; per-interval plasma_pM {min,mean,max} in '
     'pM recomputed from the trace; SynOn/SynOff crossings with direction and '
     'definition; trajectory kinetics vs the independent reference integration'),
    ('J11', 'planner_diagnostic_disabled', '5',
     'crossings keys removed when the opt-in is off (no synthesized empty list); '
     'explicit-selection required; plan() output carries no diagnostic keys'),
    ('J12', 'no_duplicated_physiology_or_policy_matrix', '5',
     'source scan of app/planning/** for transcribed model constants; import-boundary '
     'check (no Breslow13 implementation import, no app.sleep_model internals other '
     'than the public api); scanned module list recorded'),
    ('J13', 'no_saliva_no_dlmo_no_quality_claim', 'contract limits',
     'no saliva key and no DLMO key anywhere in the candidate outputs; the not-a-DLMO '
     'statements present in assumptions, deviations and the planning notes; no '
     'ranking-quality claim'),
)


class CandidateIdentityError(RuntimeError):
    """Raised when the bound candidate identity cannot be verified (hard failure)."""


# ---------------------------------------------------------------------------
# Identity harness.
# ---------------------------------------------------------------------------

def resolve_manifest_path(cli_path=None):
    """Resolve the manifest path: CLI option, then environment, then the default."""
    if cli_path:
        return Path(cli_path).resolve()
    from_environment = os.environ.get(MANIFEST_ENV_VAR)
    if from_environment:
        return Path(from_environment).resolve()
    return DEFAULT_MANIFEST_PATH


def resolve_snapshot_path(cli_path=None):
    """Resolve the persisted-snapshot capture path (same discipline as the manifest)."""
    if cli_path:
        return Path(cli_path).resolve()
    from_environment = os.environ.get(SNAPSHOT_ENV_VAR)
    if from_environment:
        return Path(from_environment).resolve()
    return DEFAULT_SNAPSHOT_PATH


def _read_json(path, label):
    path = Path(path)
    if not path.is_file():
        raise CandidateIdentityError(f'{label} not found: {path}')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise CandidateIdentityError(f'{label} unreadable: {path}: {exc}')
    if not isinstance(payload, dict):
        raise CandidateIdentityError(f'{label} is not an object: {path}')
    return payload


def load_fixture(name):
    """Load an analytic fixture from tests/fixtures/melatonin/."""
    return _read_json(FIXTURE_DIR / name, f'analytic fixture {name}')


def load_manifest(manifest_path):
    """Read the Lead-bound manifest (canonical Lead schema).

    Canonical fields: ``worktree``, ``snapshot_commit``, ``snapshot_tree``.  The
    legacy ``commit``/``tree`` keys are an adapter only: when present they must
    equal the canonical values, otherwise the manifest is rejected.
    """
    payload = _read_json(manifest_path, 'candidate manifest')
    for field in CANONICAL_FIELDS:
        if not payload.get(field):
            raise CandidateIdentityError(
                f'candidate manifest missing canonical field {field!r}: {manifest_path}')
    for legacy, canonical in LEGACY_ALIASES.items():
        if legacy in payload and payload[legacy] != payload[canonical]:
            raise CandidateIdentityError(
                f'candidate manifest legacy alias {legacy!r} disagrees with the canonical '
                f'{canonical!r}: {manifest_path}')
    return payload


def load_persisted_snapshot(snapshot_path):
    """Read the persisted ``task.verification_snapshot`` capture for this task.

    The capture is the runtime task record's ``verification_snapshot`` object
    written **verbatim** (its native shape: ``integration_id``, ``commit``,
    ``tree``) when Lead binds the candidate.  It is the authority the manifest
    is validated against; nothing about it is synthesized, paraphrased or
    extended by the harness.
    """
    payload = _read_json(snapshot_path, 'persisted task.verification_snapshot capture')
    for field in ('integration_id', 'commit', 'tree'):
        if not payload.get(field):
            raise CandidateIdentityError(
                f'persisted verification snapshot missing {field!r}: {snapshot_path}')
    return payload


def _git(worktree, *arguments):
    result = subprocess.run(['git', '-C', str(worktree), *arguments],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise CandidateIdentityError(
            f'git {" ".join(arguments)} failed in {worktree}: {result.stderr.strip()}')
    return result.stdout.strip()


def verify_candidate_identity(manifest_path, snapshot_path=None,
                              evaluation_worktree=EVALUATION_WORKTREE):
    """Verify the bound candidate identity against the persisted task record.

    Refusal order (every step is a hard failure, never a skip):
    1. the manifest must carry the canonical Lead schema;
    2. the persisted ``task.verification_snapshot`` capture must exist;
    3. the manifest identity must match that persisted record exactly
       (commit, tree, and the integration id when both carry one) - an
       internally consistent but unbacked manifest is invalid input, not a
       fallback;
    4. the candidate path may not be the evaluation worktree or live inside it;
    5. ``HEAD`` must equal the bound commit;
    6. ``HEAD^{tree}`` must equal the bound tree;
    7. the candidate worktree must be clean.
    """
    manifest = load_manifest(manifest_path)
    snapshot = load_persisted_snapshot(snapshot_path or resolve_snapshot_path())
    if str(manifest['snapshot_commit']) != str(snapshot['commit']):
        raise CandidateIdentityError(
            f'candidate manifest snapshot_commit {manifest["snapshot_commit"]!r} is not '
            f'confirmed by the persisted task.verification_snapshot commit '
            f'{snapshot["commit"]!r}')
    if str(manifest['snapshot_tree']) != str(snapshot['tree']):
        raise CandidateIdentityError(
            f'candidate manifest snapshot_tree {manifest["snapshot_tree"]!r} is not '
            f'confirmed by the persisted task.verification_snapshot tree '
            f'{snapshot["tree"]!r}')
    if manifest.get('integration_id') and snapshot.get('integration_id') \
            and str(manifest['integration_id']) != str(snapshot['integration_id']):
        raise CandidateIdentityError(
            f'candidate manifest integration_id {manifest["integration_id"]!r} is not '
            f'confirmed by the persisted task.verification_snapshot integration_id '
            f'{snapshot["integration_id"]!r}')
    worktree = Path(manifest['worktree']).resolve()
    evaluation = Path(evaluation_worktree).resolve()
    if worktree == evaluation or evaluation in worktree.parents:
        raise CandidateIdentityError(
            f'candidate worktree must not be the evaluation worktree or live inside it: '
            f'{worktree}')
    head = _git(worktree, 'rev-parse', 'HEAD')
    if head != manifest['snapshot_commit']:
        raise CandidateIdentityError(
            f'candidate HEAD drifted: bound {manifest["snapshot_commit"]}, found {head}')
    tree = _git(worktree, 'rev-parse', 'HEAD^{tree}')
    if tree != manifest['snapshot_tree']:
        raise CandidateIdentityError(
            f'candidate tree drifted: bound {manifest["snapshot_tree"]}, found {tree}')
    status = _git(worktree, 'status', '--porcelain')
    if status:
        raise CandidateIdentityError(f'candidate worktree is not clean:\n{status}')
    return manifest


def require_bound_manifest(cli_manifest=None, cli_snapshot=None):
    """Return the verified manifest, or skip with a clear reason (no binding yet)."""
    manifest_path = resolve_manifest_path(cli_manifest)
    snapshot_path = resolve_snapshot_path(cli_snapshot)
    if not manifest_path.is_file() or not snapshot_path.is_file():
        pytest.skip(f'{NO_MANIFEST_REASON} (manifest {manifest_path}, snapshot {snapshot_path})')
    return verify_candidate_identity(manifest_path, snapshot_path)


def _load_module_from_evaluation(relative_path, module_name):
    """Load a file from the evaluation worktree by absolute path (never sys.path)."""
    path = EVALUATION_WORKTREE / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_for_comparison():
    """Load the Phase-1 reference module strictly as the independent comparison side.

    It is never treated as (or named like) the candidate, and this file contains
    no import of it (AST-checked).
    """
    return _load_module_from_evaluation('tests/test_melatonin_reference.py',
                                        'melatonin_reference_compare')


def load_candidate_modules(manifest):
    """Load the verified candidate into this process and assert the binding."""
    worktree = Path(manifest['worktree']).resolve()
    if worktree == EVALUATION_WORKTREE or EVALUATION_WORKTREE in worktree.parents:
        raise CandidateIdentityError(
            f'candidate worktree must not be the evaluation worktree or live inside it: {worktree}')
    for name in [name for name in sys.modules
                 if name == 'app' or name.startswith('app.')
                 or name == 'rooster_bench' or name.startswith('rooster_bench.')]:
        del sys.modules[name]
    sys.path.insert(0, str(worktree))
    import app.sleep_model.melatonin as melatonin_module
    import app.sleep_model.api as api_module
    import app.sleep_model.experiments as experiments_module
    import app.planning.engine as engine_module
    import app.planning.melatonin_diagnostics as diagnostics_module
    modules = {
        'app.sleep_model.melatonin': melatonin_module,
        'app.sleep_model.api': api_module,
        'app.sleep_model.experiments': experiments_module,
        'app.planning.engine': engine_module,
        'app.planning.melatonin_diagnostics': diagnostics_module,
    }
    for name, module in modules.items():
        module_file = Path(module.__file__).resolve()
        if worktree not in module_file.parents:
            raise CandidateIdentityError(
                f'{name} is not loaded from the candidate worktree: {module_file}')
    for name, module in sorted(sys.modules.items()):
        if name == 'app' or name.startswith('app.'):
            module_file = Path(getattr(module, '__file__', '') or '').resolve()
            if EVALUATION_WORKTREE in module_file.parents:
                raise CandidateIdentityError(
                    f'{name} still resolves under the evaluation worktree: {module_file}')
    return modules


def binding_assertions(modules, worktree):
    """Assert the loaded modules really are the bound candidate (at compare time)."""
    for name, module in modules.items():
        module_file = Path(module.__file__).resolve()
        if worktree not in module_file.parents:
            raise CandidateIdentityError(
                f'binding lost: {name} no longer resolves under the candidate worktree '
                f'({module_file})')
    for name, module in sorted(sys.modules.items()):
        if name == 'app' or name.startswith('app.'):
            module_file = Path(getattr(module, '__file__', '') or '').resolve()
            if EVALUATION_WORKTREE in module_file.parents:
                raise CandidateIdentityError(
                    f'binding lost: {name} resolves under the evaluation worktree ({module_file})')


# ---------------------------------------------------------------------------
# Candidate fixture: verify -> purge -> pin -> load -> re-verify.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def candidate():
    manifest = require_bound_manifest()
    worktree = Path(manifest['worktree']).resolve()
    modules = load_candidate_modules(manifest)
    binding_assertions(modules, worktree)
    baseline = _load_module_from_evaluation('tests/test_decoupling_baseline.py',
                                            'candidate_frozen_baseline_check')
    # Re-verify immediately before the comparisons run.
    verify_candidate_identity(resolve_manifest_path(), resolve_snapshot_path())
    return SimpleNamespace(worktree=worktree, manifest=manifest,
                           melatonin=modules['app.sleep_model.melatonin'],
                           api=modules['app.sleep_model.api'],
                           experiments=modules['app.sleep_model.experiments'],
                           engine=modules['app.planning.engine'],
                           diagnostics=modules['app.planning.melatonin_diagnostics'],
                           baseline=baseline, frozen=baseline._load_fixture())


# ---------------------------------------------------------------------------
# Independent comparison math (re-transcribed from the contract inside this
# module; the reference module is loaded separately as the comparison side).
# ---------------------------------------------------------------------------

def _alpha(lux):
    return 0.1 * ((lux / 9500.0) ** 0.5) * (lux / (lux + 100.0))


def _light_drive(x, xc, n, lux):
    return 37.0 * _alpha(lux) * (1.0 - n) * (1.0 - 0.4 * x) * (1.0 - 0.4 * xc)


def _process_l_dark(n0, t_hours):
    return n0 * math.exp(-60.0 * 0.007 * t_hours)


def _process_l_constant(n0, lux, t_hours):
    activation = _alpha(lux)
    n_eq = activation / (activation + 0.007)
    return n_eq + (n0 - n_eq) * math.exp(-60.0 * (activation + 0.007) * t_hours)


def _clearance_h1(h10, t_hours):
    return h10 * math.exp(-(0.000783 * 3600.0) * t_hours)


def _clearance_h2(h10, h20, t_hours):
    k1, k2 = 0.000783 * 3600.0, 0.000335 * 3600.0
    return (h20 * math.exp(-k2 * t_hours)
            + k1 * h10 * (math.exp(-k1 * t_hours) - math.exp(-k2 * t_hours)) / (k2 - k1))


def _wrap(phase):
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


def _synthesis(phase, a=None):
    """A(phi) per the contract: wrapped phase, strict branch, exact exponential."""
    drive = 0.0010422 * 3600.0 if a is None else a
    phi = _wrap(phase)
    phi_on, phi_off = 6.113 - 2.0 * math.pi, 4.352 - 2.0 * math.pi
    span = (phi_on - phi_off) % (2.0 * math.pi)
    if phi < phi_on and phi > phi_off:
        activation = -math.expm1(-600.0 * ((phi_on - phi) % (2.0 * math.pi)))
        normalization = -math.expm1(-600.0 * span)
        return drive * activation / normalization
    return drive * math.exp(-15.36 * span)


def _minimal_snapshot(start_utc, end_utc, *, dark=False):
    """Small deterministic forecast snapshot.

    With ``dark=True`` an observed-light event pins the whole window to 0 lux,
    so the integrated melatonin run is dark regardless of the assumed indoor
    scenario (the J10 kinetics comparison relies on that)."""
    events = [{'id': 'env-home', 'kind': 'environment', 'start_utc': start_utc,
               'end_utc': end_utc, 'timezone': 'UTC', 'location': 'home',
               'availability': 'available', 'confirmed': True}]
    if dark:
        events.append({'id': 'light-dark', 'kind': 'light', 'metric': 'photopic_lux',
                       'lux': 0, 'start_utc': start_utc, 'end_utc': end_utc})
    return {'now_utc': start_utc, 'horizon_end_utc': end_utc, 'revision': 3,
            'profile': {'home_timezone': 'UTC', 'habit_bed_local': '23:00',
                        'habit_wake_local': '07:00', 'sleep_need_hours': 8},
            'initial_state': {}, 'events': events}


def _forbidden_key_violations(payload, forbidden):
    violations = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(token in lowered for token in forbidden):
                    violations.append(f'{path}.{key}')
                walk(value, f'{path}.{key}')
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f'{path}[{index}]')

    walk(payload, '$')
    return violations


# ---------------------------------------------------------------------------
# Harness tests (no candidate needed).
# ---------------------------------------------------------------------------

def test_joint_module_imports_no_candidate_or_reference():
    """The joint module imports neither production code nor the reference module."""
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {name for name in imported
                 if name.split('.')[0] in {'app', 'rooster_bench', 'tests'}
                 or name.endswith('test_melatonin_reference')}
    assert not forbidden, f'the joint module must not import {forbidden}'
    # The candidate loader is real now, so the refusal path is exercised instead:
    # pointing it at the evaluation worktree must hard-fail (never load anything).
    with pytest.raises(CandidateIdentityError) as failure:
        load_candidate_modules({'worktree': str(EVALUATION_WORKTREE),
                                'snapshot_commit': '0' * 40, 'snapshot_tree': '0' * 40})
    assert 'evaluation worktree' in str(failure.value)


def test_manifest_path_resolution_prefers_cli_then_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(MANIFEST_ENV_VAR, raising=False)
    monkeypatch.delenv(SNAPSHOT_ENV_VAR, raising=False)
    assert resolve_manifest_path(None) == DEFAULT_MANIFEST_PATH
    assert resolve_snapshot_path(None) == DEFAULT_SNAPSHOT_PATH
    environment = tmp_path / 'from-environment.json'
    monkeypatch.setenv(MANIFEST_ENV_VAR, str(environment))
    monkeypatch.setenv(SNAPSHOT_ENV_VAR, str(environment))
    assert resolve_manifest_path(None) == environment.resolve()
    assert resolve_snapshot_path(None) == environment.resolve()
    cli = tmp_path / 'from-cli.json'
    assert resolve_manifest_path(str(cli)) == cli.resolve()
    assert resolve_snapshot_path(str(cli)) == cli.resolve()


def test_manifest_requires_the_canonical_lead_schema(tmp_path):
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps({
        'worktree': str(tmp_path), 'commit': 'a' * 40, 'tree': 'b' * 40}),
        encoding='utf-8')
    with pytest.raises(CandidateIdentityError) as failure:
        load_manifest(manifest_path)
    assert 'snapshot_commit' in str(failure.value)
    manifest_path.write_text(json.dumps({
        'worktree': str(tmp_path), 'snapshot_commit': 'a' * 40, 'snapshot_tree': 'b' * 40,
        'commit': 'c' * 40, 'tree': 'd' * 40}), encoding='utf-8')
    with pytest.raises(CandidateIdentityError) as failure:
        load_manifest(manifest_path)
    assert 'legacy alias' in str(failure.value)
    manifest_path.write_text(json.dumps({
        'worktree': str(tmp_path), 'snapshot_commit': 'a' * 40, 'snapshot_tree': 'b' * 40,
        'commit': 'a' * 40, 'tree': 'b' * 40}), encoding='utf-8')
    assert load_manifest(manifest_path)['snapshot_commit'] == 'a' * 40


def test_missing_manifest_or_snapshot_is_a_hard_error_not_a_skip(tmp_path):
    with pytest.raises(CandidateIdentityError):
        load_manifest(tmp_path / 'absent.json')
    with pytest.raises(CandidateIdentityError):
        load_persisted_snapshot(tmp_path / 'absent.json')
    with pytest.raises(CandidateIdentityError):
        verify_candidate_identity(tmp_path / 'absent.json', tmp_path / 'absent.json')


def test_manifest_not_confirmed_by_the_persisted_snapshot_is_rejected(tmp_path):
    manifest_path = tmp_path / 'manifest.json'
    snapshot_path = tmp_path / 'snapshot.json'
    manifest_path.write_text(json.dumps({
        'worktree': str(tmp_path), 'snapshot_commit': 'a' * 40, 'snapshot_tree': 'b' * 40}),
        encoding='utf-8')
    snapshot_path.write_text(json.dumps({
        'integration_id': 'MELATONIN-JOINT-01', 'commit': 'c' * 40, 'tree': 'd' * 40}),
        encoding='utf-8')
    with pytest.raises(CandidateIdentityError) as failure:
        verify_candidate_identity(manifest_path, snapshot_path)
    assert 'persisted task.verification_snapshot' in str(failure.value)
    snapshot_path.write_text(json.dumps({
        'integration_id': 'MELATONIN-JOINT-01', 'commit': 'a' * 40, 'tree': 'e' * 40}),
        encoding='utf-8')
    with pytest.raises(CandidateIdentityError) as failure:
        verify_candidate_identity(manifest_path, snapshot_path)
    assert 'persisted task.verification_snapshot' in str(failure.value)


def test_binding_check_refuses_the_evaluation_worktree(tmp_path):
    """Negative control: the binding check must FAIL when pointed at this worktree."""
    identity = {'worktree': str(EVALUATION_WORKTREE), 'snapshot_commit': '0' * 40,
                'snapshot_tree': '0' * 40}
    manifest_path = tmp_path / 'manifest.json'
    snapshot_path = tmp_path / 'snapshot.json'
    manifest_path.write_text(json.dumps(identity), encoding='utf-8')
    snapshot_path.write_text(json.dumps({
        'integration_id': 'MELATONIN-JOINT-01', 'commit': '0' * 40, 'tree': '0' * 40}),
        encoding='utf-8')
    with pytest.raises(CandidateIdentityError) as failure:
        verify_candidate_identity(manifest_path, snapshot_path,
                                  evaluation_worktree=EVALUATION_WORKTREE)
    assert 'evaluation worktree' in str(failure.value)


def test_phase1_hold_skips_joint_cases_without_a_manifest(monkeypatch):
    monkeypatch.delenv(MANIFEST_ENV_VAR, raising=False)
    monkeypatch.delenv(SNAPSHOT_ENV_VAR, raising=False)
    if resolve_manifest_path(None).is_file() and resolve_snapshot_path(None).is_file():
        pytest.skip('a candidate manifest and snapshot exist at the default paths')
    with pytest.raises(pytest.skip.Exception):
        require_bound_manifest(None)


# ---------------------------------------------------------------------------
# J1–J13: the joint assertions against the bound candidate.
# ---------------------------------------------------------------------------

def test_J1_process_l_semantics(candidate):
    """Dark decay and constant-light Process L against the closed forms; B algebra."""
    mel = candidate.melatonin
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T06:00:00Z'
    grid = [f'2027-06-01T{hour:02d}:00:00Z' for hour in range(6)] + [end]
    # Dark: n(t) = n0 * exp(-60*beta*t) with the pinned n0 = 0.519.
    dark = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid, light_lux=[0.0] * 6)
    assert dark['status'] == 'complete'
    assert len(dark['trace']) == len(grid)
    for row in dark['trace']:
        expected = _process_l_dark(0.519, row['elapsed_hours'])
        assert math.isclose(row['n'], expected, rel_tol=1e-6, abs_tol=1e-12)
        expected_drive = _light_drive(row['x'], row['xc'], row['n'], row['light_lux'])
        assert math.isclose(row['B'], expected_drive, rel_tol=1e-9, abs_tol=1e-15)
        assert row['light_lux'] == 0.0 and row['light_provenance'] == 'assumed'
    # Constant light 1000 lux: the closed form with the pinned n0.
    bright = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                   light_lux=[1000.0] * 6)
    for row in bright['trace']:
        expected = _process_l_constant(0.519, 1000.0, row['elapsed_hours'])
        assert math.isclose(row['n'], expected, rel_tol=1e-6, abs_tol=1e-12)
    # n stays within [0, 1] on both runs.
    for result in (dark, bright):
        assert all(0.0 <= row['n'] <= 1.0 for row in result['trace'])
        assert result['validity']['state_domain_ok'] is True


def test_J2_synthesis_phase_boundaries(candidate):
    """A(phi) piecewise values, boundary/jump semantics, periodicity vs the fixture."""
    mel = candidate.melatonin
    fixture = load_fixture('synthesis_phase.json')
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T01:00:00Z'
    for case in fixture['cases']:
        target = case['phase']
        initial = {'x': math.sin(target), 'xc': math.cos(target), 'n': 0.0,
                   'H1': 1.0, 'H2': 0.0}
        result = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=[start, end],
                                       light_lux=[0.0], initial_state=initial)
        row = result['trace'][0]
        achieved = row['phase_rad']
        deviation = abs(_wrap(achieved - target))
        assert deviation < 1e-9, f'{case["label"]}: phase realization off by {deviation}'
        # The candidate's A must equal the contract piecewise function at the
        # phase the candidate itself realized (branch-aware comparison).
        expected = _synthesis(achieved)
        assert math.isclose(row['A'], expected, rel_tol=case['rtol'], abs_tol=1e-15), \
            f'{case["label"]}: A {row["A"]!r} vs {expected!r}'
        # Away from the boundaries the fixture's frozen value must be reproduced.
        if not case['label'].startswith('boundary'):
            assert math.isclose(row['A'], case['expected_A'], rel_tol=case['rtol'], abs_tol=1e-15)
            assert case['expected_branch'] == ('inside' if fixture['phi_off'] < _wrap(target)
                                               < fixture['phi_on'] else 'complementary')
    # Inside limit at phi_off is a; the boundary takes the small constant (jump).
    a_drive = 0.0010422 * 3600.0
    inside = next(case for case in fixture['cases'] if case['label'] == 'inside_just_after_phi_off')
    boundary = next(case for case in fixture['cases'] if case['label'] == 'boundary_phi_off')
    assert math.isclose(inside['expected_A'], a_drive, rel_tol=1e-15)
    assert math.isclose(boundary['expected_A'], fixture['complementary_constant'], rel_tol=1e-15)
    assert inside['expected_A'] / boundary['expected_A'] > 1e9
    # Inside limit at phi_on is zero (Lead correction).
    assert fixture['inside_limit_at_phi_on'] == 0.0
    # Periodicity: a target phase shifted by 2*pi realizes the same wrapped phase.
    base_phase = 0.4 * (fixture['phi_off'] + fixture['phi_on'])
    base_initial = {'x': math.sin(base_phase), 'xc': math.cos(base_phase), 'n': 0.0,
                    'H1': 1.0, 'H2': 0.0}
    shifted_initial = {'x': math.sin(base_phase + 2.0 * math.pi),
                       'xc': math.cos(base_phase + 2.0 * math.pi), 'n': 0.0,
                       'H1': 1.0, 'H2': 0.0}
    grid = [start, end]
    zeros = [0.0]
    base = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                 light_lux=zeros, initial_state=base_initial)
    shifted = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                    light_lux=zeros, initial_state=shifted_initial)
    assert math.isclose(base['trace'][0]['A'], shifted['trace'][0]['A'], rel_tol=1e-12)
    assert math.isclose(shifted['trace'][0]['phase_rad'], base['trace'][0]['phase_rad'],
                        rel_tol=0.0, abs_tol=1e-9)


def test_J3_suppression_guard_reachable(candidate):
    """Reachable guard behaviour plus rule parity on the full boundary table."""
    mel = candidate.melatonin
    fixture = load_fixture('suppression_guard.json')
    config = mel.melatonin_configuration()
    for case in fixture['cases'] + fixture['ulp_cases']:
        assert mel.suppression_guard(case['H1'], case['B'], config) == case['expected_S']
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T02:00:00Z'
    grid = [start, '2027-06-01T01:00:00Z', end]
    # S = 0 is reachable: H1 below the threshold, m*B > 1 (bright light, n = 0).
    suppressed = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                       light_lux=[1000.0, 1000.0],
                                       initial_state={'x': 0.0, 'xc': 0.0, 'n': 0.0,
                                                      'H1': 0.0009, 'H2': 0.0})
    assert suppressed['trace'][0]['suppression_guard'] == 0.0
    assert suppressed['validity']['guard_activations'] >= 1
    # S = 1 above the threshold under the same light.
    producing = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                      light_lux=[1000.0, 1000.0],
                                      initial_state={'x': 0.0, 'xc': 0.0, 'n': 0.0,
                                                     'H1': 1.0, 'H2': 0.0})
    assert producing['trace'][0]['suppression_guard'] == 1.0
    # A negative initial H1 raises (no clamping anywhere).
    with pytest.raises(ValueError, match='nonnegative H1'):
        mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                              light_lux=[0.0, 0.0],
                              initial_state={'x': 0.0, 'xc': 0.0, 'n': 0.5,
                                             'H1': -0.5, 'H2': 0.0})
    # H1 = 0 with the guard active: concentrations stay nonnegative throughout.
    zero = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                 light_lux=[1000.0, 1000.0],
                                 initial_state={'x': 0.0, 'xc': 0.0, 'n': 0.0,
                                                'H1': 0.0, 'H2': 0.0})
    assert all(row['H1'] >= 0.0 for row in zero['trace'])
    assert all(row['H2'] >= 0.0 for row in zero['trace'])
    assert zero['status'] == 'complete'


def test_J4_clearance_analytic_a_zero(candidate):
    """Analytic clearance with a=0 through the declared experiment channel."""
    mel = candidate.melatonin
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T06:00:00Z'
    # a = 0 is rejected outside the declared experiment channel.
    with pytest.raises(ValueError, match='declared analytic-clearance'):
        mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=[start, end],
                              light_lux=[0.0], config={'a': 0.0})
    # The configuration channel itself accepts a = 0; the published default stays positive.
    assert mel.melatonin_configuration({'a': 0.0})['a'] == 0.0
    assert mel.melatonin_configuration()['a'] == 0.0010422 * 3600.0
    # Declared experiment channel: a = 0 flows through the formal compute path.
    config_hash = mel.melatonin_model_config_hash(mel.melatonin_configuration({'a': 0.0}))
    checkpoint = {'x': -0.219, 'xc': -1.22, 'n': 0.519, 'H1': 50.0, 'H2': 80.0,
                  'at_utc': start, 'window_start_utc': start,
                  'melatonin_model_version': mel.MELATONIN_MODEL_VERSION,
                  'melatonin_config_hash': config_hash,
                  'source_sha256': mel.SOURCE_MANIFEST_SHA256,
                  'light_prefix_hash': mel.digest({'series': []})}
    snapshot = _minimal_snapshot(start, end)
    snapshot['melatonin_checkpoint'] = checkpoint
    result = candidate.experiments.run('melatonin-clearance-a0', snapshot, config={},
                                       components=['melatonin'], melatonin_config={'a': 0.0},
                                       start_utc=start, end_utc=end)
    block = result['melatonin']
    assert block['provenance']['melatonin_config_hash'] == config_hash
    assert block['provenance']['configuration']['a'] == 0.0
    assert result['experiment']['melatonin_model_config_hash'] == config_hash
    rows = block['trace']
    assert rows[0]['H1'] == 50.0 and rows[0]['H2'] == 80.0
    for row in rows:
        t_hours = row['elapsed_hours']
        assert math.isclose(row['H1'], _clearance_h1(50.0, t_hours), rel_tol=1e-6, abs_tol=1e-9)
        assert math.isclose(row['H2'], _clearance_h2(50.0, 80.0, t_hours), rel_tol=1e-6, abs_tol=1e-9)
        assert row['A'] == 0.0
    # The coupled oscillator evolves but does not contaminate the endocrine states.
    assert any(abs(row['x'] - rows[0]['x']) > 1e-9 for row in rows[1:])
    assert all(row['H1'] >= 0.0 and row['H2'] >= 0.0 for row in rows)


def test_J5_nonnegative_state_and_validity(candidate, monkeypatch):
    """Validity-block shape, guard counting, and the total-failure convention."""
    mel = candidate.melatonin
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T02:00:00Z'
    grid = [start, '2027-06-01T01:00:00Z', end]
    clean = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                  light_lux=[0.0, 0.0])
    assert clean['status'] == 'complete'
    validity = clean['validity']
    assert validity['status'] == 'complete'
    assert validity['numerical_failures'] == []
    assert validity['nonnegative_concentrations'] is True
    assert validity['state_domain_ok'] is True
    assert validity['finite_states'] is True
    assert isinstance(validity['guard_activations'], int)
    # Guard activations are counted, and concentrations stay nonnegative without
    # any clamping (the published guard is the mechanism).
    activated = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                      light_lux=[1000.0, 1000.0],
                                      initial_state={'x': 0.0, 'xc': 0.0, 'n': 0.0,
                                                     'H1': 0.0, 'H2': 0.0})
    assert activated['validity']['guard_activations'] >= 1
    assert all(row['H1'] >= 0.0 and row['H2'] >= 0.0 for row in activated['trace'])
    assert activated['status'] == 'complete'
    # A total integration failure raises ArithmeticError with the solver message.
    from types import SimpleNamespace as _Namespace

    def failing_solver(*args, **kwargs):
        return _Namespace(success=False, message='forced solver failure for the validity test',
                          y=None)

    monkeypatch.setattr(mel, 'solve_ivp', failing_solver)
    with pytest.raises(ArithmeticError, match='forced solver failure'):
        mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid, light_lux=[0.0, 0.0])


def test_J6_units_utc_grid_input_validation(candidate):
    """UTC mapping, elapsed hours, units, and the input-validation rejection table."""
    mel = candidate.melatonin
    start, middle, end = '2027-06-01T00:00:00Z', '2027-06-01T01:00:00Z', '2027-06-01T02:00:00Z'
    grid = [start, middle, end]
    result = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                   light_lux=[0.0, 0.0])
    assert [row['time_utc'] for row in result['trace']] == grid
    assert [row['elapsed_hours'] for row in result['trace']] == [0.0, 1.0, 2.0]
    assert result['units']['plasma_pM'] == 'pmol/L (pM)'
    assert 'pmol/L' in result['units']['H1'] and 'pM' in result['units']['H1']
    assert 'pmol/L' in result['units']['H2']
    assert result['units']['A'] == 'pM/hour endogenous synthesis drive'
    # A UTC offset input normalizes to UTC rows.
    offset_start = '2027-06-01T09:00:00+09:00'
    offset_end = '2027-06-01T11:00:00+09:00'
    offset = mel.compute_melatonin(start_utc=offset_start, end_utc=offset_end,
                                   grid_utc=[offset_start, '2027-06-01T01:00:00Z', offset_end],
                                   light_lux=[0.0, 0.0])
    assert offset['trace'][0]['time_utc'] == start
    # Minute-form vs hour-form: with the published default synthesis constant the
    # complementary-branch drive leaks a tiny synthesis flow even in the dark, so
    # the analytic zero-synthesis comparison belongs to the declared a=0 channel
    # (J4).  Here the exact tiny magnitude is what is asserted.
    minute_grid = [start, '2027-06-01T00:30:00Z', middle, '2027-06-01T01:30:00Z', end]
    minute_run = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=minute_grid,
                                       light_lux=[0.0] * 4)
    for row in minute_run['trace']:
        assert 0.0 <= row['H1'] < 1e-9
        assert 0.0 <= row['H2'] < 1e-8
    # Summary statistics include both endpoints of the interval.
    summaries = [{'id': 'whole', 'start_utc': start, 'end_utc': end}]
    whole = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                  light_lux=[0.0, 0.0], summaries=summaries)
    summary = whole['summaries'][0]
    assert summary['id'] == 'whole'
    assert summary['sample_count'] == len(grid)
    plasma = [row['H2'] for row in whole['trace']]
    assert math.isclose(summary['plasma_pM']['min'], min(plasma), rel_tol=1e-12)
    assert math.isclose(summary['plasma_pM']['max'], max(plasma), rel_tol=1e-12)
    assert math.isclose(summary['plasma_pM']['mean'], sum(plasma) / len(plasma), rel_tol=1e-12)
    # Rejection table: every malformed input raises (never a silent fallback).
    rejections = {
        'grid with a single instant': dict(grid_utc=[start], light_lux=[0.0]),
        'duplicate grid instants': dict(grid_utc=[start, start, end], light_lux=[0.0, 0.0]),
        'decreasing grid instants': dict(grid_utc=[start, end, middle], light_lux=[0.0, 0.0]),
        'grid not covering the start': dict(
            grid_utc=[middle, end], light_lux=[0.0]),
        'grid not covering the end': dict(grid_utc=[start, middle], light_lux=[0.0]),
        'negative light': dict(light_lux=[-1.0, 0.0]),
        'non-finite light': dict(light_lux=[float('nan'), 0.0]),
        'empty light history': dict(light_lux=[]),
        'light length mismatch': dict(light_lux=[0.0, 0.0, 0.0]),
        'light provenance length mismatch': dict(light_lux=[0.0, 0.0],
                                                  light_provenance=['assumed']),
        'empty light provenance label': dict(light_lux=[0.0, 0.0],
                                              light_provenance=['assumed', '  ']),
        'window end before start': dict(start_utc=end, end_utc=start,
                                        grid_utc=[end, start], light_lux=[0.0]),
        'dose events are rejected': dict(dose_events=[{'at_utc': start, 'dose_mg': 3.0}]),
        'unknown melatonin parameter': dict(config={'bogus': 1.0}),
        'negative synthesis constant': dict(config={'a': -1.0}),
        'b at or above one': dict(config={'b': 1.0}),
        'negative initial H1': dict(initial_state={'x': 0.0, 'xc': 0.0, 'n': 0.5,
                                                    'H1': -1.0, 'H2': 0.0}),
        'initial n outside [0, 1]': dict(initial_state={'x': 0.0, 'xc': 0.0, 'n': 1.5,
                                                         'H1': 0.0, 'H2': 0.0}),
        'non-finite initial state': dict(initial_state={'x': float('nan'), 'xc': 0.0,
                                                         'n': 0.5, 'H1': 0.0, 'H2': 0.0}),
        'summary interval reversed': dict(summaries=[{'id': 'bad', 'start_utc': end,
                                                       'end_utc': start}]),
        'summary interval outside the window': dict(
            summaries=[{'id': 'bad', 'start_utc': '2027-07-01T00:00:00Z',
                        'end_utc': '2027-07-01T01:00:00Z'}]),
    }
    for label, overrides in rejections.items():
        with pytest.raises((ValueError, TypeError)):
            mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                  light_lux=[0.0, 0.0], **overrides)


def test_J7_checkpoint_continuity_and_identity(candidate):
    """Chunked resume equals the single window; stale identities are rejected."""
    mel = candidate.melatonin
    start, middle, end = '2027-06-01T00:00:00Z', '2027-06-01T06:00:00Z', '2027-06-01T12:00:00Z'
    grid_first = [f'2027-06-01T{hour:02d}:00:00Z' for hour in range(7)]
    grid_second = [f'2027-06-01T{hour:02d}:00:00Z' for hour in range(6, 13)]
    grid_whole = [f'2027-06-01T{hour:02d}:00:00Z' for hour in range(13)]
    first = mel.compute_melatonin(start_utc=start, end_utc=middle, grid_utc=grid_first,
                                  light_lux=[0.0] * 6)
    checkpoint = first['final_checkpoint']
    assert checkpoint['at_utc'] == middle
    assert checkpoint['melatonin_model_version'] == mel.MELATONIN_MODEL_VERSION
    assert checkpoint['melatonin_config_hash'] == mel.melatonin_model_config_hash(
        mel.melatonin_configuration())
    assert checkpoint['source_sha256'] == 'd2635b34eebe9a8e4c59383f10413a452544a4405df6fcc8e3190aa4e7977435'
    resumed = mel.compute_melatonin(start_utc=middle, end_utc=end, grid_utc=grid_second,
                                    light_lux=[0.0] * 6, checkpoint=checkpoint)
    whole = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid_whole,
                                  light_lux=[0.0] * 12)
    for row_resumed, row_whole in zip(resumed['trace'], whole['trace'][6:]):
        assert row_resumed['time_utc'] == row_whole['time_utc']
        for key in ('x', 'xc', 'n', 'H1', 'H2'):
            assert math.isclose(row_resumed[key], row_whole[key], rel_tol=1e-6, abs_tol=1e-9), \
                f'seam mismatch at {row_resumed["time_utc"]} ({key})'
    # Identity rejections (each variant fails at its own check).
    base = dict(start_utc=middle, end_utc=end, grid_utc=grid_second, light_lux=[0.0] * 6)
    with pytest.raises(ValueError, match='model version'):
        mel.compute_melatonin(**base, checkpoint={**checkpoint,
                                                  'melatonin_model_version': 'other-version'})
    changed_hash = mel.melatonin_model_config_hash(mel.melatonin_configuration({'r': 15.4}))
    with pytest.raises(ValueError, match='configuration'):
        mel.compute_melatonin(**base, checkpoint={**checkpoint, 'melatonin_config_hash': changed_hash})
    with pytest.raises(ValueError, match='source pin'):
        mel.compute_melatonin(**base, checkpoint={**checkpoint, 'source_sha256': 'f' * 64})
    with pytest.raises(ValueError, match='exactly at the checkpoint instant'):
        mel.compute_melatonin(**base, checkpoint={**checkpoint, 'at_utc': start})
    with pytest.raises(ValueError, match='identically zero'):
        mel.compute_melatonin(**base, checkpoint={**checkpoint, 'H3': 0.5})
    with pytest.raises(ValueError, match='light history'):
        mel.compute_melatonin(**base, checkpoint=checkpoint,
                              light_prefix=[{'start_utc': start, 'light_lux': 500.0,
                                             'light_provenance': 'assumed'}])
    # An external H3=0 checkpoint is accepted and the oral compartment is dropped.
    accepted = mel.compute_melatonin(**base, checkpoint={**checkpoint, 'H3': 0.0})
    assert 'H3' not in accepted['trace'][0]
    assert any('H3=0' in item for item in accepted['assumptions'])


def test_J8_oral_zero_and_dose_rejection(candidate):
    """Endogenous-only operation: no H3 state, oral events rejected, never ignored."""
    mel = candidate.melatonin
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T02:00:00Z'
    grid = [start, '2027-06-01T01:00:00Z', end]
    result = mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                                   light_lux=[0.0, 0.0])
    assert all('H3' not in row for row in result['trace'])
    assert any('H3' in item and 'identically zero' in item for item in result['assumptions'])
    with pytest.raises(ValueError, match='oral dose events'):
        mel.compute_melatonin(start_utc=start, end_utc=end, grid_utc=grid,
                              light_lux=[0.0, 0.0], dose_events=[{'at_utc': start}])
    # The model states its own oral-zero / no-saliva contract.
    assert any('no saliva concentration is produced' in item for item in result['assumptions'])
    assert any('H3' in item and 'identically zero' in item for item in result['assumptions'])


def test_J9_opt_in_default_off_frozen_baseline(candidate):
    """The frozen decoupling baseline holds against the candidate; opt-in is explicit."""
    baseline, frozen = candidate.baseline, candidate.frozen
    schema_problems = baseline._schema_problems(frozen)
    assert not schema_problems, schema_problems
    failures = []
    for case_id in sorted(frozen['cases']):
        case_problems = baseline.verify_case(case_id, frozen['cases'][case_id])
        if case_problems:
            failures.append((case_id, case_problems[:5]))
    assert not failures, f'frozen baseline mismatches against the candidate: {failures}'
    api = candidate.api
    requested, resolved = api.resolve_components(None)
    assert requested == sorted(requested)
    assert set(resolved) == {'pressure', 'circadian', 'inertia', 'burden', 'metrics'}
    assert 'melatonin' not in resolved
    start, end = '2027-06-01T00:00:00Z', '2027-06-01T02:00:00Z'
    snapshot = _minimal_snapshot(start, end)
    default_result = api.compute(snapshot, components=None)
    assert 'melatonin' not in default_result
    assert set(default_result['components']['resolved']) == {
        'pressure', 'circadian', 'inertia', 'burden', 'metrics'}
    opt_in = api.compute(snapshot, components=['melatonin'])
    assert 'melatonin' in opt_in['components']['resolved']
    assert 'melatonin' in opt_in


def test_J10_planner_diagnostic_enabled(candidate, monkeypatch):
    """The enabled selection actually calls the public model and propagates its answer."""
    diagnostics, api = candidate.diagnostics, candidate.api
    start, middle, end = '2027-06-01T00:00:00Z', '2027-06-01T06:00:00Z', '2027-06-01T12:00:00Z'
    # The observed-light event pins the window to 0 lux so the integrated
    # melatonin run is dark and directly comparable with the reference
    # integration of the contract equations.
    snapshot = _minimal_snapshot(start, end, dark=True)
    requests = [{'id': 'interval-a', 'start_utc': start, 'end_utc': middle},
                {'id': 'interval-b', 'start_utc': middle, 'end_utc': end}]
    calls = []
    original_compute = api.compute

    def counting_compute(*args, **kwargs):
        calls.append(kwargs)
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(api, 'compute', counting_compute)
    envelope = diagnostics.run(snapshot, melatonin_summaries=requests, melatonin_crossings=True)
    assert len(calls) == 1
    assert calls[0]['components'] == ['melatonin']
    assert calls[0]['melatonin_summaries'] == requests
    assert calls[0]['melatonin_crossings'] is True
    assert envelope['diagnostic'] == 'melatonin'
    assert envelope['selection'] == {'melatonin_summaries': requests, 'melatonin_crossings': True}
    assert envelope['crossings_requested'] is True
    assert envelope['api_version'] and envelope['model_version']
    assert envelope['mode'] == 'forecast'
    assert envelope['knowledge_cutoff_utc'] == start
    assert envelope['evaluation_start_utc'] == start and envelope['evaluation_end_utc'] == end
    block = envelope['melatonin']
    # The melatonin component carries its own model identity (the envelope
    # propagates the compute-level model version; the two are deliberately
    # different version domains and both must be present).
    assert block['model_version'] == candidate.melatonin.MELATONIN_MODEL_VERSION
    assert envelope['model_version']
    assert 'melatonin_config_hash' in block['provenance']
    assert block['provenance']['source']['manifest_sha256'] == \
        'd2635b34eebe9a8e4c59383f10413a452544a4405df6fcc8e3190aa4e7977435'
    assert {item['id'] for item in block['summaries']} == {'interval-a', 'interval-b'}
    trace = block['trace']
    for summary in block['summaries']:
        selected = [row for row in trace
                    if summary['start_utc'] <= row['time_utc'] <= summary['end_utc']]
        assert summary['sample_count'] == len(selected)
        plasma = [row['H2'] for row in selected]
        assert math.isclose(summary['plasma_pM']['min'], min(plasma), rel_tol=1e-12)
        assert math.isclose(summary['plasma_pM']['max'], max(plasma), rel_tol=1e-12)
        assert math.isclose(summary['plasma_pM']['mean'], sum(plasma) / len(plasma), rel_tol=1e-12)
        assert summary['units']['plasma_pM'] == 'pmol/L (pM)'
        assert 'crossings' in summary
        assert 'melatonin_config_hash' in summary['provenance']
        assert summary['validity']['finite_concentrations'] is True
        assert summary['validity']['nonnegative_concentrations'] is True
    # Crossings exist on a dark trajectory, alternate, and carry the definition.
    # From the pinned initial phase (-2.964 rad) the orbit enters the synthesis
    # window (~4 h in) and leaves it again (~10.7 h in), so a 12 h window must
    # contain exactly one SynOn and one SynOff.
    crossings = envelope.get('trajectory_crossings') or []
    kinds = [record['kind'] for record in crossings]
    assert kinds.count('SynOn') == 1 and kinds.count('SynOff') == 1, \
        f'unexpected crossing sequence: {kinds}'
    for record in crossings:
        assert start <= record['at_utc'] <= end
        assert record['direction'] in ('increasing', 'decreasing')
        assert isinstance(record['validity']['valid'], bool)
        assert record['validity']['valid'] is True
        assert 'not a measured DLMO' in record['definition']
    # Half-open attribution: the SynOn (~4 h) falls into interval-a, the SynOff
    # (~10.7 h) into interval-b.
    summaries = {item['id']: item for item in block['summaries']}
    assert [record['kind'] for record in summaries['interval-a']['crossings']] == ['SynOn']
    assert [record['kind'] for record in summaries['interval-b']['crossings']] == ['SynOff']
    # Kinetics parity: the candidate trajectory matches the independent reference
    # integration of the contract equations on the same window and initial state.
    reference = _load_reference_for_comparison()
    initial = (-0.219, -1.22, 0.519, 0.0, 0.0, 0.0)
    solution = reference.reference_integrate(initial, 0.0, duration_hours=12.0)
    for row in trace:
        state = solution.sol(row['elapsed_hours'])
        for index, key in ((0, 'x'), (1, 'xc'), (2, 'n'), (3, 'H1'), (4, 'H2')):
            assert math.isclose(row[key], float(state[index]), rel_tol=1e-6, abs_tol=1e-9), \
                f'{key} at {row["elapsed_hours"]}h: candidate {row[key]!r} vs reference {float(state[index])!r}'


def test_J11_planner_diagnostic_disabled(candidate):
    """With the opt-in off, no crossings field exists anywhere; plan() is untouched."""
    diagnostics, engine = candidate.diagnostics, candidate.engine
    start, middle, end = '2027-06-01T00:00:00Z', '2027-06-01T03:00:00Z', '2027-06-01T06:00:00Z'
    snapshot = _minimal_snapshot(start, end)
    requests = [{'id': 'interval-a', 'start_utc': start, 'end_utc': middle}]
    envelope = diagnostics.run(snapshot, melatonin_summaries=requests, melatonin_crossings=False)
    assert envelope['crossings_requested'] is False
    assert 'crossings' not in envelope['melatonin']
    assert 'trajectory_crossings' not in envelope
    for summary in envelope['melatonin']['summaries']:
        assert 'crossings' not in summary
    assert envelope['melatonin']['summaries'][0]['id'] == 'interval-a'
    assert envelope['melatonin']['summaries'][0]['plasma_pM']['max'] >= \
        envelope['melatonin']['summaries'][0]['plasma_pM']['min']
    # The explicit selection is mandatory; nothing is auto-derived.
    with pytest.raises(ValueError, match='explicit melatonin_summaries'):
        diagnostics.run(snapshot, melatonin_summaries=[])
    with pytest.raises(ValueError, match='explicit melatonin_summaries'):
        diagnostics.run(snapshot)
    # plan() carries no diagnostic keys at all (the frozen baseline pins its content).
    plan_result = engine.plan(_planning_snapshot())
    assert not any('melatonin' in str(key).lower() for key in plan_result)


def _planning_snapshot():
    """Small deterministic planning snapshot (same shape as the frozen baseline cases)."""
    now, horizon = '2027-06-01T06:00:00Z', '2027-06-02T06:00:00Z'
    return {'now_utc': now, 'horizon_end_utc': horizon, 'revision': 12,
            'profile': {'home_timezone': 'Asia/Tokyo', 'sleep_need_hours': 8,
                        'preferred_sleep_start_hour': 22},
            'initial_state': {},
            'events': [
                {'id': 'env-home', 'kind': 'environment', 'start_utc': now,
                 'end_utc': '2027-06-03T06:00:00Z', 'timezone': 'Asia/Tokyo',
                 'location': 'home', 'availability': 'available', 'confirmed': True},
                {'id': 'sleep-1', 'kind': 'sleep_actual', 'source': 'synthetic_observation',
                 'start_utc': '2027-05-31T13:00:00Z', 'end_utc': '2027-05-31T21:00:00Z',
                 'timezone': 'Asia/Tokyo'},
                {'id': 'duty-1', 'kind': 'duty', 'start_utc': '2027-06-01T22:00:00Z',
                 'end_utc': '2027-06-02T02:00:00Z', 'timezone': 'Asia/Tokyo'}],
            'scheduling_policy': {'maximum_wake_hours': 20.0, 'minimum_sleep_minutes': 90}}


def test_J12_no_duplicated_physiology_or_policy_matrix(candidate):
    """app/planning/** transcribes no model constant and imports no model internals."""
    worktree = candidate.worktree
    planning_dir = worktree / 'app' / 'planning'
    scanned = sorted(path.relative_to(worktree).as_posix()
                     for path in planning_dir.rglob('*.py'))
    assert scanned, 'no planning modules found to scan'
    forbidden_fragments = ('1.0422', '3.75192', 'Mmax', 'Hsat', 'betaIP', 'betaCP',
                           'betaAP', 'H1_threshold', '1 - exp(-delta')
    for relative in scanned:
        text = (worktree / relative).read_text(encoding='utf-8')
        for fragment in forbidden_fragments:
            assert fragment not in text, \
                f'{relative} duplicates the model constant {fragment!r}'
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                # The Breslow13 implementation module is the strict boundary: the
                # planning side may use the public api module and the documented
                # lazy legacy bridges (e.g. app.sleep_model.contracts for the
                # frozen adapter version constants), but never the melatonin
                # implementation.
                assert not name.startswith('app.sleep_model.melatonin'), \
                    f'{relative} imports the Breslow13 implementation module: {name}'
    # The only allowed model surface is the public api module, used by the adapter.
    adapter_text = (worktree / 'app' / 'planning' / 'melatonin_diagnostics.py').read_text(
        encoding='utf-8')
    assert 'from app.sleep_model import api' in adapter_text
    # No physiological policy matrix: no required/soft/hard policy structure.
    for relative in scanned:
        text = (worktree / relative).read_text(encoding='utf-8')
        for fragment in ("'required'", '"required"', "'soft':", '"soft":',
                         "'hard':", '"hard":'):
            assert fragment not in text, f'{relative} looks like a physiological policy matrix'
    print('scanned planning modules: ' + ', '.join(scanned))


def test_J13_no_saliva_no_dlmo_no_quality_claim(candidate):
    """No saliva or DLMO key anywhere; the not-a-DLMO statements are present."""
    mel = candidate.melatonin
    start, middle, end = '2027-06-01T00:00:00Z', '2027-06-01T03:00:00Z', '2027-06-01T06:00:00Z'
    result = mel.compute_melatonin(start_utc=start, end_utc=end,
                                   grid_utc=[start, middle, end], light_lux=[0.0, 0.0],
                                   crossings=True,
                                   summaries=[{'id': 'interval-a', 'start_utc': start,
                                               'end_utc': end}])
    violations = _forbidden_key_violations(result, ('saliva', 'dlmo'))
    assert not violations, f'forbidden output keys: {violations}'
    assert any('not a measured DLMO' in item for item in result['assumptions'])
    assert any('no biological validation' in item for item in result['provenance']['deviations'])
    snapshot = _minimal_snapshot(start, end)
    envelope = candidate.diagnostics.run(snapshot, melatonin_summaries=[
        {'id': 'interval-a', 'start_utc': start, 'end_utc': end}], melatonin_crossings=True)
    envelope_violations = _forbidden_key_violations(envelope, ('saliva', 'dlmo'))
    assert not envelope_violations, f'forbidden envelope keys: {envelope_violations}'
    assert any('not a measured DLMO' in note for note in envelope['notes'])
    assert any('no physiological policy' in note for note in envelope['notes'])
    assert any('planning-quality claim' in note for note in envelope['notes'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Melatonin joint candidate harness.')
    parser.add_argument(MANIFEST_CLI_OPTION, dest='manifest', default=None,
                        help='path to the Lead-bound candidate manifest '
                             f'(default: {DEFAULT_MANIFEST_PATH}; env: {MANIFEST_ENV_VAR})')
    parser.add_argument(SNAPSHOT_CLI_OPTION, dest='snapshot', default=None,
                        help='path to the persisted task.verification_snapshot capture '
                             f'(default: {DEFAULT_SNAPSHOT_PATH}; env: {SNAPSHOT_ENV_VAR})')
    parser.add_argument('--verify-identity', action='store_true',
                        help='verify the bound candidate identity and exit')
    args = parser.parse_args(argv)
    if not args.verify_identity:
        parser.error('nothing to do; pass --verify-identity')
    manifest_path = resolve_manifest_path(args.manifest)
    snapshot_path = resolve_snapshot_path(args.snapshot)
    try:
        manifest = verify_candidate_identity(manifest_path, snapshot_path)
    except CandidateIdentityError as exc:
        print('CANDIDATE IDENTITY FAILURE: ' + str(exc), file=sys.stderr)
        return 2
    print('candidate identity verified: commit ' + manifest['snapshot_commit']
          + ', tree ' + manifest['snapshot_tree'] + ', worktree ' + manifest['worktree'])
    return 0


if __name__ == '__main__':
    sys.exit(main())

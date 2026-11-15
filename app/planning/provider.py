"""Constrained MiniMax-M3 adapter for offline product validation.

Legacy workflows describe intent and rank candidates. The incremental Agent
also proposes typed local edits and tool calls; planning validates timestamps,
permissions and constraints. Model tools own scores and code owns version checks.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
BASE_URL = 'https://api.minimax.cn/v1/chat/completions'
MODEL = 'MiniMax-M3'
MAX_OUTPUT_TOKENS = 8192
DEFAULT_OUTPUT_TOKENS = 4096
MAX_HTTP_ATTEMPTS = int(os.environ['ROOSTER_MINIMAX_MAX_ATTEMPTS']) if os.environ.get('ROOSTER_MINIMAX_MAX_ATTEMPTS') else None
MAX_CONCURRENCY = 2
MAX_RETRIES = 2
DEFAULT_TIMEOUT_SECONDS = 45
MAX_SCHEMA_REPAIRS_PER_STAGE = 2
_HTTP_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)
_ATTEMPT_LOCK = threading.Lock()
_HTTP_ATTEMPTS = 0
_AUTH_STOP = threading.Event()

class LLMError(RuntimeError):
    """Base class for safe-to-record provider/workflow errors."""

class BudgetExceeded(LLMError):
    pass

class AuthorizationStopped(LLMError):
    pass

class SchemaError(LLMError):
    pass

class StaleVersionError(SchemaError):
    pass

class ReplayMiss(LLMError):
    pass

def _json_default(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if hasattr(value, '__dict__'):
        return value.__dict__
    return str(value)

def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=_json_default)

def _candidate_id(candidate: dict[str, Any]) -> str:
    value = candidate.get('id', candidate.get('candidateId', candidate.get('candidate_id')))
    return '' if value is None else str(value)
_MODEL_METRICS = ('sleepAdequacyMean', 'sleepAdequacyWorst', 'circadianFit', 'dutyBurdenMean', 'dutyBurdenWorst60m', 'intentFit', 'disruptionBlocks', 'terminalCost', 'longestContinuousWakeHours')

def _compact_evaluation_report(value: Any) -> dict[str, Any]:
    """Keep measured scenario vectors and per-duty summaries; omit dense curves."""
    data = value.to_dict() if hasattr(value, 'to_dict') else value
    if not isinstance(data, dict):
        return {'report': data}
    compact: dict[str, Any] = {}
    for key in ('snapshotVersion', 'snapshot_version', 'intentRevision', 'basePlanRevision', 'level', 'modelVersion', 'policyVersion', 'evaluation_id', 'evaluationId'):
        if key in data:
            compact[key] = data[key]
    compact['candidates'] = []
    for candidate in data.get('candidates', []) or []:
        if not isinstance(candidate, dict):
            continue
        compact['candidates'].append({k: candidate[k] for k in ('id', 'candidateId', 'proposalId', 'name', 'validity', 'recommendable', 'notRecommendableReason', 'inputSnapshotVersion', 'intentRevision', 'basePlanRevision', 'sleepBlocks', 'unmetIntentIds', 'assumptions') if k in candidate})
    compact['evaluations'] = []
    for evaluation in data.get('evaluations', []) or []:
        if not isinstance(evaluation, dict):
            continue
        row = {k: evaluation[k] for k in ('proposalId', 'candidateId', 'evaluation_id', 'evaluationId', 'id', 'inputSnapshotVersion', 'intentRevision', 'basePlanRevision', 'validity', 'recommendable', 'hard_valid', 'robustUtility') if k in evaluation}
        scenarios = []
        for scenario in evaluation.get('scenarios', []) or []:
            if not isinstance(scenario, dict):
                continue
            metrics = scenario.get('rawMetrics', {})
            scenarios.append({'scenario': scenario.get('scenario', scenario.get('id')), 'rawMetrics': {k: metrics[k] for k in _MODEL_METRICS if isinstance(metrics, dict) and k in metrics}, 'normalizedUtility': scenario.get('normalizedUtility'), 'protectionTriggered': scenario.get('protectionTriggered'), 'replanSuggested': scenario.get('replanSuggested'), 'perDuty': [{k: duty[k] for k in ('dutyId', 'meanBurden', 'worst60mMean') if k in duty} for duty in scenario.get('dutyMetrics', []) or [] if isinstance(duty, dict)]})
        row['scenarios'] = scenarios
        compact['evaluations'].append(row)
    if not compact['evaluations'] and isinstance(data.get('evaluations'), list):
        compact['evaluations'] = data['evaluations']
    for key in ('pareto', 'ranked', 'rejected'):
        if key in data:
            compact[key] = data[key]
    return compact

def _compact_selection_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send enough exact provenance and metrics for ranking without time-series."""
    keep = ('id', 'candidateId', 'proposalId', 'name', 'validity', 'hard_valid', 'eligible', 'recommendable', 'notRecommendableReason', 'evaluation_id', 'evaluationId', 'inputSnapshotVersion', 'snapshot_version', 'intentRevision', 'base_plan_revision', 'basePlanRevision', 'sleepBlocks', 'unmetIntentIds', 'assumptions', 'rawMetrics', 'scenarios')
    return [{k: c[k] for k in keep if k in c} for c in candidates]

def _redact(text: str, secret: str | None=None) -> str:
    if secret:
        text = text.replace(secret, '[REDACTED]')
    text = re.sub('(?i)(authorization\\s*[:=]\\s*bearer\\s+)\\S+', '\\1[REDACTED]', text)
    text = re.sub('\\bsk-[A-Za-z0-9_-]{8,}\\b', '[REDACTED]', text)
    return text

def _is_auth_or_model_permission_error(message: str) -> bool:
    return bool(re.search('(?i)(invalid\\s+(?:api\\s+)?key|api\\s+key.{0,20}(invalid|unauthori[sz]ed)|model.{0,35}(permission|unauthori[sz]ed|not available|access denied|not allowed|forbidden|not found|does not exist|unsupported)|(?:permission|unauthori[sz]ed|access denied).{0,35}model|权限|无权|鉴权|密钥.{0,15}(无效|错误)|模型.{0,20}(无权限|不可用|不存在)|无权限.{0,15}模型)', message))

def _json_from_text(content: str) -> dict[str, Any]:
    """Parse one JSON object, tolerating a single fenced JSON block."""
    value = content.strip()
    if value.startswith('```'):
        value = re.sub('^```(?:json)?\\s*|\\s*```$', '', value, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        error = SchemaError(f'model response was not valid JSON at character {exc.pos}')
        error.raw_output = content
        raise error from None
    if not isinstance(parsed, dict):
        raise SchemaError('model response must be a JSON object')
    return parsed

def _require_keys(obj: dict[str, Any], required: set[str], allowed: set[str], label: str) -> None:
    missing = required - obj.keys()
    extra = obj.keys() - allowed
    if missing or extra:
        parts = []
        if missing:
            parts.append('missing keys: ' + ', '.join(sorted(missing)))
        if extra:
            parts.append('unsupported keys: ' + ', '.join(sorted(extra)))
        raise SchemaError(f'{label}: ' + '; '.join(parts))

@dataclass
class WorkflowResult:
    intent: dict[str, Any] | None = None
    strategies: dict[str, Any] | None = None
    patch: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evaluations: Any = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    status: str = 'incomplete'
    snapshot_version: str = ''
    intent_revision: int = 0
    base_plan_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {'status': self.status, 'snapshot_version': self.snapshot_version, 'intent_revision': self.intent_revision, 'base_plan_revision': self.base_plan_revision, 'intent': self.intent, 'strategies': self.strategies, 'patch': self.patch, 'selection': self.selection, 'candidates': self.candidates, 'evaluations': self.evaluations, 'attempts': self.attempts, 'usage': self.usage, 'elapsed_ms': self.elapsed_ms, 'errors': self.errors}

class MiniMaxPlanner:
    """Calls MiniMax China OpenAI-compatible Chat Completions with M3 only."""

    def __init__(self, *, results_dir: str | Path='results/product-validation/llm', api_key: str | None=None, timeout: float=DEFAULT_TIMEOUT_SECONDS, output_tokens: int=DEFAULT_OUTPUT_TOKENS) -> None:
        self.results_dir = Path(results_dir)
        self.api_key = api_key if api_key is not None else os.environ.get('MINIMAX_API_KEY')
        self.timeout = timeout
        if not 1 <= output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f'output_tokens must be in [1, {MAX_OUTPUT_TOKENS}]')
        self.output_tokens = output_tokens
        self.run_id = uuid.uuid4().hex[:12]
        self.attempts: list[dict[str, Any]] = []
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self._run_path: Path | None = None

    @staticmethod
    def http_attempt_count() -> int:
        with _ATTEMPT_LOCK:
            return _HTTP_ATTEMPTS

    @staticmethod
    def reset_global_state_for_tests() -> None:
        global _HTTP_ATTEMPTS
        with _ATTEMPT_LOCK:
            _HTTP_ATTEMPTS = 0
        _AUTH_STOP.clear()

    def _append_log(self, row: dict[str, Any]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self._run_path is None:
            self._run_path = self.results_dir / f'run-{self.run_id}.jsonl'
        safe = json.loads(_redact(_canonical(row), self.api_key))
        with self._run_path.open('a', encoding='utf-8') as handle:
            handle.write(_canonical(safe) + '\n')

    def _reserve_attempt(self) -> int:
        global _HTTP_ATTEMPTS
        self.results_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.results_dir / 'attempt-budget.json'
        lock_path = self.results_dir / '.attempt-budget.lock'
        deadline = time.monotonic() + 30
        lock_fd: int | None = None
        while lock_fd is None:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LLMError('timed out locking the global HTTP attempt ledger') from None
                time.sleep(0.025)
        try:
            with _ATTEMPT_LOCK:
                state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'attempts': 0, 'stopped': False}
                if state.get('stopped') or _AUTH_STOP.is_set():
                    raise AuthorizationStopped('online calls stopped after an authentication/model-permission failure')
                count = int(state.get('attempts', 0))
                if MAX_HTTP_ATTEMPTS is not None and count >= MAX_HTTP_ATTEMPTS:
                    raise BudgetExceeded(f'global MiniMax HTTP attempt cap ({MAX_HTTP_ATTEMPTS}) reached')
                count += 1
                state['attempts'] = count
                tmp = state_path.with_suffix('.tmp')
                tmp.write_text(_canonical(state), encoding='utf-8')
                os.replace(tmp, state_path)
                _HTTP_ATTEMPTS = max(_HTTP_ATTEMPTS, count)
                return count
        finally:
            os.close(lock_fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _stop_authorization(self) -> None:
        _AUTH_STOP.set()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.results_dir / 'attempt-budget.json'
        lock_path = self.results_dir / '.attempt-budget.lock'
        deadline = time.monotonic() + 30
        lock_fd: int | None = None
        while lock_fd is None:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LLMError('timed out locking the global HTTP attempt ledger') from None
                time.sleep(0.025)
        try:
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'attempts': 0}
            state['stopped'] = True
            tmp = state_path.with_suffix('.tmp')
            tmp.write_text(_canonical(state), encoding='utf-8')
            os.replace(tmp, state_path)
        finally:
            os.close(lock_fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _http_json(self, stage: str, messages: list[dict[str, str]], *, replay: bool=False) -> dict[str, Any]:
        payload = {'model': MODEL, 'messages': messages, 'temperature': 1.0, 'top_p': 0.95, 'max_completion_tokens': self.output_tokens, 'stream': False, 'thinking': {'type': 'disabled'}}
        request_key = hashlib.sha256(_canonical(payload).encode('utf-8')).hexdigest()
        cache_path = self.results_dir / 'cache' / f'{request_key}.json'
        if replay:
            if not cache_path.exists():
                raise ReplayMiss(f'no recorded response for request hash {request_key}')
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            self._append_log({'event': 'replay', 'stage': stage, 'request_hash': request_key, 'raw_request': payload, 'raw_response': cached})
            return cached
        if _AUTH_STOP.is_set():
            raise AuthorizationStopped('online calls stopped after an authentication/model-permission failure')
        if not self.api_key:
            raise LLMError('MINIMAX_API_KEY is not set; no HTTP request sent')
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        raw_request = urllib.request.Request(BASE_URL, data=_canonical(payload).encode('utf-8'), headers=headers, method='POST')
        started = time.monotonic()
        retries = 0
        while True:
            attempt_number = self._reserve_attempt()
            attempt_started = time.monotonic()
            status_code: int | None = None
            raw_response: dict[str, Any] | None = None
            error: str | None = None
            with _HTTP_SLOTS:
                try:
                    with urllib.request.urlopen(raw_request, timeout=self.timeout) as response:
                        status_code = response.status
                        raw_response = json.loads(response.read().decode('utf-8'))
                except urllib.error.HTTPError as exc:
                    status_code = exc.code
                    body = exc.read().decode('utf-8', errors='replace')
                    error = _redact(body, self.api_key)[:1500]
                    if status_code in (401, 403):
                        self.attempts.append({'stage': stage, 'attempt': attempt_number, 'status_code': status_code, 'elapsed_ms': int((time.monotonic() - attempt_started) * 1000), 'error': f'authorization HTTP {status_code}'})
                        self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': status_code, 'error': error, 'attempt': attempt_number})
                        self._stop_authorization()
                        raise AuthorizationStopped(f'provider returned HTTP {status_code}; all online experiments stopped') from None
                    if _is_auth_or_model_permission_error(error):
                        self.attempts.append({'stage': stage, 'attempt': attempt_number, 'status_code': status_code, 'elapsed_ms': int((time.monotonic() - attempt_started) * 1000), 'error': 'model permission failure'})
                        self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': status_code, 'error': error, 'attempt': attempt_number})
                        self._stop_authorization()
                        raise AuthorizationStopped(f'provider rejected model access with HTTP {status_code}; all online experiments stopped') from None
                    if status_code == 429 and retries < MAX_RETRIES:
                        self.attempts.append({'stage': stage, 'attempt': attempt_number, 'status_code': status_code, 'elapsed_ms': int((time.monotonic() - attempt_started) * 1000), 'error': 'rate limited; retrying'})
                        retries += 1
                        time.sleep(min(0.5 * 2 ** (retries - 1), 2.0))
                        continue
                    self.attempts.append({'stage': stage, 'attempt': attempt_number, 'status_code': status_code, 'elapsed_ms': int((time.monotonic() - attempt_started) * 1000), 'error': f'provider HTTP {status_code}'})
                    self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': status_code, 'error': error, 'attempt': attempt_number})
                    raise LLMError(f'provider HTTP {status_code}: {error}') from None
                except (TimeoutError, urllib.error.URLError) as exc:
                    reason = getattr(exc, 'reason', exc)
                    error = _redact(str(reason), self.api_key)[:500]
                    self.attempts.append({'stage': stage, 'attempt': attempt_number, 'status_code': None, 'elapsed_ms': int((time.monotonic() - attempt_started) * 1000), 'error': 'transport failure'})
                    self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': None, 'error': error, 'attempt': attempt_number})
                    if retries < MAX_RETRIES:
                        retries += 1
                        time.sleep(min(0.5 * 2 ** (retries - 1), 2.0))
                        continue
                    raise LLMError(f'provider transport failure after {retries} retries: {error}') from None
            elapsed = int((time.monotonic() - started) * 1000)
            self.attempts.append({'stage': stage, 'attempt': attempt_number, 'status_code': status_code, 'elapsed_ms': int((time.monotonic() - attempt_started) * 1000), 'error': None})
            if not isinstance(raw_response, dict):
                raise LLMError('provider returned a non-object JSON response')
            base_resp = raw_response.get('base_resp')
            if isinstance(base_resp, dict) and base_resp.get('status_code') not in (None, 0, '0'):
                provider_error = _redact(str(base_resp.get('status_msg', 'provider request failed')), self.api_key)
                if _is_auth_or_model_permission_error(provider_error):
                    self.attempts[-1]['error'] = 'model permission failure'
                    self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': status_code, 'error': provider_error, 'attempt': attempt_number})
                    self._stop_authorization()
                    raise AuthorizationStopped('provider rejected model access; all online experiments stopped') from None
                self.attempts[-1]['error'] = 'provider application error'
                self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': status_code, 'error': provider_error, 'attempt': attempt_number})
                raise LLMError(f"provider application error {base_resp.get('status_code')}: {provider_error}") from None
            provider_error_obj = raw_response.get('error')
            if provider_error_obj:
                provider_error = _redact(_canonical(provider_error_obj), self.api_key)
                self.attempts[-1]['error'] = 'provider error response'
                self._append_log({'event': 'http_error', 'stage': stage, 'request_hash': request_key, 'status_code': status_code, 'error': provider_error, 'attempt': attempt_number})
                if _is_auth_or_model_permission_error(provider_error):
                    self._stop_authorization()
                    raise AuthorizationStopped('provider rejected credentials or model access; all online experiments stopped') from None
                raise LLMError('provider returned an error object') from None
            self._append_log({'event': 'http_attempt', 'stage': stage, 'request_hash': request_key, 'raw_request': payload, 'raw_response': raw_response, 'status_code': status_code, 'attempts_including_retries': retries + 1, 'elapsed_ms': elapsed})
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(_redact(_canonical(raw_response), self.api_key), encoding='utf-8')
            return raw_response

    def _complete(self, stage: str, system: str, body: dict[str, Any], *, replay: bool=False) -> dict[str, Any]:
        response = self._http_json(stage, [{'role': 'system', 'content': system}, {'role': 'user', 'content': _canonical(body)}], replay=replay)
        choices = response.get('choices')
        if not isinstance(choices, list) or not choices:
            raise SchemaError('provider response has no choices')
        message = choices[0].get('message', {})
        content = message.get('content')
        if not isinstance(content, str) or not content.strip():
            raise SchemaError('provider response has no text content')
        usage = response.get('usage') or {}
        for key, source in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens'), ('total_tokens', 'total_tokens')):
            if isinstance(usage.get(source), int):
                self.usage[key] += usage[source]
        parsed = _json_from_text(content)
        self._append_log({'event': 'parsed_stage', 'stage': stage, 'result': parsed, 'usage': usage})
        return parsed

    def _complete_validated(self, stage: str, system: str, body: dict[str, Any], validator: Callable[[dict[str, Any]], Any], *, replay: bool=False) -> Any:
        """Retry a stage only to repair output format/schema, never to invent a fallback."""
        response: dict[str, Any] | None = None
        invalid_output: Any = None
        last_error: SchemaError | None = None
        current_system = system
        current_body = body
        for repair_index in range(MAX_SCHEMA_REPAIRS_PER_STAGE + 1):
            try:
                response = self._complete(stage if repair_index == 0 else f'{stage}_repair_{repair_index}', current_system, current_body, replay=replay)
            except SchemaError as exc:
                invalid_output = getattr(exc, 'raw_output', '')
                validation_error = str(exc)
                last_error = exc
            else:
                try:
                    return validator(response)
                except SchemaError as exc:
                    invalid_output = response
                    validation_error = str(exc)
                    last_error = exc
            if repair_index >= MAX_SCHEMA_REPAIRS_PER_STAGE:
                error_type = type(last_error) if last_error is not None else SchemaError
                raise error_type(f'{stage} remained invalid after {MAX_SCHEMA_REPAIRS_PER_STAGE} schema repairs: {validation_error}') from None
            current_system = "Repair the prior output for this same task. Return exactly one valid JSON object and nothing else. Use only the requested root keys; never echo a schema, example, markdown fence, explanation, or extra field. Echo every snapshot/version/revision field exactly from original_task, including base_plan_revision, snapshot_version, and intent_revision when requested. Preserve the user's intended meaning. Do not add facts, normalize or translate quoted/time text, infer a fixed activity, or change any ID/version except to copy the exact current values supplied. Fix every listed validation error."
            current_body = {'task': f'repair_{stage}', 'original_task': body, 'invalid_output': invalid_output, 'validator_error': validation_error, 'required_root_keys_only': sorted(body.get('schema', {}).keys()) if isinstance(body.get('schema'), dict) else [], 'repair_attempt': repair_index + 1, 'maximum_repairs': MAX_SCHEMA_REPAIRS_PER_STAGE}
        raise AssertionError('unreachable')

    def extract_intent(self, user_text: str, snapshot_version: str, intent_revision: int, *, replay: bool=False) -> dict[str, Any]:
        system = 'Extract only user-stated planning intent. The JSON object must have exactly these root keys: snapshot_version, intent_revision, items, clarifying_questions. Echo snapshot_version and intent_revision exactly from the request; do not increment or reinterpret them. Never output a schema key or schema description.\n\nCopy quote and time_hint as exact, case-sensitive substrings of user_text; time_hint must be null when no exact time/date phrase is present. Never normalize, translate, reformat, or add a time. Treat every wish/preference as soft and negotiable, including goals and requests to keep a time free. Only explicit quoted language such as "already booked, cannot move" can become a fixed confirmed_activity. Goal names or restatements are not activities. If a relative day is ambiguous, keep it as a soft wish and ask a clarification; do not convert the clarification into a confirmed activity.\n\nReturn valid JSON only, with no markdown or trailing characters. Shape example (replace placeholders with exact request values and copied source text):\n{"snapshot_version":"COPY_EXACT_SNAPSHOT_VERSION","intent_revision":1,"items":[{"id":"intent-1","quote":"想下午喝咖啡","kind":"wish","time_hint":"下午","certainty":"soft","negotiable":true,"needs_clarification":false}],"clarifying_questions":[]}\nThe only allowed item keys are id, quote, kind, time_hint, certainty, negotiable, needs_clarification. Do not include the schema in the output.'
        body = {'task': 'intent', 'user_text': user_text, 'snapshot_version': snapshot_version, 'intent_revision': intent_revision, 'schema': {'snapshot_version': 'string', 'intent_revision': 'integer', 'items': [{'id': 'string', 'quote': 'string', 'kind': 'wish|confirmed_activity|preference|ambiguity', 'time_hint': 'string|null', 'certainty': 'soft|fixed|unclear', 'negotiable': 'boolean', 'needs_clarification': 'boolean'}], 'clarifying_questions': ['string']}}

        def validate(result: dict[str, Any]) -> dict[str, Any]:
            _require_keys(result, {'snapshot_version', 'intent_revision', 'items', 'clarifying_questions'}, {'snapshot_version', 'intent_revision', 'items', 'clarifying_questions'}, 'intent')
            if result['snapshot_version'] != snapshot_version or result['intent_revision'] != intent_revision:
                raise StaleVersionError('intent response must echo the exact current snapshot_version and intent_revision')
            if not isinstance(result['items'], list) or not isinstance(result['clarifying_questions'], list):
                raise SchemaError('intent items and clarifying_questions must be arrays')
            allowed_kinds = {'wish', 'confirmed_activity', 'preference', 'ambiguity'}
            allowed_certainty = {'soft', 'fixed', 'unclear'}
            seen_intent_ids: set[str] = set()
            for item in result['items']:
                if not isinstance(item, dict):
                    raise SchemaError('each intent item must be an object')
                _require_keys(item, {'id', 'quote', 'kind', 'time_hint', 'certainty', 'negotiable', 'needs_clarification'}, {'id', 'quote', 'kind', 'time_hint', 'certainty', 'negotiable', 'needs_clarification'}, 'intent item')
                if item['kind'] not in allowed_kinds or item['certainty'] not in allowed_certainty:
                    raise SchemaError('intent kind/certainty is outside the whitelist')
                if not isinstance(item['quote'], str) or not isinstance(item['negotiable'], bool) or (not isinstance(item['needs_clarification'], bool)):
                    raise SchemaError('intent quote/flags have invalid types')
                if not item['id'] or item['id'] in seen_intent_ids:
                    raise SchemaError('intent IDs must be nonempty and unique')
                seen_intent_ids.add(item['id'])
                if not item['quote'] or item['quote'] not in user_text:
                    raise SchemaError('intent quote must be copied verbatim from the user text')
                if item['time_hint'] is not None and (not isinstance(item['time_hint'], str) or item['time_hint'] not in user_text):
                    raise SchemaError('intent time_hint must be copied verbatim from the user text or be null')
                if item['kind'] == 'confirmed_activity':
                    fixed_cues = ('已确定', '已订好', '已经订好', '已经安排', '已安排', '不能改', '不要改', '已约', 'already booked', 'confirmed', 'fixed', 'cannot move', "don't move", 'do not move', 'scheduled')
                    if item['certainty'] != 'fixed' or item['negotiable'] or (not any((cue.casefold() in item['quote'].casefold() for cue in fixed_cues))):
                        raise SchemaError('confirmed activities require a quoted confirmation cue and fixed, nonnegotiable status')
                if item['certainty'] == 'fixed' and item['kind'] != 'confirmed_activity':
                    raise SchemaError('only explicitly confirmed activities may be marked fixed')
            return result
        return self._complete_validated('intent', system, body, validate, replay=replay)

    def propose_strategies(self, intent: dict[str, Any], candidate_facts: list[dict[str, Any]], base_plan_revision: int, *, replay: bool=False) -> dict[str, Any]:
        system = 'Suggest 3-5 exploration strategies over only the supplied candidate and intent IDs. Return a JSON object with exactly root keys base_plan_revision and strategies. Copy base_plan_revision exactly from the request. Each strategy must include all required keys. Never invent/edit timestamps, scores, medical claims, IDs, or candidate facts. Do not echo schema. Return valid JSON only.\nExample shape: {"base_plan_revision":1,"strategies":[{"id":"strategy-1","candidate_ids":["COPY_SUPPLIED_CANDIDATE_ID"],"preserves_intent_ids":["COPY_SUPPLIED_INTENT_ID"],"tradeoff":"plain-language tradeoff","exploration_request":"bounded variation to evaluate"}]}'
        body = {'task': 'strategies', 'intent': intent, 'candidate_facts': candidate_facts, 'base_plan_revision': base_plan_revision, 'schema': {'base_plan_revision': 'integer', 'strategies': [{'id': 'string', 'candidate_ids': ['existing candidate id'], 'preserves_intent_ids': ['string'], 'tradeoff': 'string', 'exploration_request': 'string'}]}}

        def validate(result: dict[str, Any]) -> dict[str, Any]:
            _require_keys(result, {'base_plan_revision', 'strategies'}, {'base_plan_revision', 'strategies'}, 'strategies')
            if result['base_plan_revision'] != base_plan_revision:
                raise StaleVersionError('strategies response must echo the exact current base_plan_revision')
            if not isinstance(result['strategies'], list) or not 1 <= len(result['strategies']) <= 5:
                raise SchemaError('strategies must contain 1-5 entries')
            ids = {str(c.get('id')) for c in candidate_facts}
            intent_ids = {str(i.get('id')) for i in intent.get('items', [])}
            for strategy in result['strategies']:
                _require_keys(strategy, {'id', 'candidate_ids', 'preserves_intent_ids', 'tradeoff', 'exploration_request'}, {'id', 'candidate_ids', 'preserves_intent_ids', 'tradeoff', 'exploration_request'}, 'strategy')
                if not isinstance(strategy['candidate_ids'], list) or not isinstance(strategy['preserves_intent_ids'], list):
                    raise SchemaError('strategy candidate and intent references must be arrays')
                if not all((isinstance(x, str) for x in strategy['candidate_ids'] + strategy['preserves_intent_ids'])):
                    raise SchemaError('strategy candidate and intent references must be string IDs')
                if not all((isinstance(strategy[key], str) for key in ('id', 'tradeoff', 'exploration_request'))):
                    raise SchemaError('strategy text fields must be strings')
                if not set(strategy['candidate_ids']) <= ids or not set(strategy['preserves_intent_ids']) <= intent_ids:
                    raise SchemaError('strategy references an unknown candidate or intent ID')
            return result
        return self._complete_validated('strategies', system, body, validate, replay=replay)

    def propose_patch(self, intent: dict[str, Any], strategy: dict[str, Any], candidate_facts: list[dict[str, Any]], evaluations: Any, base_plan_revision: int, *, replay: bool=False) -> dict[str, Any]:
        system = 'Propose at most two small patches to supplied candidates. Return exactly the root keys base_plan_revision and patches; echo base_plan_revision exactly. Each patch item has candidate_id, changes, reason. Only change allowlisted candidate fields. Any replacement sleep block must be copied exactly from a supplied candidate block, preserving its full range and timeZone. Never invent a timestamp, block, score, or constraint result. Do not echo schema. Return valid JSON only.\nExample shape: {"base_plan_revision":1,"patches":[{"candidate_id":"COPY_SUPPLIED_CANDIDATE_ID","changes":{"name":"optional label"},"reason":"specific intent tradeoff"}]}'
        body = {'task': 'patch', 'intent': intent, 'strategy': strategy, 'candidate_facts': candidate_facts, 'evaluations': _compact_evaluation_report(evaluations), 'base_plan_revision': base_plan_revision, 'schema': {'base_plan_revision': 'integer', 'patches': [{'candidate_id': 'existing ID', 'changes': {'sleepBlocks': 'copied existing blocks only', 'unmetIntentIds': 'existing intent IDs', 'assumptions': ['string'], 'name': 'string'}, 'reason': 'string'}]}}

        def validate(result: dict[str, Any]) -> dict[str, Any]:
            _require_keys(result, {'base_plan_revision', 'patches'}, {'base_plan_revision', 'patches'}, 'patch')
            if result['base_plan_revision'] != base_plan_revision:
                raise StaleVersionError('patch response must echo the exact current base_plan_revision')
            ids = {str(c.get('id')) for c in candidate_facts}
            intent_ids = {str(i.get('id')) for i in intent.get('items', [])}
            if not isinstance(result['patches'], list) or len(result['patches']) > 2:
                raise SchemaError('patches must be an array of at most 2')
            for patch in result['patches']:
                _require_keys(patch, {'candidate_id', 'changes', 'reason'}, {'candidate_id', 'changes', 'reason'}, 'patch item')
                if patch['candidate_id'] not in ids or not isinstance(patch['changes'], dict):
                    raise SchemaError('patch references unknown candidate or changes is not an object')
                if not isinstance(patch['reason'], str):
                    raise SchemaError('patch reason must be a string')
                changes = patch['changes']
                allowed_changes = {'sleepBlocks', 'unmetIntentIds', 'assumptions', 'name'}
                if changes.keys() - allowed_changes:
                    raise SchemaError('patch contains non-allowlisted fields')
                if 'sleepBlocks' in changes:
                    if not isinstance(changes['sleepBlocks'], list):
                        raise SchemaError('sleepBlocks patch must be an array')
                    known_blocks = {_canonical(block) for c in candidate_facts for block in c.get('sleepBlocks', [])}
                    if any((_canonical(block) not in known_blocks for block in changes['sleepBlocks'])):
                        raise SchemaError('patch contains a new or modified time block not present in supplied candidate facts')
                if 'unmetIntentIds' in changes and (not isinstance(changes['unmetIntentIds'], list) or not set(changes['unmetIntentIds']) <= intent_ids):
                    raise SchemaError('patch references unknown intent IDs')
                if 'assumptions' in changes and (not isinstance(changes['assumptions'], list) or not all((isinstance(x, str) for x in changes['assumptions']))):
                    raise SchemaError('patch assumptions must be strings')
                if 'name' in changes and (not isinstance(changes['name'], str)):
                    raise SchemaError('patch name must be a string')
            return result
        return self._complete_validated('patch', system, body, validate, replay=replay)

    def select_candidates(self, intent: dict[str, Any], evaluated_candidates: list[dict[str, Any]], base_plan_revision: int, snapshot_version: str, *, replay: bool=False) -> dict[str, Any]:
        system = 'Choose 1-3 candidates only from the supplied freshly evaluated and recommendable set. If none is recommendable, return selected:[] and explain the missing assumption or constraint in negotiation. Return exactly root keys base_plan_revision, snapshot_version, selected, negotiation. Copy both version fields exactly. Every selected item must cite an exact candidate_id and that candidate\'s current evaluation_id; never invent IDs, scores, or facts. Selection is advisory and must pass local validation. Do not echo schema. Return valid JSON only.\nExample shape: {"base_plan_revision":1,"snapshot_version":"COPY_EXACT_SNAPSHOT_VERSION","selected":[{"candidate_id":"COPY_SUPPLIED_ID","evaluation_id":"COPY_CURRENT_EVALUATION_ID","reason":"specific supplied metric and intent","tradeoffs":[]}],"negotiation":null}'
        body = {'task': 'selection', 'intent': intent, 'evaluated_candidates': _compact_selection_facts(evaluated_candidates), 'base_plan_revision': base_plan_revision, 'snapshot_version': snapshot_version, 'schema': {'base_plan_revision': 'integer', 'snapshot_version': 'string', 'selected': [{'candidate_id': 'existing ID', 'evaluation_id': 'current evaluation ID', 'reason': 'string', 'tradeoffs': ['string']}], 'negotiation': 'string|null'}}

        def validate(result: dict[str, Any]) -> dict[str, Any]:
            _require_keys(result, {'base_plan_revision', 'snapshot_version', 'selected', 'negotiation'}, {'base_plan_revision', 'snapshot_version', 'selected', 'negotiation'}, 'selection')
            if result['base_plan_revision'] != base_plan_revision or result['snapshot_version'] != snapshot_version:
                raise StaleVersionError('selection response must echo exact current base_plan_revision and snapshot_version')
            if not isinstance(result['selected'], list) or len(result['selected']) > 3:
                raise SchemaError('selection must contain 0-3 candidates')
            if not result['selected'] and (not isinstance(result['negotiation'], str) or not result['negotiation'].strip()):
                raise SchemaError('empty selection requires a negotiation reason')
            if result['selected'] and result['negotiation'] is not None:
                raise SchemaError('negotiation is only allowed when selection is empty')
            candidates = {_candidate_id(c): c for c in evaluated_candidates}
            selected_ids: set[str] = set()
            for item in result['selected']:
                _require_keys(item, {'candidate_id', 'evaluation_id', 'reason', 'tradeoffs'}, {'candidate_id', 'evaluation_id', 'reason', 'tradeoffs'}, 'selection item')
                cid = str(item['candidate_id'])
                if cid not in candidates or cid in selected_ids:
                    raise SchemaError('selection includes unknown or duplicate candidate')
                candidate = candidates[cid]
                current_evaluation_id = candidate.get('evaluation_id', candidate.get('evaluationId'))
                if current_evaluation_id != item['evaluation_id']:
                    raise StaleVersionError(f'selection evaluation for {cid} is stale')
                if not isinstance(current_evaluation_id, str) or not current_evaluation_id:
                    raise StaleVersionError(f'candidate {cid} has no evaluation ID')
                if candidate.get('inputSnapshotVersion', candidate.get('snapshot_version')) != snapshot_version:
                    raise StaleVersionError(f'candidate {cid} is based on stale snapshot version')
                if candidate.get('intentRevision') != intent.get('intent_revision'):
                    raise StaleVersionError(f'candidate {cid} is based on stale intent revision')
                if candidate.get('base_plan_revision', candidate.get('basePlanRevision')) != base_plan_revision:
                    raise StaleVersionError(f'candidate {cid} is based on stale plan revision')
                if not isinstance(item['reason'], str) or not isinstance(item['tradeoffs'], list) or (not all((isinstance(x, str) for x in item['tradeoffs']))):
                    raise SchemaError('selection reason and tradeoffs have invalid types')
                if candidate.get('hard_valid') is False or candidate.get('eligible') is False:
                    raise SchemaError(f'selection includes ineligible candidate {cid}')
                if candidate.get('recommendable') is not True:
                    raise SchemaError(f'selection includes candidate {cid} without explicit recommendable status')
                selected_ids.add(cid)
            return result
        return self._complete_validated('selection', system, body, validate, replay=replay)

    @staticmethod
    def _flatten_evaluated(report: Any, fallback_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        data = report.to_dict() if hasattr(report, 'to_dict') else report
        if isinstance(data, dict) and isinstance(data.get('candidates'), list):
            evaluations = data.get('evaluations', [])
            by_id: dict[str, Any] = {}
            if isinstance(evaluations, list):
                for evaluation in evaluations:
                    if isinstance(evaluation, dict):
                        by_id[str(evaluation.get('candidate_id', evaluation.get('candidateId', evaluation.get('proposal_id', evaluation.get('proposalId', '')))))] = evaluation
            out = []
            for candidate in data['candidates']:
                row = dict(candidate)
                ev = by_id.get(_candidate_id(candidate))
                if isinstance(ev, dict):
                    row['evaluation_id'] = ev.get('evaluation_id', ev.get('evaluationId', ev.get('id')))
                    row['robustUtility'] = ev.get('robustUtility')
                    row['scenarios'] = _compact_evaluation_report({'evaluations': [ev]}).get('evaluations', [{}])[0].get('scenarios', [])
                    row['hard_valid'] = ev.get('hard_valid', candidate.get('hard_valid', True))
                    row['recommendable'] = ev.get('recommendable', candidate.get('recommendable', False))
                    row['eligible'] = row['recommendable'] is True and candidate.get('validity') == 'validated'
                    row['inputSnapshotVersion'] = ev.get('inputSnapshotVersion', candidate.get('inputSnapshotVersion'))
                    row['intentRevision'] = ev.get('intentRevision', candidate.get('intentRevision', data.get('intentRevision')))
                    row['basePlanRevision'] = ev.get('basePlanRevision', candidate.get('basePlanRevision', data.get('basePlanRevision')))
                out.append(row)
            for row in out:
                row.setdefault('evaluation_id', row.get('evaluationId'))
            return out
        if isinstance(data, list):
            return data
        return fallback_candidates

    def run_workflow(self, *, user_text: str, candidate_facts: list[dict[str, Any]], snapshot_version: str, intent_revision: int, base_plan_revision: int, evaluate: Callable[..., Any], evaluate_patch: Callable[..., Any], level: str='E4', replay: bool=False) -> WorkflowResult:
        start = time.monotonic()
        result = WorkflowResult(snapshot_version=snapshot_version, intent_revision=intent_revision, base_plan_revision=base_plan_revision)
        try:
            result.intent = self.extract_intent(user_text, snapshot_version, intent_revision, replay=replay)
            result.strategies = self.propose_strategies(result.intent, candidate_facts, base_plan_revision, replay=replay)
            result.candidates = list(candidate_facts)
            baseline = evaluate(candidates=result.candidates, level=level)
            baseline_data = baseline.to_dict() if hasattr(baseline, 'to_dict') else baseline
            result.evaluations = {'baseline': baseline_data, 'patches': []}
            evaluated = self._flatten_evaluated(baseline, result.candidates)
            strategy = result.strategies['strategies'][0]
            result.patch = self.propose_patch(result.intent, strategy, result.candidates, baseline_data, base_plan_revision, replay=replay)
            patched_rows: list[dict[str, Any]] = []
            for patch_item in result.patch['patches']:
                candidate = next((c for c in result.candidates if _candidate_id(c) == patch_item['candidate_id']))
                if not patch_item['changes']:
                    continue
                reevaluated = evaluate_patch(candidate, patch_item['changes'], level=level)
                reevaluated_data = reevaluated.to_dict() if hasattr(reevaluated, 'to_dict') else reevaluated
                result.evaluations['patches'].append(reevaluated_data)
                if isinstance(reevaluated_data, dict):
                    if 'candidates' in reevaluated_data and (not reevaluated_data['candidates']):
                        continue
                    reevaluated_rows = self._flatten_evaluated(reevaluated_data, [candidate])
                    if not reevaluated_rows:
                        continue
                    merged = reevaluated_rows[0]
                    if not merged.get('evaluation_id'):
                        merged['evaluation_id'] = reevaluated_data.get('evaluation_id', reevaluated_data.get('id'))
                    merged['hard_valid'] = reevaluated_data.get('hard_valid', merged.get('hard_valid', merged.get('validity') != 'invalid'))
                    merged['recommendable'] = reevaluated_data.get('recommendable', merged.get('recommendable', False))
                    merged['base_plan_revision'] = base_plan_revision
                    patched_rows.append(merged)
            selectable_by_id = {_candidate_id(c): c for c in evaluated}
            for patched in patched_rows:
                selectable_by_id[_candidate_id(patched)] = patched
            selectable = list(selectable_by_id.values())
            result.selection = self.select_candidates(result.intent, selectable, base_plan_revision, snapshot_version, replay=replay)
            selected_ids = {x['candidate_id'] for x in result.selection['selected']}
            result.candidates = [c for c in selectable if _candidate_id(c) in selected_ids]
            result.status = 'complete' if result.selection['selected'] else 'negotiation_required'
        except Exception as exc:
            msg = _redact(str(exc), self.api_key)
            result.errors.append({'type': type(exc).__name__, 'message': msg})
            result.status = 'failed'
        finally:
            result.attempts = list(self.attempts)
            result.usage = dict(self.usage)
            result.elapsed_ms = int((time.monotonic() - start) * 1000)
            self._append_log({'event': 'workflow_summary', **result.to_dict()})
        return result

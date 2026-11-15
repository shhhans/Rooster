"""MiniMax chat-completion client for roster vision parsing.

Fixed provider semantics (ROSTER-PARSE-01): the business model is
MiniMax-M3.1-Flash. Endpoint/model/reasoning-effort choices are explicit
configuration 鈥?no silent fallback, no automatic endpoint or model change,
no text-only degrade. A failed call raises a typed error and each real HTTP
attempt is surfaced to the caller. The API key is read only from the
environment (MINIMAX_API_KEY) and is never logged, returned, or persisted.

Streaming requests are consumed with a real ``client.stream(...)`` inside
the context (verified against the buffering defect that produced the
attempt #4鈥?6 failures): a plain ``client.post`` buffers the whole body
before the caller sees it, so raw SSE capture and first-line timing are only
possible with true streaming. On the OpenAI-compatible path
(``/chat/completions``) ``stream_options.include_usage`` is sent so the
final usage chunk is returned (its absence explains the missing usage in
attempt #6). HTTP 200 + ``application/json`` bodies are treated as platform
error/answer payloads, never skipped as empty SSE.

Config notes (official MiniMax docs, see docs/roster-parser.md):
* ``max_tokens`` is deprecated in favor of ``max_completion_tokens``.
* ``reasoning_effort`` defaults to ``max`` on the platform when unset.
"""
from __future__ import annotations
import base64
import json
import os
import time
from dataclasses import dataclass, field
import httpx
from app.roster_parser.errors import MissingApiKeyError, ProviderConfigError, ProviderError, ProviderTimeoutError
DEFAULT_BASE_URL = 'https://api.minimaxi.com/v1'
DEFAULT_MODEL = 'MiniMax-M3.1-Flash'
CHAT_COMPLETIONS_PATH = '/text/chatcompletion_v2'
OPENAI_CHAT_COMPLETIONS_PATH = '/chat/completions'
VALID_REASONING_EFFORTS = {'low', 'medium', 'high', 'xhigh', 'max'}
VALID_ENDPOINTS = {'chatcompletion_v2', 'chat_completions'}

@dataclass(frozen=True)
class ProviderConfig:
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float
    max_completion_tokens: int = 16384
    stream: bool = True
    reasoning_effort: str | None = None
    endpoint: str = 'chatcompletion_v2'

    def __post_init__(self) -> None:
        if self.endpoint not in VALID_ENDPOINTS:
            raise ProviderConfigError(f'endpoint {self.endpoint!r} is invalid (MINIMAX_ENDPOINT); expected one of {sorted(VALID_ENDPOINTS)}')
        if self.reasoning_effort is not None and self.reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ProviderConfigError(f'reasoning_effort {self.reasoning_effort!r} is invalid (MINIMAX_REASONING_EFFORT); expected one of {sorted(VALID_REASONING_EFFORTS)} (unset = platform default)')
        if self.max_completion_tokens <= 0:
            raise ProviderConfigError('max_completion_tokens must be positive (MINIMAX_MAX_COMPLETION_TOKENS)')

    @classmethod
    def from_env(cls) -> 'ProviderConfig':
        return cls(api_key=os.environ.get('MINIMAX_API_KEY'), base_url=os.environ.get('MINIMAX_BASE_URL', DEFAULT_BASE_URL).rstrip('/'), model=os.environ.get('MINIMAX_MODEL', DEFAULT_MODEL), timeout_seconds=_float_env('MINIMAX_TIMEOUT_SECONDS', 60.0), max_completion_tokens=_int_env('MINIMAX_MAX_COMPLETION_TOKENS', 16384), stream=_bool_env('MINIMAX_STREAM', True), reasoning_effort=_optional_env('MINIMAX_REASONING_EFFORT'), endpoint=_optional_env('MINIMAX_ENDPOINT', 'chatcompletion_v2'))

def _optional_env(name: str, default: str | None=None) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower()

def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}

@dataclass
class ProviderResult:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    status_code: int = 0
    elapsed_seconds: float = 0.0
    http_attempts: int = 1
    streamed: bool = False
    response_header_seconds: float | None = None
    first_line_seconds: float | None = None
    content_type: str | None = None
    trace_id: str | None = None
    finish_reason: str | None = None
    data_chunks: int = 0
    sse_lines: int = 0

class MiniMaxClient:
    """Synchronous vision chat-completion client (one HTTP attempt per call)."""

    def __init__(self, config: ProviderConfig | None=None, transport: httpx.BaseTransport | None=None) -> None:
        self._config = config or ProviderConfig.from_env()
        self._transport = transport

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def complete_vision(self, system_prompt: str, images: list[tuple[str, bytes]], user_text: str) -> ProviderResult:
        """Send one vision completion request with image inputs.

        ``images`` is a list of (mime_type, image_bytes) pairs, passed as
        base64 data URLs. Exactly one HTTP attempt is made; any failure is a
        typed error (no retries, no fallback model, no text-only degrade).
        """
        if not self._config.api_key:
            raise MissingApiKeyError('MINIMAX_API_KEY is not set in the environment; refusing to degrade to text-only parsing')
        path, payload = self._build_request(system_prompt, images, user_text)
        sent_payload = dict(payload)
        if self._config.stream:
            sent_payload['stream'] = True
            if self._config.endpoint == 'chat_completions':
                sent_payload['stream_options'] = {'include_usage': True}
        if self._config.stream:
            result = self._complete_streaming(path, payload)
        else:
            result = self._complete_once(path, payload)
        return result

    def _build_request(self, system_prompt: str, images: list[tuple[str, bytes]], user_text: str) -> tuple[str, dict]:
        content_parts = [{'type': 'text', 'text': user_text}]
        for mime_type, image_bytes in images:
            encoded = base64.b64encode(image_bytes).decode('ascii')
            content_parts.append({'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{encoded}'}})
        if self._config.endpoint == 'chat_completions':
            path = OPENAI_CHAT_COMPLETIONS_PATH
        else:
            path = CHAT_COMPLETIONS_PATH
        payload: dict = {'model': self._config.model, 'max_completion_tokens': self._config.max_completion_tokens, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content_parts}]}
        if self._config.reasoning_effort is not None:
            payload['reasoning_effort'] = self._config.reasoning_effort
        return (path, payload)

    def _complete_once(self, path: str, payload: dict) -> ProviderResult:
        started = time.monotonic()
        response = self._post_once(path, payload)
        header_seconds = time.monotonic() - started
        content_type = response.headers.get('content-type')
        trace_id = _trace_id_of(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f'MiniMax returned non-JSON body: {response.text[:300]}') from exc
        raise_for_platform_error(data)
        content = extract_content(data)
        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        model = data.get('model') if isinstance(data.get('model'), str) else self._config.model
        finish_reason = _finish_reason_of(data)
        if trace_id is None and isinstance(data.get('id'), str):
            trace_id = data['id']
        return ProviderResult(content=content, model=model, usage=usage, status_code=response.status_code, elapsed_seconds=header_seconds, http_attempts=1, streamed=False, response_header_seconds=header_seconds, content_type=content_type, trace_id=trace_id, finish_reason=finish_reason)

    def _post_once(self, path: str, payload: dict) -> httpx.Response:
        url = f'{self._config.base_url}{path}'
        headers = _headers_of(self._config)
        try:
            with httpx.Client(timeout=self._config.timeout_seconds, transport=self._transport) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f'MiniMax request timed out after {self._config.timeout_seconds}s: {type(exc).__name__}') from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f'MiniMax request failed: {type(exc).__name__}: {exc}') from exc
        if response.status_code != 200:
            raise ProviderError(f'MiniMax returned HTTP {response.status_code}: {response.text[:300]}')
        return response

    def _complete_streaming(self, path: str, payload: dict) -> ProviderResult:
        stream_payload = dict(payload)
        stream_payload['stream'] = True
        if self._config.endpoint == 'chat_completions':
            stream_payload['stream_options'] = {'include_usage': True}
        url = f'{self._config.base_url}{path}'
        headers = _headers_of(self._config)
        started = time.monotonic()
        content_pieces: list[str] = []
        model: str | None = None
        usage: dict = {}
        finish_reason: str | None = None
        trace_id: str | None = None
        content_type: str | None = None
        stats = {'sse_lines': 0, 'data_chunks': 0, 'unparsed_chunks': 0, 'content_pieces': 0, 'reasoning_pieces': 0}
        first_line_seconds: float | None = None
        debug_handle = _open_sse_debug_handle()
        try:
            with httpx.Client(timeout=self._config.timeout_seconds, transport=self._transport) as client:
                with client.stream('POST', url, headers=headers, json=stream_payload) as response:
                    response_header_seconds = time.monotonic() - started
                    content_type = response.headers.get('content-type')
                    trace_id = _trace_id_of(response)
                    if response.status_code != 200:
                        body = response.read()
                        raise ProviderError(f'MiniMax returned HTTP {response.status_code}: {body[:300]!r}')
                    if content_type and 'application/json' in content_type.lower():
                        body = response.read()
                        try:
                            data = json.loads(body)
                        except ValueError as exc:
                            raise ProviderError(f'MiniMax returned non-JSON body: {body[:300]!r}') from exc
                        raise_for_platform_error(data)
                        content = extract_content(data)
                        elapsed = time.monotonic() - started
                        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
                        model = data.get('model') if isinstance(data.get('model'), str) else self._config.model
                        if trace_id is None and isinstance(data.get('id'), str):
                            trace_id = data['id']
                        return ProviderResult(content=content, model=model, usage=usage, status_code=200, elapsed_seconds=elapsed, http_attempts=1, streamed=False, response_header_seconds=response_header_seconds, content_type=content_type, trace_id=trace_id, finish_reason=_finish_reason_of(data))
                    for line in response.iter_lines():
                        if first_line_seconds is None:
                            first_line_seconds = time.monotonic() - started
                        stats['sse_lines'] += 1
                        if debug_handle is not None:
                            debug_handle.write(line + '\n')
                            debug_handle.flush()
                        if not line or not line.startswith('data:'):
                            continue
                        data_str = line[len('data:'):].strip()
                        if data_str == '[DONE]':
                            break
                        stats['data_chunks'] += 1
                        try:
                            chunk = json.loads(data_str)
                        except ValueError:
                            stats['unparsed_chunks'] += 1
                            continue
                        raise_for_platform_error(chunk)
                        if isinstance(chunk.get('model'), str):
                            model = chunk['model']
                        if isinstance(chunk.get('usage'), dict) and chunk['usage']:
                            usage = chunk['usage']
                        choices = chunk.get('choices') or []
                        if choices:
                            finish = choices[0].get('finish_reason')
                            if isinstance(finish, str) and finish != 'none':
                                finish_reason = finish
                            delta = choices[0].get('delta') or {}
                            piece = delta.get('content')
                            if isinstance(piece, str):
                                content_pieces.append(piece)
                                stats['content_pieces'] += 1
                            if isinstance(delta.get('reasoning_content'), str):
                                stats['reasoning_pieces'] += 1
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f'MiniMax stream timed out after {self._config.timeout_seconds}s: {type(exc).__name__}') from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f'MiniMax stream failed: {type(exc).__name__}: {exc}') from exc
        finally:
            if debug_handle is not None:
                debug_handle.close()
        elapsed = time.monotonic() - started
        content = ''.join(content_pieces)
        if not content.strip():
            if stats['content_pieces'] == 0 and stats['reasoning_pieces'] > 0:
                reason = 'reasoning_only_stream'
            else:
                reason = 'empty_completion'
            raise ProviderError(f'MiniMax stream produced no content (reason={reason}; ' + ', '.join((f'{key}={value}' for key, value in stats.items())) + f', finish_reason={finish_reason})')
        return ProviderResult(content=content, model=model or self._config.model, usage=usage, status_code=200, elapsed_seconds=elapsed, http_attempts=1, streamed=True, response_header_seconds=response_header_seconds, first_line_seconds=first_line_seconds, content_type=content_type, trace_id=trace_id, finish_reason=finish_reason, data_chunks=stats['data_chunks'], sse_lines=stats['sse_lines'])

def raise_for_platform_error(data: dict) -> None:
    """MiniMax reports platform errors with HTTP 200; the status field is
    "status_code" on api.minimaxi.com / api.minimax.cn ("status" on some
    legacy shapes)."""
    base_resp = data.get('base_resp') or {}
    base_status = base_resp.get('status_code', base_resp.get('status', 0))
    if base_status not in (0, None):
        raise ProviderError(f"MiniMax base_resp status {base_status}: {base_resp.get('status_msg', 'unknown error')}")

def extract_content(data: dict) -> str:
    choices = data.get('choices') or []
    if not choices:
        raise ProviderError('MiniMax response has no choices')
    message = choices[0].get('message') or {}
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise ProviderError('MiniMax response message has no text content')
    return content

def _headers_of(config: ProviderConfig) -> dict[str, str]:
    return {'Authorization': f'Bearer {config.api_key}', 'Content-Type': 'application/json'}

def _trace_id_of(response: httpx.Response) -> str | None:
    for name in ('trace-id', 'x-trace-id', 'x-request-id'):
        value = response.headers.get(name)
        if value:
            return value
    return None

def _finish_reason_of(data: dict) -> str | None:
    choices = data.get('choices') or []
    if choices and isinstance(choices[0].get('finish_reason'), str):
        return choices[0]['finish_reason']
    return None

def _open_sse_debug_handle():
    """Optional raw-SSE capture for diagnostics (MINIMAX_SSE_DEBUG_PATH).

    The capture file contains the full model output (which may include roster
    content), so it must be kept outside Git and outside logs. Only the
    operator-provided path is used; it is never set by default.
    """
    path = os.environ.get('MINIMAX_SSE_DEBUG_PATH')
    if not path:
        return None
    try:
        return open(path, 'a', encoding='utf-8')
    except OSError as exc:
        raise ProviderError(f'MINIMAX_SSE_DEBUG_PATH {path!r} cannot be opened for append: {exc}') from exc

def dump_payload_safely(payload: dict) -> str:
    """Debug helper that never leaks the API key into logs."""
    return json.dumps(payload, ensure_ascii=False)[:500]

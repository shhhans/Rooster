"""Black-box client abstraction for the ROOSTER-002 timeline kernel contract.

The frozen contract (ROOSTER-TIMELINE-CORE) is transport-agnostic. This client
binds to the concrete invocation surface at validation time via environment
variables, so the same pytest suite works regardless of how the Executor
exposes the interfaces:

  ROOSTER_IMPORT_PATH  - Python import path of the module exposing the five
                         callables (e.g. "app.timeline.api"), invoked directly.
  ROOSTER_API_BASE_URL - HTTP base URL of the FastAPI service (e.g.
                         "http://127.0.0.1:8000"), plus
                         ROOSTER_API_PATHS_JSON - optional JSON mapping of
                         interface name -> "METHOD /path" (defaults are
                         documented in README_AT_READINESS if provided).

If neither is set, the client raises a clear error telling the validator to
bind the surface at readiness.
"""

from __future__ import annotations

import importlib
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class ClientUnboundError(RuntimeError):
    """Raised when no invocation surface has been configured."""


class RoosterClient:
    """Thin transport wrapper over the five contract interfaces."""

    INTERFACES = (
        "ingest_schedule_event",
        "correct_schedule_event",
        "retract_schedule_event",
        "list_schedule_events",
        "get_schedule_event",
    )

    def __init__(self) -> None:
        self.import_path: Optional[str] = os.environ.get("ROOSTER_IMPORT_PATH")
        self.base_url: Optional[str] = os.environ.get("ROOSTER_API_BASE_URL")
        self._paths: Dict[str, str] = {}
        if self.base_url:
            raw_paths = os.environ.get("ROOSTER_API_PATHS_JSON")
            if raw_paths:
                self._paths = json.loads(raw_paths)
        self._module: Optional[Any] = None

    # -- transport binding ------------------------------------------------

    def _resolve(self, interface: str) -> Any:
        """Return the callable for an interface on the bound transport."""
        if self.import_path:
            if self._module is None:
                self._module = importlib.import_module(self.import_path)
            fn = getattr(self._module, interface, None)
            if fn is None:
                raise ClientUnboundError(
                    f"ROOSTER_IMPORT_PATH={self.import_path} does not expose "
                    f"callable {interface!r}"
                )
            return fn
        if self.base_url:
            spec = self._paths.get(interface)
            if not spec:
                raise ClientUnboundError(
                    f"no HTTP route configured for {interface!r}; set "
                    "ROOSTER_API_PATHS_JSON"
                )
            method, path = spec.split(" ", 1)
            return _HttpEndpoint(self.base_url, method, path)
        raise ClientUnboundError(
            "no invocation surface configured: set ROOSTER_IMPORT_PATH (direct "
            "Python) or ROOSTER_API_BASE_URL (+ROOSTER_API_PATHS_JSON) (HTTP) "
            "before running validation tests"
        )

    # -- contract interfaces ---------------------------------------------

    def ingest_schedule_event(self, **payload: Any) -> Dict[str, Any]:
        return self._call("ingest_schedule_event", payload)

    def correct_schedule_event(self, **payload: Any) -> Dict[str, Any]:
        return self._call("correct_schedule_event", payload)

    def retract_schedule_event(self, **payload: Any) -> Dict[str, Any]:
        return self._call("retract_schedule_event", payload)

    def list_schedule_events(self, **payload: Any) -> Dict[str, Any]:
        return self._call("list_schedule_events", payload)

    def get_schedule_event(self, **payload: Any) -> Dict[str, Any]:
        return self._call("get_schedule_event", payload)

    # -- helpers -----------------------------------------------------------

    def _call(self, interface: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._resolve(interface)
        try:
            return fn(**payload)
        except TypeError as exc:
            # The direct import may use positional-only signatures; fall back to
            # a single-dict argument if the callable accepts **input.
            if self.import_path and "unexpected keyword" in str(exc):
                try:
                    return fn(payload)
                except Exception:
                    raise exc
            raise

    def clear_state(self) -> None:
        """Reset persistence between tests if the surface exposes it.

        Prefers a reset hook on the bound import module (e.g.
        reset_isolated_db on app.invocation); falls back to env
        ROOSTER_RESET_FN=module.func. No-op otherwise.
        """
        if self.import_path:
            try:
                mod = self._ensure_module()
                reset_fn = getattr(mod, "reset_isolated_db", None)
                if reset_fn is not None:
                    reset_fn()
                    return
            except Exception:
                pass
        reset = os.environ.get("ROOSTER_RESET_FN")
        if reset:
            mod_name, _, fn_name = reset.partition(".")
            mod = importlib.import_module(mod_name)
            getattr(mod, fn_name)()

    def _ensure_module(self):
        if self._module is None:
            self._module = importlib.import_module(self.import_path)
        return self._module


class _HttpEndpoint:
    def __init__(self, base_url: str, method: str, path: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.method = method.upper()
        self.path = path

    def __call__(self, **payload: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{self.path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method=self.method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"error": body, "http_status": exc.code}

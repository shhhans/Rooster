"""Deterministic fake planning seam for incremental job tests.

Implements the frozen ``begin_run``/``step`` surface (planning's actual
function names per plan-review point ①) with a scripted list of step
results. This seam drives contract tests ONLY — a replay/fake seam never
proves live-provider acceptance (recorded distinction, Lead scope).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional


def _digest(obj: Any) -> str:
    """Same canonical form as the frozen contract: sorted keys, compact
    separators, UTF-8, sha256 (G3)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ScriptedPlanning:
    """Scripted IncrementalPlanning: records calls, pops scripted results."""

    def __init__(self, script: Optional[List[Dict[str, Any]]] = None,
                 draft: Optional[Dict[str, Any]] = None):
        self.script: List[Dict[str, Any]] = script or []
        self.default_draft = draft or {
            "draft_version": "incremental-run-v1",
            "plan": {"draft_revision": 1, "sleep_blocks": []},
            "segments": [],
            "pending_confirmations": [],
        }
        self.calls: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        """The test seam is always 'integrated'."""
        return True

    def _pop(self) -> Dict[str, Any]:
        if not self.script:
            raise AssertionError("script exhausted")
        return self.script.pop(0)

    def begin_run(self, snapshot, *, request, run_id=None, budget=None,
                  client=None, decision_mode=None):
        self.calls.append({"method": "begin_run", "snapshot": copy.deepcopy(snapshot),
                           "request": copy.deepcopy(request), "run_id": run_id,
                           "budget": budget, "client": client, "decision_mode": decision_mode})
        draft = copy.deepcopy(self.default_draft)
        draft.update(run_id=run_id, snapshot=copy.deepcopy(snapshot),
                     decision_mode="deterministic" if decision_mode == "deterministic" else "llm")
        return draft

    def resume_run(self, draft, *, input=None, budget_extension=None, snapshot=None, client=None):
        self.calls.append({"method": "resume_run", "draft": copy.deepcopy(draft),
                           "input": copy.deepcopy(input), "budget_extension": budget_extension})
        return draft

    def step(self, draft, snapshot=None, *, mode="step", limit=None, client=None):
        self.calls.append({"method": "step", "draft": copy.deepcopy(draft),
                           "snapshot": copy.deepcopy(snapshot), "client": client})
        scripted = self._pop()
        result = dict(scripted)
        run_state = copy.deepcopy(result.get("run_state") or draft)
        run_state.setdefault("run_id", draft.get("run_id"))
        result["run_state"] = run_state
        result["draft_digest"] = _digest(run_state)
        return result


class ScriptedPlanningException(ScriptedPlanning):
    """A seam that raises when scripted results are exhausted."""

    def __init__(self, exception: Exception):
        super().__init__(script=[])
        self.exception = exception

    def _pop(self) -> Dict[str, Any]:
        raise self.exception


class RecordingDecisionClient:
    """Test-only decision client (agent.ModelClient.complete shape).

    Records calls and never used in production paths; the production client
    is built from ROOSTER_DECISION_CLIENT (expected: planning's
    agent.ModelClient factory).
    """

    def __init__(self):
        self.calls = []

    def complete(self, stage_id: str, system: str, body: dict,
                 validate) -> dict:
        self.calls.append({"stage_id": stage_id, "body": body})
        return validate({"stage_id": stage_id, "decision": None})


def make_decision_client() -> RecordingDecisionClient:
    """Test factory for ROOSTER_DECISION_CLIENT='tests.fixtures.incremental_fake:make_decision_client'."""
    return RecordingDecisionClient()

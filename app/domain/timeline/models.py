"""Schedule event model.

The unified event shape defined in the frozen contract.
Every event record — ingest, correction, or retraction — has this shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


STATUS_PLANNED = "planned"
STATUS_OCCURRED = "occurred"
VALID_STATUSES = {STATUS_PLANNED, STATUS_OCCURRED}


@dataclass
class ScheduleEvent:
    """Unified schedule event record."""

    event_id: str
    track: str
    user_id: str
    dedupe_key: str
    occurred_at_utc: datetime  # tz-aware UTC datetime
    local_time: str  # ISO-8601 local timestamp string (no zone designator)
    timezone: str  # IANA timezone name
    source: str
    status: str  # 'planned' | 'occurred'
    title: str
    payload: Dict[str, Any] = field(default_factory=dict)
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    retracted: bool = False
    retracted_at: Optional[datetime] = None
    retracted_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    confirmation: str = "accepted"
    realization: str = "planned"
    created_in_txn: Optional[str] = None
    retired_in_txn: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict matching the contract output shape."""
        d = asdict(self)
        # datetime -> ISO 8601 string with Z suffix for UTC, or original offset
        d["occurred_at_utc"] = _utc_iso(self.occurred_at_utc)
        d["created_at"] = _utc_iso(self.created_at)
        d["updated_at"] = _utc_iso(self.updated_at)
        if self.retracted_at is not None:
            d["retracted_at"] = _utc_iso(self.retracted_at)
        else:
            d["retracted_at"] = None
        return d


def _utc_iso(dt: Optional[datetime]) -> Optional[str]:
    """Render a tz-aware datetime as an ISO-8601 UTC string with 'Z' suffix."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Treat naive as UTC; this should not happen in well-formed code.
        dt = dt.replace(tzinfo=timezone.utc)
    utc = dt.astimezone(timezone.utc)
    # Use 'Z' suffix for unambiguous UTC.
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

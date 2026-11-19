"""SQLite-backed persistence for timeline events.

This is the only persistence layer for the timeline kernel in this
iteration. The store speaks in plain schedule events and exposes the
query shapes needed by the schedule track and the kernel.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from app.domain.timeline.errors import EventNotFoundError
from app.domain.timeline.models import ScheduleEvent


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    track           TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    dedupe_key      TEXT NOT NULL,
    occurred_at_utc TEXT NOT NULL,
    local_time      TEXT NOT NULL,
    timezone        TEXT NOT NULL,
    source          TEXT NOT NULL,
    status          TEXT NOT NULL,
    title           TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}',
    supersedes      TEXT,
    superseded_by   TEXT,
    retracted       INTEGER NOT NULL DEFAULT 0,
    retracted_at    TEXT,
    retracted_reason TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(track, user_id, dedupe_key)
);

CREATE TABLE IF NOT EXISTS roster_confirmations (
    confirmation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    coverage_month TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, file_sha256, coverage_month)
);

CREATE TABLE IF NOT EXISTS reactive_detection_status (
    user_id TEXT PRIMARY KEY,
    error TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_user_window
    ON events (track, user_id, occurred_at_utc);
CREATE INDEX IF NOT EXISTS idx_events_supersedes
    ON events (supersedes);
CREATE INDEX IF NOT EXISTS idx_events_superseded_by
    ON events (superseded_by);

CREATE TABLE IF NOT EXISTS event_lineage (
    child_id TEXT NOT NULL REFERENCES events(event_id),
    parent_id TEXT NOT NULL REFERENCES events(event_id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (child_id, parent_id)
);
CREATE INDEX IF NOT EXISTS idx_event_lineage_parent ON event_lineage (parent_id);

CREATE TABLE IF NOT EXISTS proposal_batches (
    txn_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected')),
    input_revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS proposal_retirements (
    txn_id TEXT NOT NULL REFERENCES proposal_batches(txn_id),
    event_id TEXT NOT NULL REFERENCES events(event_id),
    PRIMARY KEY (txn_id, event_id)
);
CREATE TABLE IF NOT EXISTS sleep_profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_generation_state (
    user_id TEXT PRIMARY KEY,
    generated_through TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reactive_issues (
    issue_key TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL,
    issue_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, issue_key)
);
CREATE TABLE IF NOT EXISTS reactive_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL,
    input_revision INTEGER NOT NULL,
    txn_id TEXT,
    state_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reactive_model_requests (
    request_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES reactive_jobs(job_id),
    step_index INTEGER NOT NULL,
    max_tokens INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    finish_reason TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_reactive_model_requests_job ON reactive_model_requests(job_id, started_at);
CREATE TABLE IF NOT EXISTS reactive_model_chunks (
    request_id TEXT NOT NULL REFERENCES reactive_model_requests(request_id),
    seq INTEGER NOT NULL,
    received_at TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY(request_id, seq)
);

CREATE TABLE IF NOT EXISTS reactive_tool_activity (
    job_id TEXT NOT NULL REFERENCES reactive_jobs(job_id),
    step_index INTEGER NOT NULL,
    tool_index INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    details TEXT NOT NULL,
    summary TEXT,
    PRIMARY KEY(job_id, step_index, tool_index)
);

-- Event-log revision bookkeeping (INCREMENTAL-API-01).
-- Every inserted event gets a per-event revision: 1 for a fresh ingest,
-- parent revision + 1 for a correction (supersedes chain). Legacy events
-- predating this table read as revision 1 (COALESCE at read time).
CREATE TABLE IF NOT EXISTS event_revisions (
    event_id    TEXT PRIMARY KEY,
    revision    INTEGER NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Per-user input-fact version (the "event-log revision" of the frozen
-- incremental contract). Bumped transactionally by every event mutation
-- (insert / retract) so legacy endpoints and incremental commands share
-- one monotonic counter.
CREATE TABLE IF NOT EXISTS input_state (
    user_id          TEXT PRIMARY KEY,
    current_revision INTEGER NOT NULL,
    updated_at       TEXT NOT NULL
);

-- One row per input mutation with the affected local day (used for the
-- minimal calendar-coverage invalidation check at read time). ``origin``
-- distinguishes real input changes ('input') from an apply's own writes
-- ('apply') — the latter must not invalidate the coverage written by the
-- same atomic apply.
CREATE TABLE IF NOT EXISTS input_mutations (
    user_id     TEXT NOT NULL,
    revision    INTEGER NOT NULL,
    day         TEXT NOT NULL,
    origin      TEXT NOT NULL DEFAULT 'input',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, revision)
);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    # Stored as 'YYYY-MM-DDTHH:MM:SSZ' or full ISO-8601 with offset.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _row_to_event(row: sqlite3.Row) -> ScheduleEvent:
    payload = json.loads(row["payload"]) if row["payload"] else {}
    return ScheduleEvent(
        event_id=row["event_id"],
        track=row["track"],
        user_id=row["user_id"],
        dedupe_key=row["dedupe_key"],
        occurred_at_utc=_parse_dt(row["occurred_at_utc"]),
        local_time=row["local_time"],
        timezone=row["timezone"],
        source=row["source"],
        status=row["status"],
        title=row["title"],
        payload=payload,
        supersedes=row["supersedes"],
        superseded_by=row["superseded_by"],
        retracted=bool(row["retracted"]),
        retracted_at=_parse_dt(row["retracted_at"]),
        retracted_reason=row["retracted_reason"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        confirmation=row["confirmation"],
        realization=row["realization"],
        created_in_txn=row["created_in_txn"],
        retired_in_txn=row["retired_in_txn"],
        duty_start_utc=_parse_dt(row['duty_start_utc']),
        duty_end_utc=_parse_dt(row['duty_end_utc']),
    )


class EventStore:
    """Thread-safe SQLite event store.

    A single connection guarded by a lock keeps the schema and writes
    simple. For higher-throughput scenarios this can be swapped for a
    connection pool or a different backend without changing the schedule
    track or kernel surface.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._tx_depth = 0
        # check_same_thread=False so the same connection can serve across
        # threads; the lock provides serialization.
        self._conn = sqlite3.connect(
            db_path, detect_types=0, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            existing = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone()
            columns = {r['name'] for r in self._conn.execute('PRAGMA table_info(events)')} if existing else set()
            if existing and ('confirmation' not in columns or 'duty_start_utc' not in columns or 'duty_end_utc' not in columns) and db_path != ':memory:':
                db_file = Path(db_path).resolve()
                backup_dir = db_file.parent / 'backups'
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / (db_file.stem + '-before-event-dimensions-' + uuid.uuid4().hex + '.db')
                with sqlite3.connect(str(backup_path)) as backup:
                    self._conn.backup(backup)
            self._conn.executescript(SCHEMA_SQL)
            columns = {r['name'] for r in self._conn.execute('PRAGMA table_info(events)')}
            for name, definition in (('confirmation', "TEXT NOT NULL DEFAULT 'accepted'"),
                                     ('realization', "TEXT NOT NULL DEFAULT 'planned'"),
                                     ('created_in_txn', 'TEXT'), ('retired_in_txn', 'TEXT'),
                                     ('duty_start_utc', 'TEXT'), ('duty_end_utc', 'TEXT')):
                if name not in columns:
                    self._conn.execute(f'ALTER TABLE events ADD COLUMN {name} {definition}')
            self._conn.execute("UPDATE events SET source='system' WHERE source='roster'")
            self._conn.execute("UPDATE events SET realization='actual' WHERE status='occurred'")
            self._conn.execute(
                'INSERT OR IGNORE INTO event_lineage(child_id, parent_id, created_at) '
                'SELECT child.event_id, parent.event_id, child.created_at FROM events child '
                'JOIN events parent ON parent.event_id = child.supersedes AND parent.user_id = child.user_id')
            if "affected_at_utc" not in {r["name"] for r in self._conn.execute("PRAGMA table_info(input_mutations)")}:
                self._conn.execute("ALTER TABLE input_mutations ADD COLUMN affected_at_utc TEXT")

    # ---------- connection / transactions ----------

    @property
    def connection(self) -> sqlite3.Connection:
        """The shared SQLite connection (guarded by this store's lock)."""
        return self._conn

    @property
    def lock(self) -> threading.RLock:
        """The re-entrant lock serializing all access to the connection."""
        return self._lock

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Re-entrant ``BEGIN IMMEDIATE`` transaction on the shared connection.

        Nested calls join the outer transaction; only the outermost exit
        issues COMMIT (or ROLLBACK on error). The store lock is held for the
        whole scope, serializing writers — sufficient for the single-process
        v1 storage model.
        """
        with self._lock:
            if self._tx_depth == 0:
                self._conn.execute("BEGIN IMMEDIATE")
            self._tx_depth += 1
            try:
                yield self._conn
            except BaseException:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error:  # pragma: no cover - defensive
                        pass
                raise
            else:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.execute("COMMIT")

    # ---------- input-fact bookkeeping (shared by all mutation paths) ----------

    def _bump_input_revision(self, user_id: str, local_day: Optional[str],
                             origin: str = "input", affected_at_utc=None) -> int:
        """Bump the user's input-fact version and record the affected day.

        Must be called inside a transaction (or while holding the lock).
        ``origin='apply'`` marks mutations produced by an atomic apply of a
        job draft (its own writes are not "new input" for its own coverage).
        """
        now = _iso_z(_utcnow())
        row = self._conn.execute(
            "SELECT current_revision FROM input_state WHERE user_id = ?", (user_id,)
        ).fetchone()
        new_revision = (row["current_revision"] if row else 0) + 1
        self._conn.execute(
            "INSERT INTO input_state(user_id, current_revision, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "current_revision = excluded.current_revision, updated_at = excluded.updated_at",
            (user_id, new_revision, now),
        )
        day = (local_day or "")[:10]
        if not day or len(day) != 10:
            day = "0001-01-01"  # sorts before every real day (invalidates all)
        self._conn.execute(
            "INSERT INTO input_mutations(user_id, revision, day, origin, created_at, affected_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, new_revision, day, origin, now, affected_at_utc),
        )
        return new_revision

    def mark_apply_mutations(self, user_id: str, after_revision: int) -> int:
        """Re-tag mutation rows created after ``after_revision`` as
        origin='apply' (used inside an atomic apply transaction). Returns
        the number of re-tagged rows."""
        cur = self._conn.execute(
            "UPDATE input_mutations SET origin = 'apply' "
            "WHERE user_id = ? AND revision > ?",
            (user_id, after_revision),
        )
        return cur.rowcount

    def get_input_revision(self, user_id: str) -> int:
        """Current per-user input-fact version (0 when never mutated)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT current_revision FROM input_state WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row["current_revision"] if row else 0

    def get_event_revision(self, event_id: str) -> int:
        """Per-event revision; events predating the bookkeeping read as 1."""
        with self._lock:
            row = self._conn.execute(
                "SELECT revision FROM event_revisions WHERE event_id = ?", (event_id,)
            ).fetchone()
            return row["revision"] if row else 1

    # ---------- writes ----------

    def insert(self, event: ScheduleEvent, *, bump_revision: bool = True) -> None:
        with self.transaction():
            # Per-event revision: fresh ingest = 1, correction = parent + 1.
            if event.supersedes:
                parent = self._conn.execute(
                    "SELECT revision FROM event_revisions WHERE event_id = ?",
                    (event.supersedes,),
                ).fetchone()
                revision = (parent["revision"] if parent else 1) + 1
            else:
                revision = 1
            self._conn.execute(
                """
                INSERT INTO events(
                    event_id, track, user_id, dedupe_key,
                    occurred_at_utc, local_time, timezone,
                    source, status, title, payload,
                    supersedes, superseded_by,
                    retracted, retracted_at, retracted_reason,
                    created_at, updated_at, confirmation, realization, created_in_txn, retired_in_txn
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.event_id,
                    event.track,
                    event.user_id,
                    event.dedupe_key,
                    _iso_z(event.occurred_at_utc),
                    event.local_time,
                    event.timezone,
                    event.source,
                    event.status,
                    event.title,
                    json.dumps(event.payload or {}),
                    event.supersedes,
                    event.superseded_by,
                    1 if event.retracted else 0,
                    _iso_z(event.retracted_at) if event.retracted_at else None,
                    event.retracted_reason,
                    _iso_z(event.created_at) if event.created_at else None,
                    _iso_z(event.updated_at) if event.updated_at else None,
                    event.confirmation,
                    event.realization,
                    event.created_in_txn,
                    event.retired_in_txn,
                ),
            )
            if event.track == 'schedule' and event.title == 'flight':
                self._conn.execute('UPDATE events SET duty_start_utc=?, duty_end_utc=? WHERE event_id=?',
                    (_iso_z(event.duty_start_utc) if event.duty_start_utc else None,
                     _iso_z(event.duty_end_utc) if event.duty_end_utc else None, event.event_id))
            self._conn.execute(
                "INSERT INTO event_revisions(event_id, revision, updated_at) "
                "VALUES (?, ?, ?)",
                (event.event_id, revision, _iso_z(_utcnow())),
            )
            if event.supersedes:
                self.add_lineage(event.user_id, event.event_id, [event.supersedes])
            affected = _iso_z(event.occurred_at_utc)
            if event.supersedes:
                prior = self.get_by_id(event.supersedes)
                if prior:
                    affected = min(affected, _iso_z(prior.occurred_at_utc))
            if bump_revision:
                self._bump_input_revision(event.user_id, event.local_time, affected_at_utc=affected)

    def add_lineage(self, user_id: str, child_id: str, parent_ids: List[str]) -> None:
        """Link a newly created event to existing events owned by the same user."""
        with self.transaction():
            child = self.get_by_id(child_id)
            if child is None or child.user_id != user_id:
                raise EventNotFoundError(f'event {child_id} not found')
            for parent_id in dict.fromkeys(parent_ids):
                parent = self.get_by_id(parent_id)
                if parent is None or parent.user_id != user_id or parent_id == child_id:
                    raise EventNotFoundError(f'parent event {parent_id} not found')
                cycle = self._conn.execute(
                    'WITH RECURSIVE ancestors(id) AS ('
                    'SELECT parent_id FROM event_lineage WHERE child_id = ? UNION '
                    'SELECT edge.parent_id FROM event_lineage edge JOIN ancestors a ON edge.child_id = a.id) '
                    'SELECT 1 FROM ancestors WHERE id = ? LIMIT 1', (parent_id, child_id)).fetchone()
                if cycle:
                    raise ValueError('event lineage cannot contain a cycle')
                self._conn.execute('INSERT OR IGNORE INTO event_lineage VALUES (?, ?, ?)',
                                   (child_id, parent_id, _iso_z(_utcnow())))

    def event_lineage(self, user_id: str, event_id: str) -> Dict[str, List[str]]:
        with self._lock:
            event = self.get_by_id(event_id)
            if event is None or event.user_id != user_id:
                raise EventNotFoundError(f'event {event_id} not found')
            parents = self._conn.execute('SELECT parent_id FROM event_lineage WHERE child_id = ? ORDER BY parent_id',
                                         (event_id,)).fetchall()
            children = self._conn.execute('SELECT child_id FROM event_lineage WHERE parent_id = ? ORDER BY child_id',
                                          (event_id,)).fetchall()
            return {'parents': [row['parent_id'] for row in parents],
                    'children': [row['child_id'] for row in children]}

    def update_superseded_by(self, target_event_id: str, new_event_id: str) -> None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET superseded_by = ?, updated_at = ? "
                "WHERE event_id = ?",
                (new_event_id, _iso_z(_utcnow()), target_event_id),
            )
            if cur.rowcount == 0:
                raise EventNotFoundError(f"event {target_event_id} not found")

    def set_retracted(
        self,
        event_id: str,
        retracted_at: datetime,
        reason: Optional[str],
    ) -> None:
        with self.transaction():
            target = self._conn.execute(
                "SELECT user_id, local_time, occurred_at_utc FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            cur = self._conn.execute(
                "UPDATE events SET retracted = 1, retracted_at = ?, "
                "retracted_reason = ?, updated_at = ? "
                "WHERE event_id = ?",
                (_iso_z(retracted_at), reason, _iso_z(retracted_at), event_id),
            )
            if cur.rowcount == 0:
                raise EventNotFoundError(f"event {event_id} not found")
            # Retraction removes an input fact -> bump the input revision.
            self._bump_input_revision(target["user_id"], target["local_time"],
                                      affected_at_utc=target["occurred_at_utc"])

    # ---------- reads ----------

    def is_visible_event(self, event: ScheduleEvent) -> bool:
        """Candidate records are only exposed by the proposal interface."""
        if not event.created_in_txn:
            return True
        with self._lock:
            row = self._conn.execute('SELECT status FROM proposal_batches WHERE txn_id=?', (event.created_in_txn,)).fetchone()
            return bool(row and row['status'] == 'accepted')

    def get_by_id(self, event_id: str) -> Optional[ScheduleEvent]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return _row_to_event(row) if row else None

    def get_by_dedupe_key(
        self, user_id: str, track: str, dedupe_key: str
    ) -> Optional[ScheduleEvent]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE track = ? AND user_id = ? "
                "AND dedupe_key = ?",
                (track, user_id, dedupe_key),
            ).fetchone()
            return _row_to_event(row) if row else None

    def get_existing_correction(
        self,
        target_event_id: str,
        occurred_at_utc: datetime,
        status: str,
        title: str,
        payload: Dict[str, Any],
    ) -> Optional[ScheduleEvent]:
        """Find an existing correction whose content matches the candidate.

        Content equality is checked on (occurred_at_utc, status, title,
        payload). The payload is compared via JSON normalization to ignore
        key ordering and trivial whitespace differences.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE supersedes = ?", (target_event_id,)
            ).fetchall()
            candidate_payload_json = json.dumps(payload or {}, sort_keys=True)
            for row in rows:
                e = _row_to_event(row)
                if (
                    e.occurred_at_utc == occurred_at_utc
                    and e.status == status
                    and e.title == title
                    and json.dumps(e.payload or {}, sort_keys=True) == candidate_payload_json
                ):
                    return e
            return None

    def list_for_user_window(
        self,
        user_id: str,
        track: str,
        window_start_utc: datetime,
        window_end_utc: datetime,
        include_history: bool,
    ) -> List[ScheduleEvent]:
        with self._lock:
            if include_history:
                rows = self._conn.execute(
                    "SELECT * FROM events "
                    "WHERE track = ? AND user_id = ? "
                    "AND (created_in_txn IS NULL OR created_in_txn IN (SELECT txn_id FROM proposal_batches WHERE status = 'accepted')) "
                    "AND occurred_at_utc >= ? AND occurred_at_utc < ? "
                    "ORDER BY occurred_at_utc ASC, created_at ASC, event_id ASC",
                    (
                        track,
                        user_id,
                        _iso_z(window_start_utc),
                        _iso_z(window_end_utc),
                    ),
                ).fetchall()
            else:
                # Exclude retracted and superseded events (those with a
                # superseded_by pointer).
                rows = self._conn.execute(
                    "SELECT * FROM events "
                    "WHERE track = ? AND user_id = ? "
                    "AND occurred_at_utc >= ? AND occurred_at_utc < ? "
                    "AND retracted = 0 AND superseded_by IS NULL "
                    "AND (created_in_txn IS NULL OR created_in_txn IN (SELECT txn_id FROM proposal_batches WHERE status = 'accepted')) "
                    "AND (retired_in_txn IS NULL OR retired_in_txn NOT IN (SELECT txn_id FROM proposal_batches WHERE status = 'accepted')) "
                    "ORDER BY occurred_at_utc ASC, created_at ASC, event_id ASC",
                    (
                        track,
                        user_id,
                        _iso_z(window_start_utc),
                        _iso_z(window_end_utc),
                    ),
                ).fetchall()
            return [_row_to_event(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _iso_z(dt: datetime) -> str:
    """Render a tz-aware datetime as 'YYYY-MM-DDTHH:MM:SSZ' for storage."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_event_id() -> str:
    """Generate a fresh UUID4 event_id."""
    return str(uuid.uuid4())

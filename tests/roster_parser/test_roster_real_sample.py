"""REAL-sample verification — the only roster test that touches the network.

Gated: runs only when BOTH RUN_ROSTER_REAL_SAMPLE=1 and MINIMAX_API_KEY are
set (and the sample file exists). Counts against the task's shared budget of
at most 6 real MiniMax HTTP attempts (including failures). Never runs in the
default offline test pass.
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SAMPLE_PATH = os.environ.get(
    "ROSTER_REAL_SAMPLE_PATH", r"D:\Coding\Projects\Rooster\Nov 26 Roster.pdf"
)
RUN_REAL = os.environ.get("RUN_ROSTER_REAL_SAMPLE") == "1"
HAVE_KEY = bool(os.environ.get("MINIMAX_API_KEY"))
HAVE_SAMPLE = os.path.exists(SAMPLE_PATH)

pytestmark = [
    pytest.mark.skipif(
        not (RUN_REAL and HAVE_KEY and HAVE_SAMPLE),
        reason="REAL test: requires RUN_ROSTER_REAL_SAMPLE=1, MINIMAX_API_KEY "
        "and the local sample file (kept out of Git)",
    ),
]


def test_real_sample_pdf_preview():
    """REAL MiniMax image-input parse of the local November 2026 roster PDF."""
    with open(SAMPLE_PATH, "rb") as handle:
        pdf_bytes = handle.read()

    client = TestClient(create_app(), raise_server_exceptions=False)
    with client as http:
        response = http.post(
            "/api/v1/roster/preview",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
            data={
                "home_timezone": "Asia/Hong_Kong",
                "airport_timezone_map": json_dumps_default_map(),
                "roster_year": "2026",
                "roster_month": "11",
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()

    meta = body["meta"]

    # Full evidence dump BEFORE asserts, so a single attempt is never wasted
    # on a count mismatch (the roster itself never reaches Git; only run with
    # RUN_ROSTER_REAL_SAMPLE=1 on a machine holding the sample).
    print("\nREAL meta: model=%s usage=%s sha256=%s elapsed=%ss" % (
        meta["model"], meta["usage"], meta["file_sha256"], meta["elapsed_seconds"]))
    for event in body["proposed_events"]:
        print("REAL event:", event["event_type"],
              json.dumps(event["schedule_payload"], ensure_ascii=False),
              "| quote:", str(event["provenance"]["quote"])[:60])
    for item in body["unresolved_items"]:
        print("REAL unresolved:", item["code"], "|", item["message"][:100])
    for warning in body["warnings"]:
        print("REAL warning:", warning["code"], "|", warning["message"][:100])

    assert meta["content_type"] == "application/pdf"
    assert meta["render"]["pdf_page_count"] == 1
    assert meta["http_attempts"] == 1
    assert meta["model"]
    assert meta["file_sha256"]

    # A1/A2: every sector either traceable in proposed_events or explicitly
    # unresolved; blank/G markers never appear as events with times.
    flights = [
        e for e in body["proposed_events"] if e["event_type"] == "flight"
    ]
    duties = [e for e in body["proposed_events"] if e["event_type"] == "duty"]
    assert len(flights) == 10, (
        f"expected 10 flight sectors, got {len(flights)}; "
        f"unresolved={[i['code'] for i in body['unresolved_items']]}"
    )
    assert len(duties) >= 1
    for event in flights + duties:
        assert event["provenance"]["page"] == 1
        assert event["confirm_required"] is True

    # No fabricated rest events (A3)
    assert not any(e["event_type"] not in {"flight", "duty"} for e in body["proposed_events"])
    print(
        "REAL sample evidence: model=%s usage=%s sha256=%s flights=%d duties=%d "
        "unresolved=%s warnings=%s"
        % (
            meta["model"],
            meta["usage"],
            meta["file_sha256"],
            len(flights),
            len(duties),
            [i["code"] for i in body["unresolved_items"]],
            [w["code"] for w in body["warnings"]],
        )
    )


def json_dumps_default_map() -> str:
    import json

    return json.dumps(
        {
            "HKG": "Asia/Hong_Kong",
            "LHR": "Europe/London",
            "SYD": "Australia/Sydney",
        }
    )
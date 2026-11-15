"""Shared helpers for roster parser tests (all offline, constructed data)."""
from __future__ import annotations

import io
import json

import httpx
from PIL import Image

from app.roster_parser.provider import ProviderResult


def make_png(width: int = 8, height: int = 8, color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg(width: int = 8, height: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (0, 128, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_pdf(page_count: int = 1) -> bytes:
    """Build a minimal deterministic PDF with blank pages (no external deps)."""
    objects: list[str] = [
        "<</Type/Catalog/Pages 2 0 R>>",
        "<</Type/Pages/Kids["
        + " ".join(f"{3 + i} 0 R" for i in range(page_count))
        + f"]/Count {page_count}>>",
    ]
    objects.extend(
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>" for _ in range(page_count)
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj{body}endobj\n".encode("ascii")
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


class FakeProvider:
    """Offline stand-in for MiniMaxClient: captures calls, returns canned data."""

    def __init__(
        self,
        content: str | None = None,
        error: Exception | None = None,
        model: str = "MiniMax-M3.1-Flash",
    ) -> None:
        self.content = content
        self.error = error
        self.model = model
        self.calls: list[dict] = []

    def complete_vision(self, system_prompt, images, user_text) -> ProviderResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "images": list(images),
                "user_text": user_text,
            }
        )
        if self.error is not None:
            raise self.error
        return ProviderResult(
            content=self.content or "",
            model=self.model,
            usage={"total_tokens": 7},
            status_code=200,
            elapsed_seconds=0.01,
            http_attempts=1,
        )


def model_output(events: list[dict], unresolved=None, blank_dates=None) -> str:
    return json.dumps(
        {"events": events, "unresolved": unresolved or [], "blank_dates": blank_dates or []}
    )


def flight_event(**overrides) -> dict:
    event = {
        "type": "flight",
        "page": 1,
        "raw_row": "XX089  HKG LHR  23:55  06:10+1",
        "flight_number": "XX089",
        "origin": "HKG",
        "destination": "LHR",
        "date": "2026-11-03",
        "dep": "23:55",
        "arr": "06:10",
        "arr_day_offset": 1,
        "markers": ["+1"],
        "continuation": False,
        "continues_index": None,
    }
    event.update(overrides)
    return event


def duty_event(**overrides) -> dict:
    event = {
        "type": "duty",
        "page": 1,
        "raw_row": "PSJ00  03:15  12:30",
        "code": "PSJ00",
        "date": "2026-11-04",
        "start": "03:15",
        "end": "12:30",
        "markers": [],
    }
    event.update(overrides)
    return event


def mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)

"""Input sniffing and PDF rendering for the roster preview pipeline.

PDF pages are rasterized with pypdfium2 (Apache-2.0, self-contained wheel —
no poppler or other external binary) and passed to the vision model as PNG
images. PNG/JPEG uploads bypass rendering and go straight to the model as
image inputs. The pipeline never degrades to text-only matching.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass

from app.roster_parser.errors import (
    FileTooLargeError,
    PdfRenderError,
    TooManyPagesError,
    UnsupportedFileTypeError,
)

MAGIC_PDF = b"%PDF-"
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_JPEG = b"\xff\xd8\xff"

CONTENT_TYPE_PDF = "application/pdf"
CONTENT_TYPE_PNG = "image/png"
CONTENT_TYPE_JPEG = "image/jpeg"


@dataclass(frozen=True)
class RosterLimits:
    """Upload/render limits. Documented in docs/roster-parser.md."""

    max_upload_bytes: int
    max_pages: int
    render_dpi: int

    @classmethod
    def from_env(cls) -> "RosterLimits":
        return cls(
            max_upload_bytes=_int_env("ROSTER_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_pages=_int_env("ROSTER_MAX_PAGES", 1),
            render_dpi=_int_env("ROSTER_RENDER_DPI", 200),
        )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def sniff_content_type(data: bytes) -> str:
    """Return the MIME type from magic bytes; reject anything else."""
    if data.startswith(MAGIC_PDF):
        return CONTENT_TYPE_PDF
    if data.startswith(MAGIC_PNG):
        return CONTENT_TYPE_PNG
    if data.startswith(MAGIC_JPEG):
        return CONTENT_TYPE_JPEG
    raise UnsupportedFileTypeError(
        "unsupported file type: expected PDF, PNG or JPEG "
        f"(received {len(data)} bytes with unrecognized content)"
    )


def enforce_upload_limit(data: bytes, limits: RosterLimits) -> None:
    if len(data) > limits.max_upload_bytes:
        raise FileTooLargeError(
            f"upload exceeds limit: {len(data)} bytes > "
            f"{limits.max_upload_bytes} bytes"
        )


def render_pdf_pages(data: bytes, limits: RosterLimits) -> list[bytes]:
    """Render each (limited) PDF page to PNG bytes via pypdfium2."""
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(data)
    except Exception as exc:  # pypdfium2 raises PdfiumError (unchecked)
        raise PdfRenderError(f"invalid or unreadable PDF: {exc}") from exc
    try:
        page_count = len(document)
        if page_count > limits.max_pages:
            raise TooManyPagesError(
                f"PDF has {page_count} pages, limit is {limits.max_pages}; "
                "multi-page rosters require explicitly raising ROSTER_MAX_PAGES"
            )
        pages: list[bytes] = []
        for index in range(page_count):
            bitmap = document[index].render(scale=limits.render_dpi / 72.0)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append(buffer.getvalue())
        return pages
    finally:
        document.close()

"""Rendering/sniffing tests (offline, constructed PDF and image fixtures)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from app.roster_parser.errors import (
    FileTooLargeError,
    PdfRenderError,
    TooManyPagesError,
    UnsupportedFileTypeError,
)
from app.roster_parser.rendering import (
    CONTENT_TYPE_JPEG,
    CONTENT_TYPE_PDF,
    CONTENT_TYPE_PNG,
    RosterLimits,
    enforce_upload_limit,
    render_pdf_pages,
    sniff_content_type,
)
from tests.roster_parser.conftest import make_pdf, make_png

LIMITS = RosterLimits(max_upload_bytes=10 * 1024 * 1024, max_pages=3, render_dpi=72)


class TestSniff:
    def test_pdf_magic(self):
        assert sniff_content_type(b"%PDF-1.4\nrest") == CONTENT_TYPE_PDF

    def test_png_magic(self):
        assert sniff_content_type(make_png()) == CONTENT_TYPE_PNG

    def test_jpeg_magic(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
        assert sniff_content_type(buffer.getvalue()) == CONTENT_TYPE_JPEG

    def test_rejects_text(self):
        with pytest.raises(UnsupportedFileTypeError):
            sniff_content_type(b"just some text")

    def test_rejects_empty(self):
        with pytest.raises(UnsupportedFileTypeError):
            sniff_content_type(b"")

    def test_rejects_bmp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="BMP")
        with pytest.raises(UnsupportedFileTypeError):
            sniff_content_type(buffer.getvalue())


class TestRender:
    def test_single_page_renders_png(self):
        pages = render_pdf_pages(make_pdf(1), LIMITS)
        assert len(pages) == 1
        assert pages[0].startswith(b"\x89PNG\r\n\x1a\n")
        image = Image.open(io.BytesIO(pages[0]))
        # 612x792 pt at 72 dpi -> 612x792 px
        assert image.size == (612, 792)

    def test_multi_page_within_limit(self):
        pages = render_pdf_pages(make_pdf(2), LIMITS)
        assert len(pages) == 2

    def test_page_limit_enforced(self):
        with pytest.raises(TooManyPagesError):
            render_pdf_pages(make_pdf(2), RosterLimits(max_upload_bytes=100, max_pages=1, render_dpi=72))

    def test_invalid_pdf_raises_render_error(self):
        with pytest.raises(PdfRenderError):
            render_pdf_pages(b"%PDF-1.4\nthis is not a pdf body", LIMITS)


class TestUploadLimit:
    def test_within_limit_ok(self):
        enforce_upload_limit(b"x" * 10, RosterLimits(max_upload_bytes=10, max_pages=1, render_dpi=72))

    def test_over_limit_raises(self):
        with pytest.raises(FileTooLargeError):
            enforce_upload_limit(b"x" * 11, RosterLimits(max_upload_bytes=10, max_pages=1, render_dpi=72))

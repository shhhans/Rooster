"""Typed errors for the roster preview pipeline.

Each error maps to one HTTP status and one stable ``error_code`` so routes can
raise ``HTTPException`` with the shared error envelope. No error is ever
silently swallowed: a failed parse always surfaces as an explicit error or an
unresolved item, never as an empty successful result.
"""
from __future__ import annotations


class RosterError(Exception):
    """Base class for roster preview pipeline errors."""

    http_status = 500
    error_code = "roster_internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(RosterError):
    http_status = 415
    error_code = "unsupported_file_type"


class FileTooLargeError(RosterError):
    http_status = 413
    error_code = "file_too_large"


class TooManyPagesError(RosterError):
    http_status = 413
    error_code = "too_many_pages"


class PdfRenderError(RosterError):
    http_status = 422
    error_code = "pdf_render_error"


class MissingApiKeyError(RosterError):
    http_status = 503
    error_code = "missing_api_key"


class ProviderTimeoutError(RosterError):
    http_status = 504
    error_code = "provider_timeout"


class ProviderError(RosterError):
    """Provider returned an unusable HTTP response (bad status/body)."""

    http_status = 502
    error_code = "provider_error"


class InvalidModelOutputError(RosterError):
    """Provider answered but the content is not the contracted JSON."""

    http_status = 502
    error_code = "invalid_model_output"


class InvalidTimezoneContextError(RosterError):
    """Caller-supplied home_timezone / airport map is not valid IANA data."""

    http_status = 400
    error_code = "invalid_timezone_context"


class ProviderConfigError(RosterError):
    """Operator-supplied provider configuration (env) is invalid.

    Configuration problems are always explicit errors — never silently
    replaced by a default (no silent fallback for endpoint/effort choices).
    """

    http_status = 500
    error_code = "provider_config_error"

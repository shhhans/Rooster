"""Roster file parsing, MiniMax model adaptation and standardized preview.

Scope (ROSTER-PARSE-01): turn uploaded PDF/PNG/JPEG crew rosters into
proposed schedule events with provenance and unresolved items for user
confirmation. Preview only — nothing is persisted by this module.
"""
from __future__ import annotations

# Version of the roster parser implementation itself.
ROSTER_PARSER_VERSION = "0.1.0"
# Version of the preview response contract (proposed_events / unresolved_items
# / warnings / meta). Bump on any breaking shape change.
ROSTER_CONTRACT_VERSION = "roster-preview-v1"

__all__ = ["ROSTER_PARSER_VERSION", "ROSTER_CONTRACT_VERSION"]

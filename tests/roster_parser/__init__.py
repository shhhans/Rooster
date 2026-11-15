"""Offline deterministic tests for the roster parser (no network access).

Real-sample verification is isolated in test_roster_real_sample.py and gated
behind RUN_ROSTER_REAL_SAMPLE=1 + MINIMAX_API_KEY; everything here uses
constructed fixtures and fake/mock providers only.
"""

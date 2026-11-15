"""Normalized Process S, exact update across a constant sleep/wake interval."""
import math


def advance(S, hours, asleep, tau_w=18.2, tau_s=4.2):
    if not 0 <= S <= 1 or hours < 0 or min(tau_w, tau_s) <= 0:
        raise ValueError('invalid Process S state or duration')
    return S * math.exp(-hours / tau_s) if asleep else 1 - (1-S)*math.exp(-hours/tau_w)

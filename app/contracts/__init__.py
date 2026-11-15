"""Neutral shared contracts (Lead-managed).

Currently exposes the shared time/interval utilities extracted from
``app.sleep_model.normalize`` (module decoupling contract step 2). Old import
paths remain re-exported by ``app.sleep_model.normalize``.
"""
from .time import clock_hours, dt, interval, local_instant, merge, overlap, stamp

__all__ = ['clock_hours', 'dt', 'interval', 'local_instant', 'merge', 'overlap', 'stamp']

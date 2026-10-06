"""Opt-in PEPS phase timings without retaining states or changing numerics."""

from functools import wraps
import time


class PhaseTimer:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.seconds = {}
        self.calls = {}

    def snapshot(self):
        return {"seconds": dict(self.seconds), "calls": dict(self.calls)}

    def since(self, previous):
        return {
            "seconds": {k: v - previous["seconds"].get(k, 0.0)
                        for k, v in self.seconds.items()},
            "calls": {k: v - previous["calls"].get(k, 0)
                      for k, v in self.calls.items()},
        }


def timed_phase(name):
    """Time an owning phase only while run(timing=True) is active."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            timer = getattr(self, "_phase_timer", None)
            if timer is None:
                return method(self, *args, **kwargs)
            timer.synchronize()
            started = time.perf_counter()
            try:
                return method(self, *args, **kwargs)
            finally:
                timer.synchronize()
                timer.seconds[name] = timer.seconds.get(name, 0.0) + time.perf_counter() - started
                timer.calls[name] = timer.calls.get(name, 0) + 1
        return wrapped
    return decorate


def profile_run(method):
    """Keep successful and failed run summaries, clearing active state on exit."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._last_timing = {}
        self._phase_timer = None
        if not kwargs.get("timing", False):
            return method(self, *args, **kwargs)
        sync = bool(kwargs.get("timing_sync_device", False))

        def synchronize():
            if sync:
                from ...fitting.local import FIT

                FIT.synchronize_backend(self.state)

        timer = PhaseTimer(synchronize)
        self._phase_timer = timer
        status = "failed"
        started = time.perf_counter()
        try:
            synchronize()
            started = time.perf_counter()
            result = method(self, *args, **kwargs)
            status = "complete"
            return result
        finally:
            try:
                synchronize()
                self._last_timing = {
                    **timer.snapshot(), "total_seconds": time.perf_counter() - started,
                    "status": status, "synchronized": sync,
                }
            finally:
                self._phase_timer = None
    return wrapped

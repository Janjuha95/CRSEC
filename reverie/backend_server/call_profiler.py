"""
Lightweight global LLM-call profiler.

Records per prompt-function call counts and cumulative wall-clock seconds
(attributed to the nearest run_gpt_* caller on the stack), plus arbitrary
named counters (cache hits, pre-filter skips, ...). Zero behavior change to
the simulation; this exists so optimization wins can be verified.

Dumped to profile.json on sim save/fin (see reverie.py ReverieServer.save)
and on interpreter exit as a fallback.
"""
import atexit
import json
import os
import threading

_lock = threading.Lock()
_calls = {}      # fn_name -> {"count": int, "seconds": float}
_counters = {}   # counter_name -> int

_DEFAULT_DUMP_PATH = os.path.join(os.getcwd(), "profile.json")


def record_call(fn_name, seconds):
    """Record one LLM call attributed to prompt-function fn_name."""
    with _lock:
        entry = _calls.setdefault(fn_name, {"count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] += seconds


def incr(counter_name, n=1):
    """Increment a named counter (cache hits, pre-filter skips, ...)."""
    with _lock:
        _counters[counter_name] = _counters.get(counter_name, 0) + n


def snapshot():
    with _lock:
        total_calls = sum(v["count"] for v in _calls.values())
        total_seconds = sum(v["seconds"] for v in _calls.values())
        return {
            "total_llm_calls": total_calls,
            "total_llm_seconds": round(total_seconds, 3),
            "per_function": {
                k: {"count": v["count"], "seconds": round(v["seconds"], 3)}
                for k, v in sorted(_calls.items())
            },
            "counters": dict(_counters),
        }


def dump(path=None):
    """Write the current profile to profile.json. Never raises."""
    try:
        with open(path or _DEFAULT_DUMP_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot(), f, indent=2)
    except Exception:
        pass


def reset():
    """Clear all recorded data (used by tests)."""
    with _lock:
        _calls.clear()
        _counters.clear()


atexit.register(dump)

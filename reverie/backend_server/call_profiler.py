"""
Lightweight global LLM-call profiler.

Records per prompt-function call counts and cumulative wall-clock seconds
(attributed to the nearest run_gpt_* caller on the stack), plus arbitrary
named counters (cache hits, pre-filter skips, ...). Zero behavior change to
the simulation; this exists so optimization wins can be verified.

The single instrumentation point is llm_router.llm_call (every Ollama
generate call funnels through it, including the norm module's
safe_generate_response path), so recording lives there rather than in the
gpt_structure wrappers — those merely delegate to llm_router and recording
in both would double-count.

Dumped to profile.json:
  * incrementally, every _FLUSH_EVERY calls (so a run killed mid-way still
    leaves a partial profile),
  * on sim save/fin (see reverie.py ReverieServer.save), and
  * on interpreter exit as a fallback.
The incremental and atexit dumps only fire once a real run has configured a
target via set_dump_path(); this keeps unit tests (which call record_call but
never set a path) from clobbering a checked-in profile.json.
"""
import atexit
import json
import os
import threading

_lock = threading.Lock()
_calls = {}      # fn_name -> {"count": int, "seconds": float}
_counters = {}   # counter_name -> int

# Where dump() writes when no explicit path is passed. set_dump_path() points
# this at the active sim folder; until then incremental/atexit dumps are off.
_dump_path = None
_DEFAULT_DUMP_PATH = os.path.join(os.getcwd(), "profile.json")

# Flush cadence for the incremental dump (A2). Counts every recorded call.
_FLUSH_EVERY = 200
_calls_since_flush = 0


def set_dump_path(path):
    """Point incremental/atexit dumps at `path` (e.g. <sim_folder>/profile.json).

    Called once at sim startup. Enabling a path is also what turns the
    incremental and atexit auto-dumps on.
    """
    global _dump_path
    with _lock:
        _dump_path = path


def record_call(fn_name, seconds):
    """Record one LLM call attributed to prompt-function fn_name."""
    global _calls_since_flush
    do_flush = False
    with _lock:
        entry = _calls.setdefault(fn_name, {"count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] += seconds
        _calls_since_flush += 1
        if _dump_path is not None and _calls_since_flush >= _FLUSH_EVERY:
            _calls_since_flush = 0
            do_flush = True
    # dump() reacquires _lock via snapshot(); flush outside the lock to avoid
    # a re-entrant deadlock.
    if do_flush:
        dump()


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
    target = path or _dump_path or _DEFAULT_DUMP_PATH
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(snapshot(), f, indent=2)
    except Exception:
        pass


def reset():
    """Clear all recorded data (used by tests)."""
    global _calls_since_flush
    with _lock:
        _calls.clear()
        _counters.clear()
        _calls_since_flush = 0


def _atexit_dump():
    """Fallback flush on interpreter exit — only for real runs that called
    set_dump_path(); avoids polluting ./profile.json from tests/imports."""
    if _dump_path is not None:
        dump()


atexit.register(_atexit_dump)

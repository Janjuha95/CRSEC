"""
LLM Router — routes calls to the appropriate Ollama model based on call_type.
"""
import os
import sys
import json
import time
import datetime
import ollama

import call_profiler

# Two-tier routing: primary handles high-volume simple calls,
# reasoning handles critical reasoning. Both currently point at 30b-a3b
# because the reasoning tier model (qwen3:32b dense) isn't pulled yet.
# When you pull qwen3:32b on VSC, flip REASONING_MODEL below.
PRIMARY_MODEL = "qwen3:30b-a3b"
REASONING_MODEL = "qwen3:32b"  

PRIMARY_CALL_TYPES = {
    "format_check",
    "type_check",
    "duplicate_check",
    "fact_consistency",
}

REASONING_CALL_TYPES = {
    "norm_creation",
    "norm_evaluation",
    "conflict_detection",
    "conversation",
    "defection_assessment",
    "violation_check",
    "default",
}

# Tiered routing (Step 2 of the perf pass): cosmetic / noise-tolerant prompt
# functions go to a small model. Flag-gated; default ON. Routing is keyed on
# the run_gpt_* caller name (also used for profiling attribution) so the
# legacy gpt_structure wrappers don't need a call_type threaded through.
# A/B: set CRSEC_TIERED_ROUTING=0 to restore single-model routing.
SMALL_MODEL = os.environ.get("CRSEC_SMALL_MODEL", "qwen3:4b")
SMALL_ROUTE_PROMPT_FNS = {
    "run_gpt_prompt_pronunciatio",       # emoji cosmetics
    "run_gpt_prompt_event_triple",       # (s, p, o) extraction
}
# NOTE: the three poignancy functions were removed from small-model routing.
# They go through ChatGPT_safe_generate_response, which demands a strict
# {"output": "<int>"} JSON envelope; qwen3:4b does not reliably emit that
# envelope, so json.loads(...)["output"] threw on every repeat and the
# wrapper returned False -> [FAIL_SAFE] on nearly every poignancy call. The
# larger primary model satisfies the envelope, so poignancy stays on it.
# See PATCH_LOG.md (Part B).

# Per-request Ollama options (Step 1d). The sim's determinism (and therefore
# the memoization in run_gpt_prompt*.py) relies on temperature 0, so it is
# pinned here at the single choke point rather than trusting model defaults.
# num_ctx caps the KV cache per request; override via env for A/B.
OLLAMA_NUM_CTX = int(os.environ.get("CRSEC_NUM_CTX", "16384"))
OLLAMA_TEMPERATURE = float(os.environ.get("CRSEC_TEMPERATURE", "0"))

# Qwen3 enters thinking mode by default. /no_think disables it via the
# system prompt; think=False on the API call is the belt-and-suspenders.
NO_THINK_SYSTEM = "/no_think"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_logs")
LOG_FILE = os.path.join(LOG_DIR, "calls.jsonl")


def _caller_prompt_fn() -> str:
    """Walk the stack for the nearest run_gpt_* frame (profiling attribution
    + tiered routing). Returns "unknown" if none found within 30 frames."""
    try:
        frame = sys._getframe(2)
    except Exception:
        return "unknown"
    depth = 0
    while frame is not None and depth < 30:
        name = frame.f_code.co_name
        if name.startswith("run_gpt"):
            return name
        frame = frame.f_back
        depth += 1
    return "unknown"


def _get_model(call_type: str) -> str:
    if call_type in REASONING_CALL_TYPES:
        return REASONING_MODEL
    if call_type in PRIMARY_CALL_TYPES:
        return PRIMARY_MODEL
    # Unknown call_type — default to reasoning tier to be safe, but log it.
    print(f"[llm_router] WARNING: unknown call_type '{call_type}', defaulting to REASONING_MODEL")
    return REASONING_MODEL


def _log_call(model, call_type, prompt, response, schema_enforced, error=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "call_type": call_type,
        "prompt": prompt,
        "response": response,
        "schema_enforced": schema_enforced,
        "error": error,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def llm_call(prompt: str, call_type: str, json_schema: dict = None, max_retries: int = 2) -> str:
    """
    Route a prompt to the correct Ollama model based on call_type.
    If json_schema is provided, it is passed as format= to enforce structured output.
    Returns the response content as a string.
    """
    prompt_fn = _caller_prompt_fn()

    model = _get_model(call_type)
    # Tiered routing: cosmetic/noise-tolerant prompt functions go to the
    # small model unless disabled (CRSEC_TIERED_ROUTING=0).
    if (prompt_fn in SMALL_ROUTE_PROMPT_FNS
            and os.environ.get("CRSEC_TIERED_ROUTING", "1") != "0"):
        model = SMALL_MODEL

    # Routing-tier attribution (A3): one counter per logical call (before the
    # retry loop) so the small/primary split is auditable over a long run.
    if model == SMALL_MODEL:
        call_profiler.incr("small_model_call")
    else:
        call_profiler.incr("primary_model_call")

    messages = [
        {"role": "system", "content": NO_THINK_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "think": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if json_schema is not None:
        kwargs["format"] = json_schema

    last_err = None
    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            result = ollama.chat(**kwargs)
            content = result["message"]["content"]
            call_profiler.record_call(prompt_fn, time.perf_counter() - start)
            _log_call(model, call_type, prompt, content, json_schema is not None)
            return content
        except Exception as e:
            call_profiler.record_call(prompt_fn, time.perf_counter() - start)
            last_err = str(e)
            if attempt < max_retries:
                continue
            _log_call(model, call_type, prompt, "", json_schema is not None, error=last_err)
            raise


def openai_compat_call(prompt: str, call_type: str = "conversation", json_schema: dict = None) -> str:
    """
    Drop-in wrapper that mirrors the old OpenAI-style interface.
    """
    return llm_call(prompt, call_type, json_schema)
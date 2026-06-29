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

PRIMARY_MODEL = "qwen3:30b-a3b"
REASONING_MODEL = "qwen3:32b"
SMALL_MODEL = os.environ.get("CRSEC_SMALL_MODEL", "qwen3:4b")

# ── Tier 1: cosmetic / noise-tolerant prompt functions → SMALL_MODEL ─────────
# A/B: set CRSEC_TIERED_ROUTING=0 to route everything to REASONING_MODEL.
SMALL_ROUTE_PROMPT_FNS = {
    "run_gpt_prompt_pronunciatio",       # emoji cosmetics
}
# NOTE: run_gpt_prompt_event_triple was moved from SMALL → PRIMARY.
# qwen3:4b mis-parsed (s,p,o); qwen3:30b-a3b handles it cleanly.
# NOTE: the three poignancy functions were removed from small-model routing.
# They use ChatGPT_safe_generate_response's strict {"output":<int>} envelope;
# qwen3:4b doesn't emit that, so every call fail-safed. See PATCH_LOG Part B.

# ── Tier 2: agent conversations and norm decisions → REASONING_MODEL ──────────
# Any prompt_fn starting with "run_gpt_prompt_norm" is also routed here
# (matched at runtime via str.startswith).
REASONING_ROUTE_PROMPT_FNS = {
    "run_gpt_generate_iterative_chat_utt",
    "run_gpt_prompt_agent_chat",
    "run_gpt_prompt_create_conversation",
    "run_gpt_prompt_generate_next_convo_line",
    "run_gpt_prompt_agent_chat_summarize_ideas",
    "run_gpt_prompt_agent_chat_summarize_relationship",
    "run_gpt_prompt_decide_if_norm_conflict",
}

# ── Tier 3: call_type-based reasoning routing (norm module direct callers) ────
REASONING_CALL_TYPES = {
    "norm_creation",
    "norm_evaluation",
    "conflict_detection",
    "defection_assessment",
}
# Removed from REASONING_CALL_TYPES: "conversation", "default", "violation_check".
# "conversation" was the universal default but is high-volume bulk → PRIMARY.
# "violation_check" is high-volume and pre-filtered anyway → PRIMARY.

# Known bulk call_types that go to PRIMARY without logging (they are intentional,
# not unrecognized).
_KNOWN_BULK_CALL_TYPES = {
    "conversation", "default", "violation_check",
    "format_check", "type_check", "duplicate_check", "fact_consistency",
}

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
    """Tier-3/4 fallback: resolve model from call_type alone."""
    if call_type in REASONING_CALL_TYPES:
        return REASONING_MODEL
    if call_type not in _KNOWN_BULK_CALL_TYPES:
        print(f"[llm_router] info: call_type '{call_type}' → PRIMARY_MODEL")
    return PRIMARY_MODEL


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
    Route a prompt to the correct Ollama model using three-tier routing.
    If json_schema is provided, it is passed as format= to enforce structured output.
    Returns the response content as a string.

    Routing precedence (when CRSEC_TIERED_ROUTING != "0"):
      1. prompt_fn in SMALL_ROUTE_PROMPT_FNS                            → SMALL_MODEL
      2. prompt_fn in REASONING_ROUTE_PROMPT_FNS
         or prompt_fn.startswith("run_gpt_prompt_norm")                 → REASONING_MODEL
      3. call_type in REASONING_CALL_TYPES                              → REASONING_MODEL
      4. else                                                            → PRIMARY_MODEL
    When CRSEC_TIERED_ROUTING=0: everything → REASONING_MODEL (A/B baseline).
    """
    prompt_fn = _caller_prompt_fn()

    tiered = os.environ.get("CRSEC_TIERED_ROUTING", "1") != "0"
    if not tiered:
        model = REASONING_MODEL
    elif prompt_fn in SMALL_ROUTE_PROMPT_FNS:
        model = SMALL_MODEL
    elif (prompt_fn in REASONING_ROUTE_PROMPT_FNS
          or prompt_fn.startswith("run_gpt_prompt_norm")):
        model = REASONING_MODEL
    else:
        model = _get_model(call_type)

    # Routing-tier attribution: one counter per logical call (before retry loop)
    # so the three-way split is auditable over a long run.
    if model == SMALL_MODEL:
        call_profiler.incr("small_model_call")
    elif model == REASONING_MODEL:
        call_profiler.incr("reasoning_model_call")
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
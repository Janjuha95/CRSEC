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
# Spoken-line generators and norm-conflict decisions stay on the 32b; the two
# agent-chat summarize fns were moved to PRIMARY — they are recall aids, not
# decisions, and they fire up to 10x per conversation. Set
# CRSEC_SUMMARIZE_ON_REASONING=1 to put them back on the 32b.
REASONING_ROUTE_PROMPT_FNS = {
    "run_gpt_generate_iterative_chat_utt",
    "run_gpt_prompt_agent_chat",
    "run_gpt_prompt_create_conversation",
    "run_gpt_prompt_generate_next_convo_line",
    "run_gpt_prompt_decide_if_norm_conflict",
}
if os.environ.get("CRSEC_SUMMARIZE_ON_REASONING", "0") == "1":
    REASONING_ROUTE_PROMPT_FNS |= {
        "run_gpt_prompt_agent_chat_summarize_ideas",
        "run_gpt_prompt_agent_chat_summarize_relationship",
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

# ── Output-token caps (num_predict), keyed by prompt_fn ──────────────────────
# Generation tokens dominate wall time, and without num_predict a
# classifier-style call can ramble unboundedly. Caps are sized from each fn's
# parser contract (what the validate/clean_up pair actually needs to see) plus
# the legacy gpt_param max_tokens as intent evidence — llm_logs/calls.jsonl
# has no prompt_fn field, so data-derived p99 caps were not possible.
# Floor is 32: Qwen3 may spend a few tokens on an empty <think></think>
# preamble even with think=False.
# A too-tight cap is WORSE than none: truncation → parse failure → the
# safe_generate repeat loop re-fires the call 3-5x. The done_reason=="length"
# tripwire below (truncated_* counters) is how a bad cap shows up in the next
# calibration run.
# Kill-switch: CRSEC_NUM_PREDICT=0 disables all caps.
# Unlisted fns (incl. prompt_fn=="unknown" from direct llm_call callers such
# as defection_engine/creation.py) get CRSEC_NUM_PREDICT_DEFAULT.
PROMPT_FN_NUM_PREDICT = {
    # -- int-score / yes-no / option pickers (bare token + slack) --
    "run_gpt_prompt_event_poignancy": 32,
    "run_gpt_prompt_thought_poignancy": 32,
    "run_gpt_prompt_chat_poignancy": 32,
    "run_gpt_prompt_decide_to_talk": 32,
    "run_gpt_prompt_decide_to_react": 32,
    "run_gpt_prompt_wake_up_hour": 32,
    "run_gpt_generate_safety_score": 32,       # strict {"output": N} json
    "run_gpt_norm_duplicate_check": 32,        # template: ONLY "YES"/"NO"
    "run_gpt_prompt_pronunciatio": 32,         # emoji; 16 would breach the floor
    # -- event triples "(s, p, o)" --
    "run_gpt_prompt_event_triple": 64,
    "run_gpt_prompt_act_obj_event_triple": 64,
    # -- short names / phrases --
    "run_gpt_prompt_action_sector": 64,
    "run_gpt_prompt_action_arena": 64,
    "run_gpt_prompt_action_game_object": 64,
    "run_gpt_prompt_act_obj_desc": 64,
    # -- one-liners (legacy max_tokens 40-50) --
    "run_gpt_prompt_summarize_conversation": 128,
    "run_gpt_prompt_extract_keywords": 128,
    "run_gpt_prompt_keyword_to_thoughts": 128,
    "run_gpt_prompt_convo_to_thoughts": 128,
    "run_gpt_prompt_generate_whisper_inner_thought": 128,
    "run_gpt_prompt_planning_thought_on_convo": 128,
    "run_gpt_prompt_memo_on_convo": 128,
    "run_gpt_prompt_generate_hourly_schedule": 128,  # single schedule entry, not a full plan
    # -- yes/no WITH required reasoning or structured segments; capping at 32
    #    would truncate before the parser's markers appear --
    "run_gpt_prompt_violation_check": 128,     # 4-key json {violation,severity,certainty,response}
    "run_gpt_norm_recognize_conflict_check": 128,   # "Answer: yes" + short reason
    "run_gpt_seeds_content_check": 256,        # Answer 1###/Answer 2###/STAGE 1 segments
    "run_gpt_seeds_type_check": 256,           # OUTPUT + type-classification echo
    "run_gpt_seeds_type_check_v2": 256,        # STEP 1/STEP 2/type lines
    "run_gpt_norm_fact_consistency_check": 256,  # Answer + full "New norm:" text
    "run_gpt_norm_recognize": 256,             # Answer 1..4 lines
    "run_gpt_immediate_evaluate_recognization": 256,  # NORM UTILITY + ANSWER lines
    "run_gpt_norm_utility": 256,               # "OUTPUT: N. <reason>" (reason is consumed)
    "run_gpt_long_term_norm_utility": 256,     # FINAL SCORE + rationale
    # -- summaries --
    "run_gpt_prompt_agent_chat_summarize_ideas": 256,
    "run_gpt_prompt_agent_chat_summarize_relationship": 256,
    "run_gpt_prompt_summarize_ideas": 256,
    "run_gpt_chat_norms_summarize": 256,
    "run_gpt_conflict_chat_reflect": 256,
    "run_gpt_non_norm_conflict_chat_reflect": 256,
    "run_gpt_prompt_focal_pt": 256,
    "run_gpt_active_norms_classfication": 256,
    "run_gpt_revise_identity_plan": 256,
    "run_gpt_revise_identity_thought": 256,
    "run_gpt_revise_identity_currently": 256,
    "run_gpt_revise_identity_daily_plan_req": 256,
    # -- dialogue json --
    "run_gpt_generate_iterative_chat_utt": 512,
    "run_gpt_prompt_agent_chat": 512,
    "run_gpt_prompt_create_conversation": 512,
    "run_gpt_prompt_generate_next_convo_line": 512,
    "run_gpt_active_norms_classfication_v2": 512,  # free-form, passed through raw
    # -- norm creation / format / long-term synthesis --
    "run_gpt_prompt_norm_reflect_from_thoughts": 768,
    "run_gpt_prompt_norm_format": 768,
    "run_gpt_norm_long_term_synthesis": 768,
    # check_conflict_decide_talk_v5 demands step-by-step reasoning for three
    # questions before FINAL OUTPUT, and _final_output_decision's fallback
    # bottom-scans for any yes/no — a truncated response could silently invert
    # the decision instead of failing. Keep this one roomy.
    "run_gpt_prompt_decide_if_norm_conflict": 768,
    # -- planning / schedule / decomposition --
    "run_gpt_prompt_daily_plan": 2048,
    "run_gpt_prompt_daily_plan_v2": 2048,
    "run_gpt_prompt_task_decomp": 2048,
    "run_gpt_prompt_task_decomp_v2": 2048,
    "run_gpt_prompt_new_decomp_schedule": 2048,
    "run_gpt_prompt_insight_and_guidance": 1024,  # n insights + evidence (legacy 1500)
}

NUM_PREDICT_ENABLED = os.environ.get("CRSEC_NUM_PREDICT", "1") != "0"
NUM_PREDICT_DEFAULT = int(os.environ.get("CRSEC_NUM_PREDICT_DEFAULT", "1024"))

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
        {"role": "user", "content": prompt + "\n/no_think"},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if NUM_PREDICT_ENABLED:
        kwargs["options"]["num_predict"] = PROMPT_FN_NUM_PREDICT.get(
            prompt_fn, NUM_PREDICT_DEFAULT)
    if json_schema is not None:
        kwargs["format"] = json_schema

    last_err = None
    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            result = ollama.chat(**kwargs)
            content = result["message"]["content"]
            call_profiler.record_call(prompt_fn, time.perf_counter() - start)
            # Truncation tripwire: a response cut off by num_predict will
            # usually fail its parser and get re-fired by the safe_generate
            # repeat loop — i.e. a too-tight cap makes things SLOWER. The
            # truncated_* counters are how that shows up in the next
            # calibration run's profile.json.
            truncated = None
            try:
                if result.get("done_reason") == "length":
                    call_profiler.incr(f"truncated_{prompt_fn}")
                    truncated = ("truncated: done_reason=length "
                                 f"(num_predict={kwargs['options'].get('num_predict')})")
            except Exception:
                pass
            _log_call(model, call_type, prompt, content, json_schema is not None,
                      error=truncated)
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
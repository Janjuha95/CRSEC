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

PRIMARY_MODEL = "qwen3:30b-instruct"
REASONING_MODEL = "qwen3:32b"
SMALL_MODEL = os.environ.get("CRSEC_SMALL_MODEL", "qwen3:4b-instruct")

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
# Spoken-line generators stay on the 32b; the two agent-chat summarize fns
# were moved to PRIMARY — they are recall aids, not decisions, and they fire
# up to 10x per conversation. Set CRSEC_SUMMARIZE_ON_REASONING=1 to put them
# back on the 32b.
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

# ── A/B flag: the two dominant norm fns → PRIMARY_MODEL ──────────────────────
# calib_008: decide_if_norm_conflict (264x @ 8.7s avg) + norm_reflect_from_
# thoughts (71x @ 21.8s avg) = 3,845s of 7,661s total LLM time (~50%), both
# on the dense 32b. CRSEC_NORM_ON_PRIMARY=1 (default) routes exactly these
# two to PRIMARY_MODEL; checked BEFORE the tier-2 set and the
# "run_gpt_prompt_norm" prefix rule, which would otherwise catch them.
# All other norm_* fns stay on the 32b. Set CRSEC_NORM_ON_PRIMARY=0 to
# restore the previous routing exactly. Read at call time (like
# CRSEC_TIERED_ROUTING) so the flag can be flipped without re-import.
NORM_PRIMARY_OVERRIDE_FNS = {
    "run_gpt_prompt_decide_if_norm_conflict",
    "run_gpt_prompt_norm_reflect_from_thoughts",
    # calib_011: 254 utility-scorer calls x 8.5s on the 32b (as prompt_fn
    # "unknown") were the #1 remaining cost. Same norm-reasoning tier as the
    # two fns above; flag=0 restores its old 32b routing exactly (it falls
    # back to the tier-3 call_type "norm_evaluation" rule).
    "run_gpt_specific_norm_utility",
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

# ── Output-token caps (num_predict), keyed by prompt_fn — v2 ────────────────
# v1 was sized from parser contracts + legacy max_tokens because the old
# calls.jsonl had no way to attribute calls; it truncated live planning calls
# and was disabled via CRSEC_NUM_PREDICT=0. v2 is derived from calib_008
# (2,249 calls attributed per-fn by template fingerprint; per-fn p99 response
# length, tokens ≈ chars/3.5, cap = max(64, ceil(2 × p99_tokens))).
#
# TIGHT caps go ONLY to truncation-safe fns — parsers verified to read from
# the TOP of the output (first int / first tuple / option token), where a
# post-payload cut cannot corrupt the parse. Everything else (plans,
# schedules, dialogue, bottom-scanning yes/no deciders, whole-string
# summaries, norm reflection/format/evaluation) gets a blanket 4096: pure
# runaway protection, comfortably above every measured p99 (largest:
# new_decomp_schedule ≈ 3,407 tok).
# A too-tight cap is WORSE than none: truncation → parse failure → the
# safe_generate repeat loop re-fires the call 3-5x. The done_reason=="length"
# tripwire below (truncated_* counters) is how a bad cap shows up in the next
# calibration run.
# Kill-switch: CRSEC_NUM_PREDICT=0 disables all caps (calib_009 should run
# with caps ENABLED — leave CRSEC_NUM_PREDICT unset).
# Unlisted fns (incl. prompt_fn=="unknown" from direct llm_call callers such
# as defection_engine/creation.py) get CRSEC_NUM_PREDICT_DEFAULT.
PROMPT_FN_NUM_PREDICT = {
    # -- data-derived tight caps: top-reading parsers only (p99 tok in note) --
    "run_gpt_prompt_event_poignancy": 64,      # p99 4; first-int
    "run_gpt_prompt_thought_poignancy": 64,    # no calib_008 calls; same parser family as the other poignancy fns
    "run_gpt_prompt_chat_poignancy": 64,       # p99 4; first-int
    "run_gpt_prompt_wake_up_hour": 614,        # p99 307; first-int, hour leads the prose
    "run_gpt_prompt_action_sector": 64,        # p99 7; option token
    "run_gpt_prompt_action_arena": 64,         # p99 2; option token
    "run_gpt_prompt_action_game_object": 96,   # p99 48; option token
    "run_gpt_prompt_event_triple": 530,        # p99 265; tuple leads even rambling responses
    "run_gpt_prompt_act_obj_event_triple": 105,  # p99 52; first tuple
    "run_gpt_prompt_violation_check": 64,      # p99 21; one-line 4-key json
    "run_gpt_prompt_pronunciatio": 64,         # stubbed in-sim; emoji floor
    # calib_011 (341 captured calls incl. the calib_010 carryover): p99 4745
    # chars ≈ 1356 tok; score+reason lead the response (top-reader), and the
    # old "unknown" default of 1024 truncated 4 responses.
    "run_gpt_specific_norm_utility": 2712,
    # -- blanket runaway protection: NOT truncation-safe (bottom-scan yes/no,
    #    whole-string consumers, json whose loss silently drops data) or no
    #    calib_008 data (norm_evaluate family idle while the seed pipeline
    #    was stalled pre-Change-A) --
    "run_gpt_prompt_decide_to_talk": 4096,     # p99 732; bottom-scan Answer: yes/no
    "run_gpt_prompt_decide_to_react": 4096,    # p99 983; bottom-scan
    "run_gpt_prompt_decide_if_norm_conflict": 4096,  # p99 679; FINAL OUTPUT bottom-scan
    "run_gpt_prompt_norm_reflect_from_thoughts": 4096,  # p99 1917
    "run_gpt_prompt_norm_format": 4096,        # p99 192, but truncated json = dropped norm
    "run_gpt_prompt_focal_pt": 4096,           # p99 525; envelope needs closing brace
    "run_gpt_prompt_new_decomp_schedule": 4096,  # p99 3407 — the biggest producer
    "run_gpt_prompt_daily_plan": 4096,
    "run_gpt_prompt_daily_plan_v2": 4096,      # p99 1612
    "run_gpt_prompt_task_decomp": 4096,        # p99 1440 (with _v2 combined)
    "run_gpt_prompt_task_decomp_v2": 4096,
    "run_gpt_prompt_generate_hourly_schedule": 4096,  # p99 165 but max ~1694
    "run_gpt_prompt_insight_and_guidance": 4096,  # p99 208; numbered list consumed whole
    "run_gpt_generate_iterative_chat_utt": 4096,  # p99 235; dialogue json
    "run_gpt_prompt_agent_chat": 4096,
    "run_gpt_prompt_create_conversation": 4096,
    "run_gpt_prompt_generate_next_convo_line": 4096,
    "run_gpt_prompt_agent_chat_summarize_ideas": 4096,
    "run_gpt_prompt_agent_chat_summarize_relationship": 4096,  # p99 541
    "run_gpt_prompt_summarize_ideas": 4096,
    "run_gpt_prompt_summarize_conversation": 4096,  # p99 118; whole-string
    "run_gpt_prompt_act_obj_desc": 4096,       # p99 91; whole-string
    "run_gpt_prompt_extract_keywords": 4096,
    "run_gpt_prompt_keyword_to_thoughts": 4096,
    "run_gpt_prompt_convo_to_thoughts": 4096,
    "run_gpt_prompt_generate_whisper_inner_thought": 4096,
    "run_gpt_prompt_planning_thought_on_convo": 4096,  # p99 158
    "run_gpt_prompt_memo_on_convo": 4096,      # p99 98
    "run_gpt_generate_safety_score": 4096,
    "run_gpt_chat_norms_summarize": 4096,      # p99 53; list consumed whole
    "run_gpt_conflict_chat_reflect": 4096,     # p99 638; Step 3 bottom-scan
    "run_gpt_non_norm_conflict_chat_reflect": 4096,  # p99 237
    "run_gpt_norm_duplicate_check": 4096,
    "run_gpt_norm_recognize_conflict_check": 4096,
    "run_gpt_seeds_content_check": 4096,
    "run_gpt_seeds_type_check": 4096,
    "run_gpt_seeds_type_check_v2": 4096,
    "run_gpt_norm_fact_consistency_check": 4096,
    "run_gpt_norm_recognize": 4096,
    "run_gpt_immediate_evaluate_recognization": 4096,
    "run_gpt_norm_utility": 4096,
    "run_gpt_long_term_norm_utility": 4096,
    "run_gpt_norm_long_term_synthesis": 4096,
    "run_gpt_active_norms_classfication": 4096,
    "run_gpt_active_norms_classfication_v2": 4096,
    "run_gpt_revise_identity_plan": 4096,
    "run_gpt_revise_identity_thought": 4096,
    "run_gpt_revise_identity_currently": 4096,
    "run_gpt_revise_identity_daily_plan_req": 4096,
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


def llm_call(prompt: str, call_type: str, json_schema: dict = None, max_retries: int = 2,
             prompt_fn: str = None) -> str:
    """
    Route a prompt to the correct Ollama model using three-tier routing.
    If json_schema is provided, it is passed as format= to enforce structured output.
    prompt_fn overrides the stack-walk attribution — direct llm_call callers
    (SpecificNormUtility, defection_engine, creation) have no run_gpt_* frame
    and otherwise profile as "unknown" with default caps/routing.
    Returns the response content as a string.

    Routing precedence (when CRSEC_TIERED_ROUTING != "0"):
      1. prompt_fn in SMALL_ROUTE_PROMPT_FNS                            → SMALL_MODEL
      2. CRSEC_NORM_ON_PRIMARY!=0 and fn in NORM_PRIMARY_OVERRIDE_FNS  → PRIMARY_MODEL
         (explicit exclusion, checked BEFORE the norm prefix rule below)
      3. prompt_fn in REASONING_ROUTE_PROMPT_FNS
         or prompt_fn.startswith("run_gpt_prompt_norm")                 → REASONING_MODEL
      4. call_type in REASONING_CALL_TYPES                              → REASONING_MODEL
      5. else                                                            → PRIMARY_MODEL
    When CRSEC_TIERED_ROUTING=0: everything → REASONING_MODEL (A/B baseline).
    """
    if prompt_fn is None:
        prompt_fn = _caller_prompt_fn()

    tiered = os.environ.get("CRSEC_TIERED_ROUTING", "1") != "0"
    norm_on_primary = os.environ.get("CRSEC_NORM_ON_PRIMARY", "1") != "0"
    if not tiered:
        model = REASONING_MODEL
    elif prompt_fn in SMALL_ROUTE_PROMPT_FNS:
        model = SMALL_MODEL
    elif norm_on_primary and prompt_fn in NORM_PRIMARY_OVERRIDE_FNS:
        model = PRIMARY_MODEL
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
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if NUM_PREDICT_ENABLED:
        kwargs["options"]["num_predict"] = PROMPT_FN_NUM_PREDICT.get(
            prompt_fn, NUM_PREDICT_DEFAULT)
    if model == REASONING_MODEL:
        # qwen3:32b is an original hybrid: think=False cleanly disables thinking.
        # NEVER send think to instruct-2507 (unsupported) and never think=False
        # to thinking-2507 builds (leaks reasoning into content).
        kwargs["think"] = False

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
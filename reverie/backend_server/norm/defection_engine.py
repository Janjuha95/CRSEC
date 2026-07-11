"""
File: defection_engine.py
Description: Dual-utility function for strategic defector agents (Modification 2).

Defector agents evaluate each norm through a self-interested cost-benefit lens
rather than treating norms as group obligations.
"""
import sys

sys.path.append('../')

from llm_router import llm_call
from persona.prompt_template.gpt_structure import generate_prompt


DEFECTION_PROMPT = "norm/defection_prompt/defection_assessment_v1.txt"
DEFECTOR_NORM_UTILITY_PROMPT = "norm/defection_prompt/specific_norm_utility_defector_v1.txt"

# (persona_name, norm_content, sim_date) -> (decision, reasoning).
# Growth is bounded: defectors x active norms x sim days (~50/day in cond C).
_DECISION_CACHE = {}


def decide_defection_cached(persona, norm, context_dict, metrics=None):
    """One defection decision per defector, norm, and sim-day.

    Every behavior-shaping act-norm injection site (daily plan, hourly
    schedule, task decomposition, interview/convo prompt building) consults
    this instead of calculate_defection_utility directly, so a defector makes
    ONE LLM-backed decision per (norm, sim-day) no matter how many prompts
    are built that day. Cache hits return the stored (decision, reasoning)
    with no LLM call and no duplicate metrics logging. Non-defectors
    short-circuit without touching the cache.
    """
    scratch = persona.scratch
    # Same source of truth as the gate sites (scratch.is_defector());
    # agent_type fallback matches calculate_defection_utility's own check.
    if hasattr(scratch, "is_defector"):
        is_def = scratch.is_defector()
    else:
        is_def = getattr(scratch, "agent_type", "citizen") == "defector"
    if not is_def:
        return "comply", "Not a defector"
    curr_time = getattr(persona.scratch, "curr_time", None)
    if curr_time is None:
        # no sim clock yet (bootstrap): fall back to an uncached decision
        return calculate_defection_utility(persona, norm, context_dict,
                                           metrics=metrics)
    key = (persona.scratch.name, norm.content, curr_time.date())
    if key not in _DECISION_CACHE:
        _DECISION_CACHE[key] = calculate_defection_utility(
            persona, norm, context_dict, metrics=metrics)
    return _DECISION_CACHE[key]


def _parse_decision(response):
    """Pull Decision / Reasoning out of the LLM response. Defaults to comply
    on any parse failure (fail-safe)."""
    decision = None
    reasoning = ""
    for raw_line in response.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("decision:"):
            value = line.split(":", 1)[1].strip().lower()
            value = value.strip("[]").strip()
            if "defect" in value:
                decision = "defect"
            elif "comply" in value:
                decision = "comply"
        elif lowered.startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    if decision is None:
        return "comply", "LLM response did not specify a decision; defaulting to comply."
    if not reasoning:
        reasoning = "No reasoning provided."
    return decision, reasoning


def calculate_defection_utility(persona, norm, context_dict, metrics=None):
    """
    Decide whether a defector persona will comply with or defect from a norm.

    Args:
        persona: persona object (uses persona.scratch).
        norm: norm node (uses norm.content).
        context_dict: dict with optional keys "nearby_agents", "description".
        metrics: optional MetricsCollector; logs the decision when provided.

    Returns:
        (decision, reasoning) where decision is "comply" or "defect".
    """
    if getattr(persona.scratch, "agent_type", "citizen") != "defector":
        return "comply", "Not a defector"

    prompt_input = [
        norm.content,
        persona.scratch.get_str_iss(),
        str(persona.scratch.boldness),
        str(persona.scratch.reputation_concern),
        str(persona.scratch.trust_score),
        str(context_dict.get("nearby_agents", "none")),
        context_dict.get("description", "daily planning"),
    ]

    try:
        prompt = generate_prompt(prompt_input, DEFECTION_PROMPT)
        response = llm_call(prompt, call_type="defection_assessment",
                            prompt_fn="run_gpt_defection_assessment")
    except Exception as e:
        decision, reasoning = "comply", f"Defection assessment failed ({e}); defaulting to comply."
    else:
        decision, reasoning = _parse_decision(response)

    if metrics:
        metrics.log_defection_attempt(
            agent_name=persona.scratch.name,
            agent_identity=persona.scratch.identity,
            norm_content=norm.content,
            decision=decision,
            reasoning=reasoning,
            boldness=persona.scratch.boldness,
            trust_score=persona.scratch.trust_score,
            step=getattr(persona.scratch, 'curr_time', None),
        )

    return decision, reasoning


def get_defector_norm_utility(norm_content, persona):
    """
    Self-interested utility score for a norm, from a defector's perspective.

    Mirrors SpecificNormUtility.specific_norm_utility so callers can treat the
    return value identically: a length-2 [score:int, reason:str] on success,
    and a shape-matched [int, reason] fail-safe (never a bare False or [False])
    on failure, so the consumer's `len(utility) == 2` check always holds.
    """
    fail_safe = [4, "fail_safe"]
    try:
        prompt = generate_prompt([norm_content], DEFECTOR_NORM_UTILITY_PROMPT)
        response = llm_call(prompt, call_type="norm_evaluation",
                            prompt_fn="run_gpt_defector_norm_utility")
    except Exception:
        return fail_safe

    try:
        tail = response.split("OUTPUT: ")[-1]
        score = int(tail.split(".")[0].strip())
        reason = tail.split(". ", 1)[1].strip() if ". " in tail else ""
        return [score, reason]
    except Exception:
        return fail_safe

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


def calculate_defection_utility(persona, norm, context_dict):
    """
    Decide whether a defector persona will comply with or defect from a norm.

    Args:
        persona: persona object (uses persona.scratch).
        norm: norm node (uses norm.content).
        context_dict: dict with optional keys "nearby_agents", "description".

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
        response = llm_call(prompt, call_type="defection_assessment")
    except Exception as e:
        return "comply", f"Defection assessment failed ({e}); defaulting to comply."

    return _parse_decision(response)


def get_defector_norm_utility(norm_content, persona):
    """
    Self-interested utility score for a norm, from a defector's perspective.

    Mirrors SpecificNormUtility.specific_norm_utility so callers can treat the
    return value identically: [score:int, reason:str] on success, [False] on failure.
    """
    try:
        prompt = generate_prompt([norm_content], DEFECTOR_NORM_UTILITY_PROMPT)
        response = llm_call(prompt, call_type="norm_evaluation")
    except Exception:
        return [False]

    try:
        tail = response.split("OUTPUT: ")[-1]
        score = int(tail.split(".")[0].strip())
        reason = tail.split(". ", 1)[1].strip() if ". " in tail else ""
        return [score, reason]
    except Exception:
        return [False]

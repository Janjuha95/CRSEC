"""
File: violation_detection.py
Description: Norm-violation perception and response pipeline.

detect_violations  — inspects perceived events for norm violations by other agents.
process_violations — updates observer trust, logs, and triggers confront/gossip/ignore.
"""
import sys

sys.path.append('../')

from norm.run_gpt_prompt_norm import run_gpt_prompt_violation_check


def detect_violations(observer_persona, perceived_events, personas):
    """
    Inspect perceived events and return a list of detected violations.

    A perceived event is a ConceptNode with .subject, .predicate, .object,
    .description. Events authored by the observer themselves are skipped, and
    only events whose subject is another persona name are checked (so ambient
    world events like "the cafe is open" do not burn LLM calls).

    Returns a list of dicts:
        {"violator", "norm", "event", "severity", "certainty", "response"}
    """
    if perceived_events is None:
        return []

    observer_name = observer_persona.scratch.name
    persona_names = set(personas.keys()) if hasattr(personas, "keys") else set(personas)

    active_norms = []
    for norm_id, a_norm in observer_persona.norm_database.act_norm.items():
        if a_norm.activation_state == True:
            active_norms.append(a_norm)
    if not active_norms:
        return []

    violations = []
    for event in perceived_events:
        subject = getattr(event, "subject", None)
        predicate = getattr(event, "predicate", None) or ""
        obj = getattr(event, "object", None) or ""

        if not subject or subject == observer_name:
            continue
        if subject not in persona_names:
            continue

        event_desc = f"{subject} is {predicate} {obj}".strip()

        for norm in active_norms:
            try:
                result = run_gpt_prompt_violation_check(
                    event_desc, norm.content, observer_name)[0]
            except Exception:
                continue

            if not isinstance(result, dict):
                continue
            if not result.get("violation"):
                continue

            violations.append({
                "violator": subject,
                "norm": norm,
                "event": event_desc,
                "severity": result.get("severity", 0),
                "certainty": result.get("certainty", 0),
                "response": result.get("response", "ignore"),
            })

    return violations


def process_violations(observer_persona, violations, personas, reputation_system, metrics=None):
    """
    Apply observer-side trust decay, log, and dispatch the suggested response
    (confront / gossip / ignore) for each detected violation.
    """
    if not violations:
        return

    observer_name = observer_persona.scratch.name

    for v in violations:
        violator = v["violator"]
        severity = v.get("severity", 0) or 0
        certainty = v.get("certainty", 0) or 0
        response = v.get("response", "ignore")

        decay = (severity * certainty) / 100.0

        current = observer_persona.scratch.reputation_beliefs.get(violator, 100)
        new_val = current - decay
        if new_val < 0:
            new_val = 0
        observer_persona.scratch.reputation_beliefs[violator] = new_val

        if reputation_system is not None:
            reputation_system.update_trust(observer_name, violator, -decay)

        norm_obj = v.get("norm")
        observer_persona.scratch.observed_violations.append({
            "violator": violator,
            "norm_id": getattr(norm_obj, "id", None),
            "norm_content": getattr(norm_obj, "content", None),
            "event": v.get("event"),
            "severity": severity,
            "certainty": certainty,
            "response": response,
        })

        if response == "confront":
            observer_persona.scratch.norm_conflict = True
        elif response == "gossip":
            if reputation_system is not None:
                reputation_system.spread_gossip(observer_name, violator, severity, personas)
        # "ignore" -> nothing further

        if metrics:
            norm_content = getattr(norm_obj, "content", None)
            step = getattr(observer_persona.scratch, 'curr_time', None)
            metrics.log_violation(
                observer=observer_name,
                violator=violator,
                norm_content=norm_content,
                severity=severity,
                certainty=certainty,
                step=step,
            )
            metrics.log_enforcement(
                enforcer=observer_name,
                target=violator,
                action_type=response,
                norm_content=norm_content,
                step=step,
            )

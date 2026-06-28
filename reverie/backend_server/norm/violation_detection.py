"""
File: violation_detection.py
Description: Norm-violation perception and response pipeline.

detect_violations  — inspects perceived events for norm violations by other agents.
process_violations — updates observer trust, logs, and triggers confront/gossip/ignore.
"""
import os
import re
import sys

sys.path.append('../')

import call_profiler
from norm.run_gpt_prompt_norm import run_gpt_prompt_violation_check


# ----------------------------------------------------------------------------
# Violation-check pre-filter (Step 3 of the perf pass). Flag-gated, default
# ON; disable with CRSEC_VIOLATION_PREFILTER=0. Two stages before any LLM
# call:
#   (a) benign-pattern skip: events that cannot constitute a norm violation
#       (idle / sleeping / waiting / routine chatting);
#   (b) keyword-stem overlap: the event description must share at least one
#       content-word stem with the norm (built from the norm's
#       subject/predicate/object fields, falling back to norm.content).
# Skip counts are recorded in profile.json (counters: prefilter_skip_benign,
# prefilter_skip_no_overlap, prefilter_llm_checked) to audit reach.
# ----------------------------------------------------------------------------

_BENIGN_PATTERNS = (
    "is chat with",
    "chatting with",
    "conversing about",
    "is idle",
    "sleeping",
    "<waiting",
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "being", "been",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "not", "no",
    "do", "does", "did", "doing", "have", "has", "had", "it", "its", "this",
    "that", "these", "those", "by", "from", "as", "into", "their", "his",
    "her", "they", "them", "he", "she", "you", "your", "we", "our", "i",
    "should", "must", "shall", "will", "would", "can", "could", "may",
    "might", "one", "everyone", "anyone", "people", "person", "all", "any",
    "allowed", "allow", "while", "when", "during", "other", "others",
}


def _stem(word):
    """Crude suffix-stripping stem; good enough for overlap screening."""
    w = word.lower()
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:len(w) - len(suffix)]
            break
    return w


def _stems(text):
    """Set of content-word stems from free text."""
    out = set()
    for word in re.findall(r"[a-zA-Z]+", text or ""):
        if word.lower() in _STOPWORDS or len(word) < 3:
            continue
        out.add(_stem(word))
    return out


def _norm_stems(norm):
    """Stem set for a norm, cached on the norm object."""
    cached = getattr(norm, "_prefilter_stems", None)
    if cached is not None:
        return cached
    text = " ".join(str(getattr(norm, field, "") or "")
                    for field in ("subject", "predicate", "object"))
    stems = _stems(text)
    if not stems:  # malformed/empty s-p-o: fall back to the full content
        stems = _stems(getattr(norm, "content", "") or "")
    try:
        norm._prefilter_stems = stems
    except Exception:
        pass
    return stems


def _is_benign_event(event_desc):
    low = event_desc.lower()
    return any(p in low for p in _BENIGN_PATTERNS)


def _prefilter_enabled():
    return os.environ.get("CRSEC_VIOLATION_PREFILTER", "1") != "0"


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

        prefilter = _prefilter_enabled()

        # Pre-filter stage (a): benign events can't violate a norm.
        if prefilter and _is_benign_event(event_desc):
            call_profiler.incr("prefilter_skip_benign", len(active_norms))
            continue

        event_stems = _stems(event_desc) if prefilter else None

        for norm in active_norms:
            # Pre-filter stage (b): require at least one stem overlap between
            # the event and the norm; empty norm stems fail open (LLM check).
            if prefilter:
                norm_stems = _norm_stems(norm)
                if norm_stems and not (event_stems & norm_stems):
                    call_profiler.incr("prefilter_skip_no_overlap")
                    continue
                call_profiler.incr("prefilter_llm_checked")

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

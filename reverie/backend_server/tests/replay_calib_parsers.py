"""Replay a calibration run's captured LLM responses through the current
parsers (Change A acceptance test).

Usage (from reverie/backend_server):
    python tests/replay_calib_parsers.py [path/to/calls.jsonl]

Default calls.jsonl path: <repo root>/calib_artifacts/calib_008/calls.jsonl
(the calib_008 artifacts are gitignored; run this on a machine that has them).

calls.jsonl has no prompt_fn field, so calls are attributed to run_gpt_*
functions by fingerprinting the static text of each fn's prompt template.
Attribution was verified exact against calib_008's profile.json counts for
all 29 functions with calls.

Acceptance (calib_008 baseline, parsers as of the Change A commit):
    event_triple          396/396  100%   (was   0%)
    act_obj_event_triple  170/170  100%   (was   0%)
    norm_format             75/75  100%   (was   0%)
    new_decomp_schedule      85/90  94.4% (was   0%)  see note
    focal_pt                 72/72  100%   (was  25%)
Note: the 5 new_decomp residuals are ONE degenerate response (the model
emitted 18 minutes of a 120-minute window, then em-space padding) repeated
across its 5 retry attempts; there is no schedule in it to parse and the
fail_safe reconstruction is the correct outcome. 85/85 responses that
contain a full schedule parse.
"""
import json
import os
import re
import sys
from collections import defaultdict

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(BACKEND))
sys.path.insert(0, BACKEND)
# Resolve a relative calls.jsonl argument against the caller's cwd BEFORE
# chdir-ing into backend_server (templates are read via relative paths).
if len(sys.argv) > 1:
    sys.argv[1] = os.path.abspath(sys.argv[1])
os.chdir(BACKEND)

from persona.prompt_template.run_gpt_prompt import (  # noqa: E402
    _parse_event_triple_completion,
    _parse_new_decomp_schedule,
    _parse_focal_pt_candidate,
)
from persona.prompt_template.gpt_structure import (  # noqa: E402
    _strip_scaffolding, _extract_yes_no,
)
from norm.run_gpt_prompt_norm import (  # noqa: E402
    _parse_norm_format_json,
    _parse_seeds_type_check,
)

# fn -> active template(s). event_triple and act_obj_event_triple share one
# template and are split on whether the primed subject is a persona name.
FN_TEMPLATES = {
    "EVENT_TRIPLE_FAMILY": ["persona/prompt_template/v2/generate_event_triple_v1.txt"],
    "run_gpt_prompt_new_decomp_schedule": ["persona/prompt_template/v2/new_decomp_schedule_v1.txt"],
    "run_gpt_prompt_focal_pt": ["persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt",
                                "persona/prompt_template/v2/generate_focal_pt_v1.txt"],
    "run_gpt_prompt_norm_format": ["norm/norm_identify_prompt/identify_norm_save_v3.txt"],
    # seed-evaluation chain (calib_009 coverage; zero calls in calib_008)
    "run_gpt_seeds_type_check_v2": ["norm/norm_evaluate_prompt/seeds_type_check_v3.txt"],
    "run_gpt_norm_duplicate_check": ["norm/norm_evaluate_prompt/duplicate_check_v1.txt"],
    "run_gpt_norm_fact_consistency_check": ["norm/norm_evaluate_prompt/fact_consistency_check_v1.txt"],
    "run_gpt_norm_recognize_conflict_check": ["norm/norm_evaluate_prompt/recognize_conflict_check_v1.txt"],
}


def _template_chunks(path):
    with open(os.path.join(BACKEND, path), encoding="utf-8") as f:
        body = f.read()
    if "<commentblockmarker>###</commentblockmarker>" in body:
        body = body.split("<commentblockmarker>###</commentblockmarker>")[1]
    parts = re.split(r"!<INPUT \d+>!", body)
    chunks = sorted((p.strip() for p in parts if len(p.strip()) >= 25),
                    key=len, reverse=True)
    return chunks[:3] or [body.strip()[:80]]


FINGERPRINTS = {fn: [_template_chunks(p) for p in paths]
                for fn, paths in FN_TEMPLATES.items()}


def _attribute(prompt, persona_names):
    best, best_len = None, 0
    for fn, variants in FINGERPRINTS.items():
        for chunks in variants:
            if all(c in prompt for c in chunks):
                score = sum(len(c) for c in chunks)
                if score > best_len:
                    best, best_len = fn, score
    if best == "EVENT_TRIPLE_FAMILY":
        m = re.findall(r"Input: (.+?) is ", prompt)
        subj = m[-1] if m else ""
        if subj in persona_names:
            return "run_gpt_prompt_event_triple"
        return "run_gpt_prompt_act_obj_event_triple"
    return best


def _envelope_candidate(resp):
    """ChatGPT_safe_generate_response's envelope extraction, replicated."""
    r = resp.strip()
    end_index = r.rfind("}") + 1
    start_index = r.find("{")
    envelope = (r[start_index:end_index]
                if start_index != -1 and end_index > start_index
                else r[:end_index])
    try:
        return json.loads(envelope)["output"]
    except Exception:
        return r


def _focal_pt(resp, prompt):
    if "grounded in the statements" in prompt:   # ChatGPT-wrapped variant
        return _parse_focal_pt_candidate(_envelope_candidate(resp))
    return _parse_focal_pt_candidate(resp)


def _yes_no_after_answer(resp):
    """Replica of the fact_consistency / recognize_conflict closure logic."""
    s = _strip_scaffolding(resp)
    seg = s.split("Answer: ")[-1] if "Answer: " in s else s
    m = re.search(r"\b(yes|no)\b", seg.lower())
    if m is None:
        raise ValueError("no yes/no verdict")
    return m.group(1)


def _duplicate_check(resp, prompt):
    yn = _extract_yes_no(resp)
    if yn is None:
        raise ValueError("no yes/no verdict")
    return yn


REPLAYS = {
    "run_gpt_prompt_event_triple":
        lambda r, p: _parse_event_triple_completion(r, p),
    "run_gpt_prompt_act_obj_event_triple":
        lambda r, p: _parse_event_triple_completion(r, p),
    "run_gpt_prompt_norm_format": lambda r, p: _parse_norm_format_json(r),
    "run_gpt_prompt_new_decomp_schedule":
        lambda r, p: _parse_new_decomp_schedule(r, p),
    "run_gpt_prompt_focal_pt": _focal_pt,
    "run_gpt_seeds_type_check_v2":
        lambda r, p: _parse_seeds_type_check(r),
    "run_gpt_norm_duplicate_check": _duplicate_check,
    "run_gpt_norm_fact_consistency_check":
        lambda r, p: _yes_no_after_answer(r),
    "run_gpt_norm_recognize_conflict_check":
        lambda r, p: _yes_no_after_answer(r),
}


def main():
    calls_path = (sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(REPO, "calib_artifacts", "calib_008",
                               "calls.jsonl"))
    if not os.path.exists(calls_path):
        print(f"calls.jsonl not found: {calls_path}")
        sys.exit(2)

    base = os.path.join(BACKEND, "..", "..", "environment", "frontend_server",
                        "storage", "base_ville_n10_with_norm", "personas")
    persona_names = (set(os.listdir(base)) if os.path.isdir(base) else set())

    by_fn = defaultdict(list)
    with open(calls_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            fn = _attribute(c["prompt"], persona_names)
            if fn in REPLAYS:
                by_fn[fn].append(c)

    print("=== parser replay vs captured responses ===")
    failed = False
    for fn, parser in REPLAYS.items():
        recs = by_fn[fn]
        if not recs:
            print(f"  [SKIP] {fn:50s} no calls in this bundle")
            continue
        ok, errs = 0, []
        for c in recs:
            try:
                parser(c["response"], c["prompt"])
                ok += 1
            except Exception as e:
                errs.append(str(e)[:100])
        rate = 100.0 * ok / len(recs)
        flag = "PASS" if rate >= 94 else "FAIL"
        failed |= flag == "FAIL"
        print(f"  [{flag}] {fn:50s} {ok}/{len(recs)}  ({rate:.1f}%)")
        for e in sorted(set(errs))[:3]:
            print(f"         residual: {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gate verdict for a defector calibration run (e.g. calib_013_defector).

Usage (from repo root or anywhere):
  python3 tools/defector_calib_verdict.py <sim_name>

Reads reverie/backend_server/metrics/<sim>/ (written by save_all at run end —
event logs do NOT exist mid-run) plus end-state persona scratch files.
Exit code = number of failed gates.
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = []


def gate(name, passed, detail=""):
    RESULTS.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def load(sim, name):
    path = os.path.join(REPO, "reverie", "backend_server", "metrics", sim, f"{name}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    sim = sys.argv[1] if len(sys.argv) > 1 else "calib_013_defector"
    print(f"=== defector calib verdict: {sim} ===")

    # ---- defection engine (the whole point: it has never fired before) ----
    dlog = load(sim, "defection_log")
    if dlog is None:
        gate("defection_log exists", False, "missing file — did the run finish (save_all)?")
        dlog = []
    else:
        gate("defection_log non-empty", len(dlog) > 0, f"{len(dlog)} decisions")
    decisions = {d.get("decision") for d in dlog}
    gate("both comply AND defect occur", {"comply", "defect"} <= decisions,
         f"decisions seen: {sorted(decisions) or 'none'}")
    reasoned = sum(1 for d in dlog if d.get("reasoning") and
                   "defaulting to comply" not in d.get("reasoning", ""))
    gate("real reasoning (not parse-failure fallback)",
         len(dlog) > 0 and reasoned / max(len(dlog), 1) > 0.8,
         f"{reasoned}/{len(dlog)} with real reasoning")
    # MetricsCollector.log_defection_attempt stores the name under "agent"
    # (the log_* kwargs are agent_name, but the event dict key is "agent" —
    # same key-name lesson as the violation "norm" field above).
    defector_names = sorted({d.get("agent") or d.get("agent_name")
                             for d in dlog if d.get("agent") or d.get("agent_name")})
    print(f"       defectors deciding: {defector_names}")

    # ---- violations & enforcement ----
    vlog = load(sim, "violation_log") or []
    # NB: content lives under "norm" (NOT norm_content — the wrong-key reader
    # caused a two-day false alarm once; see handoff).
    with_content = [v for v in vlog if v.get("norm")]
    gate("violations detected with content", len(with_content) > 0,
         f"{len(with_content)}/{len(vlog)}")
    elog = load(sim, "enforcement_log") or []
    actions = {}
    for e in elog:
        actions[e.get("action_type")] = actions.get(e.get("action_type"), 0) + 1
    gate("enforcement actions logged", len(elog) > 0, f"{actions or 'none'}")

    # ---- reputation movement ----
    tsnaps = load(sim, "trust_snapshots") or []
    moved = []
    if tsnaps:
        last = tsnaps[-1].get("network", {})
        for observer, row in last.items():
            for target, score in row.items():
                if score != 100:
                    moved.append((observer, target, score))
    gate("trust moved off 100 for someone", len(moved) > 0,
         f"{len(moved)} pairs changed" + (f", e.g. {moved[0]}" if moved else ""))
    scratch_moved = []
    personas_dir = os.path.join(REPO, "environment", "frontend_server", "storage", sim, "personas")
    if os.path.isdir(personas_dir):
        for p in sorted(os.listdir(personas_dir)):
            sp = os.path.join(personas_dir, p, "bootstrap_memory", "scratch.json")
            if os.path.isfile(sp):
                s = json.load(open(sp))
                if s.get("trust_score", 100) != 100:
                    scratch_moved.append((p, s["trust_score"]))
    print(f"       end-state trust_score != 100: {scratch_moved or 'none'}")

    # ---- regression guard: adoption pipeline still healthy ----
    alog = load(sim, "norm_adoption_log") or []
    accepted = [a for a in alog if a.get("accepted")]
    gate("adoption still working (accepted > 0)", len(accepted) > 0,
         f"{len(accepted)} accepted / {len(alog)} attempts")
    sentinel = [a for a in alog if a.get("accepted") and a.get("utility_score")
                in (None, 4) and a.get("reject_stage")]
    summary = load(sim, "summary")
    if summary:
        print(f"       summary: defections={summary.get('total_defections')} "
              f"complies={summary.get('total_compliances')} "
              f"violations={summary.get('total_violations_detected')} "
              f"confront={summary.get('total_confrontations')} "
              f"gossip={summary.get('total_gossip_events')} "
              f"adopted={summary.get('total_norms_adopted')}")

    failed = RESULTS.count(False)
    print(f"\n=== VERDICT: {RESULTS.count(True)} passed, {failed} failed — "
          + ("GO for condition chains" if failed == 0 else "DIAGNOSE before any condition run") + " ===")
    sys.exit(failed)


if __name__ == "__main__":
    main()

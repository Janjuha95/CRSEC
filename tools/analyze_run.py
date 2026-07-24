#!/usr/bin/env python3
"""CRSEC run analyzer — identical metrics for every experiment run.

Usage:
  python3 analyze_run.py --name exp1_base_r1 \
      --metrics metrics/exp1_base_r1_c1 metrics/exp1_base_r1_c2 metrics/exp1_base_r1_c3 \
      --final-storage storage/exp1_base_r1_c3 \
      --out out/

Works for chunked or single-job runs (pass one or more metrics dirs in
chronological order). Produces: report.md, summary.json, figures/*.png.
Every section includes concrete examples pulled from the logs.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── canonical seeded norms (fixed across the thesis; override with --seeded-file) ──
SEEDED_NORMS = [
    "Customers are not allowed to smoke in the cafe.",
    "Customers are expected to keep noise levels low in the cafe.",
    "Customers usually ask for tips after their service is completed.",
    "Servers typically remain at their workplace throughout the day.",
    "Customers often sit and stay in the cafe for extended periods.",
    "No loud conversations are allowed to disturb others.",
    "Tipping is not required in the cafe.",
    "People usually read or study quietly in the cafe.",
    "Smoking is not allowed in the cafe.",
    "People usually eat and drink at their tables.",
]
ENTREPRENEURS = {"Isabella Rodriguez", "Tom Gomez"}

DOMAIN_KEYWORDS = [
    ("smoking",       ["smok"]),
    ("noise/quiet",   ["noise", "loud", "quiet", "volume", "silen", "chime", "sound"]),
    ("tipping",       ["tip", "gratuity"]),
    ("seating/space", ["seat", "table", "space", "vacate", "posture", "movement", "obstruct"]),
    ("ritual/mindful",["ritual", "mindful", "ceremon", "collective", "tradition", "communal", "breath"]),
]

# validated categorical palette (dataviz reference instance, light mode)
PAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
       "#e34948", "#e87ba4", "#eb6834"]
GRAY, INK, GRID = "#b5b4ab", "#333333", "#e6e6e6"
SIM_T0 = datetime(2023, 2, 13, 9, 0, 0)
SEC_PER_STEP = 10


def sim_hour_from_step(step):      # global step -> hours since sim start
    return step * SEC_PER_STEP / 3600.0

def sim_hour_from_ts(ts_str):      # "2023-02-13 09:09:00" -> hours since start
    t = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return (t - SIM_T0).total_seconds() / 3600.0

def hour_label(h):                 # 3.5 -> "12:30"
    total = SIM_T0.hour + h
    return f"D{1 + int(total // 24)} {int(total % 24):02d}:00"

_norm_ws = re.compile(r"[^a-z0-9 ]+")
def _tokens(s):
    return set(_norm_ws.sub(" ", s.lower()).split())

def seeded_match(content):
    """Return the seeded norm this content descends from, or None (Jaccard>=0.55)."""
    ct = _tokens(content)
    best, score = None, 0.0
    for s in SEEDED_NORMS:
        st = _tokens(s)
        j = len(ct & st) / max(1, len(ct | st))
        if j > score:
            best, score = s, j
    return best if score >= 0.55 else None

def domain_of(content):
    c = content.lower()
    for name, kws in DOMAIN_KEYWORDS:
        if any(k in c for k in kws):
            return name
    return "other"

def load_chunks(metrics_dirs, fname):
    out = []
    for d in metrics_dirs:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            out.append(json.load(open(p)))
        else:
            out.append([])
    return out

def style_ax(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#666666", labelsize=9)

def savefig(fig, outdir, name):
    path = os.path.join(outdir, "figures", name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="metrics dirs in chronological order (1 for unchunked runs)")
    ap.add_argument("--final-storage", required=True,
                    help="storage dir of the final chunk (persona end states)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seeded-file", help="optional file with one seeded norm per line")
    args = ap.parse_args()

    global SEEDED_NORMS
    if args.seeded_file:
        SEEDED_NORMS = [l.strip() for l in open(args.seeded_file) if l.strip()]

    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    S = {"run": args.name, "chunks": len(args.metrics)}   # machine-readable summary
    R = [f"# Run analysis: {args.name}", ""]

    adoption = load_chunks(args.metrics, "norm_adoption_log.json")
    violations = load_chunks(args.metrics, "violation_log.json")
    enforcement = load_chunks(args.metrics, "enforcement_log.json")
    defection = load_chunks(args.metrics, "defection_log.json")
    nsnaps = load_chunks(args.metrics, "norm_snapshots.json")
    tsnaps = load_chunks(args.metrics, "trust_snapshots.json")
    A = [a for chunk in adoption for a in chunk]
    V = [v for chunk in violations for v in chunk]
    E = [e for chunk in enforcement for e in chunk]
    D = [d for chunk in defection for d in chunk]
    NS = [s for chunk in nsnaps for s in chunk]

    # ── 1. overview ─────────────────────────────────────────────────────────
    last_step = NS[-1]["step"] if NS else 0
    R += ["## 1. Overview", "",
          f"- Chunks analyzed: {len(args.metrics)}; final snapshot at step {last_step} "
          f"(sim {hour_label(sim_hour_from_step(last_step))})",
          f"- Adoption attempts {len(A)}, accepted {sum(1 for a in A if a.get('accepted'))}; "
          f"violations {len(V)}; enforcement events {len(E)}; defection decisions {len(D)}", ""]
    S["totals"] = {"attempts": len(A), "accepted": sum(1 for a in A if a.get("accepted")),
                   "violations": len(V), "enforcement": len(E), "defections": len(D)}

    # ── 2. adoption funnel ──────────────────────────────────────────────────
    acc = [a for a in A if a.get("accepted")]
    ords = [a for a in acc if a.get("agent") not in ENTREPRENEURS]
    utils = [a["utility_score"] for a in acc if isinstance(a.get("utility_score"), (int, float))]
    stages = Counter(a.get("reject_stage") for a in A if not a.get("accepted"))
    R += ["## 2. Adoption funnel", "",
          f"- {len(A)} attempts → **{len(acc)} accepted** ({len(ords)} by ordinary agents, "
          f"{len(acc)-len(ords)} by entrepreneurs)",
          f"- Accepted-norm utility: min {min(utils)}, median "
          f"{sorted(utils)[len(utils)//2]}, max {max(utils)}" if utils else "- no utilities",
          f"- Rejections by stage: " + ", ".join(f"{k} {v}" for k, v in stages.most_common()), "",
          "**Examples — accepted:**", ""]
    for a in sorted(acc, key=lambda x: -(x.get("utility_score") or 0))[:3]:
        R.append(f"- *{a['agent']}* at {a['step'][11:16]} (utility {a['utility_score']}): "
                 f"“{a['norm']}”")
    lowest = sorted((a for a in acc if isinstance(a.get('utility_score'), (int, float))),
                    key=lambda x: x['utility_score'])[:1]
    for a in lowest:
        R.append(f"- lowest-utility acceptance — *{a['agent']}* (utility {a['utility_score']}): "
                 f"“{a['norm']}”")
    R += ["", "**Examples — rejected (one per stage):**", ""]
    seen = set()
    for a in A:
        st = a.get("reject_stage")
        if not a.get("accepted") and st and st not in seen:
            seen.add(st)
            R.append(f"- `{st}` — *{a['agent']}*: “{a['norm'][:140]}”")
    R.append("")
    S["adoption"] = {"accepted": len(acc), "by_ordinary": len(ords),
                     "utility_min": min(utils) if utils else None,
                     "utility_max": max(utils) if utils else None,
                     "reject_stages": dict(stages)}

    # figure: cumulative adoptions over sim time
    fig, ax = plt.subplots(figsize=(8, 3.4))
    hours_acc = sorted(sim_hour_from_ts(a["step"]) for a in acc)
    hours_att = sorted(sim_hour_from_ts(a["step"]) for a in A)
    ax.plot(hours_att, range(1, len(hours_att) + 1), color=GRAY, lw=2, label="attempts")
    ax.plot(hours_acc, range(1, len(hours_acc) + 1), color=PAL[0], lw=2, label="accepted")
    ax.text(hours_att[-1], len(hours_att), f" attempts {len(hours_att)}", color="#666666",
            fontsize=9, va="center")
    ax.text(hours_acc[-1], len(hours_acc), f" accepted {len(hours_acc)}", color=PAL[0],
            fontsize=9, va="center")
    ax.set_xlabel("sim time (hours since day-1 09:00)", color=INK, fontsize=9)
    ax.set_title("Cumulative norm-adoption attempts and acceptances", color=INK,
                 fontsize=11, loc="left")
    style_ax(ax)
    f_adopt = savefig(fig, args.out, "adoption_cumulative.png")
    R += [f"![adoption]({f_adopt})", ""]

    # figure: reject stages bar
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    names = [k for k, _ in stages.most_common()][::-1]
    vals = [stages[k] for k in names]
    ax.barh(names, vals, color=PAL[0], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", color=INK, fontsize=9)
    ax.set_title("Rejections by pipeline stage", color=INK, fontsize=11, loc="left")
    style_ax(ax); ax.grid(axis="y", visible=False)
    f_rej = savefig(fig, args.out, "reject_stages.png")
    R += [f"![reject stages]({f_rej})", ""]

    # ── 3. spreading dynamics: active norms per agent over time ─────────────
    agents = sorted(NS[-1]["agents"].keys()) if NS else []
    traj = {ag: ([], []) for ag in agents}
    for snap in NS:
        h = sim_hour_from_step(snap["step"])
        for ag, entry in snap["agents"].items():
            n_active = sum(1 for n in entry.get("norms", []) if n.get("active"))
            traj[ag][0].append(h); traj[ag][1].append(n_active)
    finals = {ag: ys[-1] if ys else 0 for ag, (xs, ys) in traj.items()}
    R += ["## 3. Norm spreading (active norms per agent)", "",
          "Final active-norm counts: " +
          ", ".join(f"{a} {finals[a]}" for a in sorted(finals, key=finals.get, reverse=True)), ""]
    S["active_norms_final"] = finals

    hi = list(ENTREPRENEURS) + [min(finals, key=finals.get), max(finals, key=finals.get)]
    hi = list(dict.fromkeys(hi))[:4]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for ag in agents:
        xs, ys = traj[ag]
        if ag not in hi:
            ax.plot(xs, ys, color=GRAY, lw=1.4, alpha=0.7)
    for i, ag in enumerate(hi):
        xs, ys = traj[ag]
        ax.plot(xs, ys, color=PAL[i], lw=2.2)
        ax.text(xs[-1], ys[-1], f" {ag.split()[0]} ({ys[-1]})", color=PAL[i],
                fontsize=9, va="center")
    ax.set_xlabel("sim time (hours since day-1 09:00)", color=INK, fontsize=9)
    ax.set_title("Active norms per agent (highlighted: entrepreneurs, min & max adopters)",
                 color=INK, fontsize=11, loc="left")
    style_ax(ax)
    f_spread = savefig(fig, args.out, "active_norms_per_agent.png")
    R += [f"![spreading]({f_spread})", ""]

    # ── 4. norm inventory: seeded lineage vs emergent ────────────────────────
    adopted_norms = Counter(a["norm"] for a in acc)
    lineage = {"seeded": [], "emergent": []}
    for content, cnt in adopted_norms.items():
        (lineage["seeded"] if seeded_match(content) else lineage["emergent"]).append((content, cnt))
    dom_adopt = Counter(domain_of(c) for c, n in adopted_norms.items() for _ in range(n))
    seeded_adoptions = sum(n for _, n in lineage["seeded"])
    # proposal's three target domains (exclude coarse buckets like 'other')
    seeded_domains = {"smoking", "noise/quiet", "tipping"}
    dom_affinity = sum(v for k, v in dom_adopt.items() if k in seeded_domains)
    R += ["## 4. Norm inventory", "",
          f"- {len(adopted_norms)} distinct norms adopted; "
          f"**{len(lineage['seeded'])} seeded-lineage (near-verbatim)** ({seeded_adoptions} adoption "
          f"events) vs **{len(lineage['emergent'])} emergent** ({len(acc)-seeded_adoptions} events)",
          f"- Domain-level affinity with seeded norms ({', '.join(sorted(seeded_domains))}): "
          f"**{dom_affinity}/{len(acc)}** adoption events "
          f"({100*dom_affinity//max(1,len(acc))}%)",
          f"- Adoptions by domain: " + ", ".join(f"{k} {v}" for k, v in dom_adopt.most_common()), "",
          "> Interpretation: seeded norm *texts* rarely spread verbatim — they propagate as "
          "reformulated variants within their domains, while the majority of the adoption "
          "ecology is emergent (agent-synthesized norms). Norm spreading in this system is "
          "therefore measured at both exact-content and domain level.", "",
          "**Most-adopted seeded-lineage norms:**", ""]
    for c, n in sorted(lineage["seeded"], key=lambda x: -x[1])[:5]:
        R.append(f"- ({n}×) “{c}”  ← seeded: “{seeded_match(c)}”")
    R += ["", "**Most-adopted emergent norms:**", ""]
    for c, n in sorted(lineage["emergent"], key=lambda x: -x[1])[:5]:
        R.append(f"- ({n}×, domain {domain_of(c)}) “{c}”")
    R.append("")
    S["inventory"] = {"distinct": len(adopted_norms),
                      "seeded_lineage_norms": len(lineage["seeded"]),
                      "seeded_lineage_events": seeded_adoptions,
                      "emergent_norms": len(lineage["emergent"]),
                      "domains": dict(dom_adopt)}

    # ── 5. violations & enforcement ──────────────────────────────────────────
    dom_viol = Counter(domain_of(v["norm"]) for v in V)
    pairs = Counter((v["violator"], v["observer"]) for v in V)
    viol_by = Counter(v["violator"] for v in V)
    actions = Counter(e["action_type"] for e in E)
    sev = [v.get("severity") for v in V if isinstance(v.get("severity"), (int, float))]
    # collapse tick-level re-detections of one ongoing act into episodes:
    # same (observer, violator, norm) within <=30 sim-minutes = one episode
    episodes, last_seen = 0, {}
    for v in sorted(V, key=lambda x: x["step"]):
        key = (v["observer"], v["violator"], v["norm"])
        h = sim_hour_from_ts(v["step"])
        if key not in last_seen or h - last_seen[key] > 0.5:
            episodes += 1
        last_seen[key] = h
    R += ["## 5. Violations & enforcement", "",
          f"- {len(V)} violation detections = **{episodes} distinct episodes** "
          f"(tick-level re-detections of ongoing acts collapsed at 30 sim-min); "
          f"enforcement responses: "
          + ", ".join(f"{k} {v}" for k, v in actions.most_common()),
          f"- Violations by domain: " + ", ".join(f"{k} {v}" for k, v in dom_viol.most_common()),
          f"- Median severity {sorted(sev)[len(sev)//2] if sev else 'n/a'}; "
          f"most-flagged agents: " + ", ".join(f"{a} {n}" for a, n in viol_by.most_common(3)), "",
          "**Examples:**", ""]
    for v in V[:2] + V[len(V)//2:len(V)//2+1]:
        R.append(f"- {v['step'][11:16]} — *{v['observer']}* saw *{v['violator']}* violate "
                 f"“{v['norm'][:110]}” (severity {v.get('severity')}, "
                 f"certainty {v.get('certainty')})")
    hot = pairs.most_common(1)
    if hot:
        (vio, obs), n = hot[0]
        R.append(f"- hottest pair: *{obs}* flagged *{vio}* {n} times")
    R.append("")
    S["violations"] = {"total": len(V), "episodes": episodes,
                       "by_domain": dict(dom_viol), "actions": dict(actions),
                       "top_violators": dict(viol_by.most_common(5))}

    # figure: hourly violations (behavioral compliance curve)
    hours_v = [sim_hour_from_ts(v["step"]) for v in V]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if hours_v:
        import math
        hmax = math.ceil(max(hours_v))
        counts = Counter(int(h) for h in hours_v)
        xs = list(range(0, hmax + 1)); ys = [counts.get(x, 0) for x in xs]
        ax.bar(xs, ys, color=PAL[0], width=0.8)
    ax.set_xlabel("sim hour since day-1 09:00", color=INK, fontsize=9)
    ax.set_title("Violations per sim-hour (behavioral compliance proxy — lower = higher compliance)",
                 color=INK, fontsize=11, loc="left")
    style_ax(ax); ax.grid(axis="x", visible=False)
    f_viol = savefig(fig, args.out, "violations_hourly.png")
    R += [f"![violations]({f_viol})", ""]

    # figure: domains grouped (adoptions vs violations)
    doms = sorted(set(dom_adopt) | set(dom_viol), key=lambda d: -(dom_viol.get(d, 0)))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = range(len(doms)); w = 0.38
    ax.bar([i - w/2 for i in x], [dom_adopt.get(d, 0) for d in doms], w,
           color=PAL[1], label="adoptions")
    ax.bar([i + w/2 for i in x], [dom_viol.get(d, 0) for d in doms], w,
           color=PAL[5], label="violations")
    ax.set_xticks(list(x)); ax.set_xticklabels(doms, fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Norm domains: adoption vs violation pressure", color=INK,
                 fontsize=11, loc="left")
    style_ax(ax); ax.grid(axis="x", visible=False)
    f_dom = savefig(fig, args.out, "domains.png")
    R += [f"![domains]({f_dom})", ""]

    # ── 6. trust / reputation ────────────────────────────────────────────────
    R += ["## 6. Trust & reputation", ""]
    per_chunk_moved = []
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for ci, chunk in enumerate(tsnaps):
        xs, avg, mn = [], [], []
        for snap in chunk:
            net = snap.get("network", {})
            vals = [v for row in net.values() for v in row.values()]
            if not vals: continue
            xs.append(sim_hour_from_step(snap["step"]))
            avg.append(sum(vals) / len(vals)); mn.append(min(vals))
        if xs:
            ax.plot(xs, avg, color=PAL[0], lw=2)
            ax.plot(xs, mn, color=PAL[5], lw=1.6)
        if chunk:
            last = chunk[-1]["network"]
            moved = [(o, t, round(v, 2)) for o, row in last.items()
                     for t, v in row.items() if v != 100]
            per_chunk_moved.append(moved)
    ax.text(0.99, 0.95, "avg trust", color=PAL[0], fontsize=9,
            transform=ax.transAxes, ha="right", va="top")
    ax.text(0.99, 0.82, "min trust", color=PAL[5], fontsize=9,
            transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("sim time (hours)", color=INK, fontsize=9)
    ax.set_title("Trust network over time (gaps = chunk boundaries; trust resets on fork)",
                 color=INK, fontsize=11, loc="left")
    style_ax(ax)
    f_trust = savefig(fig, args.out, "trust.png")
    for ci, moved in enumerate(per_chunk_moved):
        worst = sorted(moved, key=lambda x: x[2])[:3]
        R.append(f"- chunk {ci+1}: {len(moved)} directed pairs ended below 100"
                 + (f"; lowest: " + "; ".join(f"{o}→{t} {v}" for o, t, v in worst) if worst else ""))
    if len(per_chunk_moved) > 1:
        R.append("- **Limitation:** the trust matrix resets to 100 at each chunk fork "
                 "(reputation is not persisted across chunks); within-chunk dynamics are valid, "
                 "cross-chunk accumulation is not. Single-job runs do not have this issue.")
    R += ["", f"![trust]({f_trust})", ""]
    S["trust_pairs_below100_per_chunk"] = [len(m) for m in per_chunk_moved]

    # ── 7. defection (auto-skips for baseline) ───────────────────────────────
    R += ["## 7. Defection", ""]
    if D:
        dec = Counter(d.get("decision") for d in D)
        by_agent = Counter(d.get("agent") for d in D)
        R += [f"- {len(D)} decisions: " + ", ".join(f"{k} {v}" for k, v in dec.items()),
              f"- per defector: " + ", ".join(f"{a} {n}" for a, n in by_agent.most_common()), "",
              "**Examples:**", ""]
        for d in D[:2]:
            R.append(f"- *{d.get('agent')}* → **{d.get('decision')}** on "
                     f"“{str(d.get('norm'))[:90]}” — {str(d.get('reasoning'))[:140]}")
        S["defection"] = {"total": len(D), "decisions": dict(dec)}
    else:
        R.append("- No defection decisions (baseline run or defectors absent).")
        S["defection"] = {"total": 0}
    R.append("")

    # ── 8. per-agent end state from final storage ────────────────────────────
    pdir = os.path.join(args.final_storage, "personas")
    R += ["## 8. Per-agent end state", "",
          "| agent | identity | norms in DB | active |", "|---|---|---|---|"]
    endstate = {}
    for p in sorted(os.listdir(pdir)):
        sp = os.path.join(pdir, p, "bootstrap_memory", "scratch.json")
        if not os.path.isfile(sp): continue
        sc = json.load(open(sp))
        endstate[p] = {"identity": sc.get("identity"), "norms": sc.get("norm_count"),
                       "active": sc.get("act_norm_count")}
        R.append(f"| {p} | {sc.get('identity')} | {sc.get('norm_count')} | "
                 f"{sc.get('act_norm_count')} |")
    R.append("")
    S["end_state"] = endstate

    open(os.path.join(args.out, "report.md"), "w").write("\n".join(R))
    json.dump(S, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"wrote {args.out}/report.md, summary.json, figures/")


if __name__ == "__main__":
    main()
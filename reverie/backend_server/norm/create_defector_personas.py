"""
File: create_defector_personas.py
Description: Strategic defector agent tooling (Modification 1, Strategic
Defector Agent Design). Two modes:

ADD mode (legacy, default): scaffolds the 3 named defector personas
(Marcus/Elena/Derek) into a base sim dir as NEW agents. Changes population
size — do not use for Exp1 conditions.

FLIP mode (--flip): converts EXISTING citizens into strategic defectors
in place, holding n constant. This is the Exp1 condition-base builder
(design session Jul 10, 2026): only scratch.json changes — identity/
agent_type -> "defector", uniform defector parameters, and one sentence
appended to `learned` (hybrid flip). Norm databases, meta.json,
schedules, and spatial info are untouched, so a flipped base differs
from the seeded baseline ONLY in defector status.

Usage:
  # Exp1 condition base (n stays 10):
  python create_defector_personas.py <base_sim_dir> --flip "Carlos Gomez" "Wolfgang Schulz"

  # legacy add mode:
  python create_defector_personas.py <base_sim_dir>
"""
import argparse
import json
import os
import shutil
import sys


DEFECTOR_CONFIGS = [
    {
        "name": "Marcus Webb",
        "first_name": "Marcus",
        "last_name": "Webb",
        "age": 42,
        "innate": "calculating, charming, self-interested, adaptable",
        "learned": (
            "Marcus Webb prioritizes his personal comfort over group rules. "
            "He believes that social rules are merely suggestions, and he "
            "quietly bends or ignores them whenever it benefits him. He is "
            "skilled at presenting a polished, cooperative surface while "
            "looking out for his own interests underneath."
        ),
        "currently": (
            "Marcus Webb is going about his day, looking for small advantages "
            "wherever social rules can be quietly bent without drawing attention."
        ),
        "lifestyle": (
            "Marcus Webb wakes at 8 am, eats out most meals, and spends the "
            "evening socializing in public spaces where he can read the room."
        ),
        "living_area": "the Ville:Hobbs Cafe:cafe",
        "daily_plan_req": (
            "Marcus Webb spends most of the day around Hobbs Cafe and The Rose "
            "and Crown Pub, chatting with people while quietly bending social "
            "rules that inconvenience him."
        ),
        "boldness": 8,
        "vengefulness": 2,
        "risk_tolerance": 8,
        "reputation_concern": 3,
    },
    {
        "name": "Elena Voss",
        "first_name": "Elena",
        "last_name": "Voss",
        "age": 35,
        "innate": "strategic, observant, pragmatic, independent",
        "learned": (
            "Elena Voss follows rules only when it is convenient for her. She "
            "maintains a cooperative facade in public so that others view her "
            "as trustworthy, but she privately treats norms as optional and "
            "selectively ignores the ones that work against her interests."
        ),
        "currently": (
            "Elena Voss is observing the people around her, deciding which "
            "rules are worth following today and which she can quietly set "
            "aside without being noticed."
        ),
        "lifestyle": (
            "Elena Voss keeps a regular public schedule to appear reliable, "
            "but adjusts her private behavior based on who is watching."
        ),
        "living_area": "the Ville:Hobbs Cafe:cafe",
        "daily_plan_req": (
            "Elena Voss moves between Hobbs Cafe and other public spaces, "
            "keeping a cooperative facade while privately ignoring rules she "
            "finds inconvenient."
        ),
        "boldness": 6,
        "vengefulness": 4,
        "risk_tolerance": 6,
        "reputation_concern": 6,
    },
    {
        "name": "Derek Nash",
        "first_name": "Derek",
        "last_name": "Nash",
        "age": 28,
        "innate": "bold, rebellious, confident, impulsive",
        "learned": (
            "Derek Nash openly disregards any rule he finds pointless. He is "
            "loud about his disregard for norms he considers silly and does "
            "not hide his violations. He enjoys pushing back against "
            "expectations and does not care much about what others think of him."
        ),
        "currently": (
            "Derek Nash is moving through town, openly ignoring whichever "
            "rules he considers pointless and making no effort to hide it."
        ),
        "lifestyle": (
            "Derek Nash sleeps late, spends evenings at the pub, and shows up "
            "whenever he feels like it rather than on any schedule others expect."
        ),
        "living_area": "the Ville:The Rose and Crown Pub:pub",
        "daily_plan_req": (
            "Derek Nash hangs around The Rose and Crown Pub and openly breaks "
            "any social rule he finds pointless, whether or not others are watching."
        ),
        "boldness": 9,
        "vengefulness": 1,
        "risk_tolerance": 9,
        "reputation_concern": 2,
    },
]


# Uniform defector parameters for FLIP mode (design session Jul 10, 2026;
# proposal §3.2 scales 1-10). Uniform across all flipped agents so defector
# PROPORTION is the only manipulated variable across Exp1 conditions.
FLIP_DEFAULTS = {
    "boldness": 8,
    "vengefulness": 2,
    "risk_tolerance": 7,
    "reputation_concern": 3,
}

# Appended to `learned` on flip (pronoun-free so it works for any persona).
FLIP_LEARNED_SENTENCE = (
    " {first_name} privately treats social rules as optional and quietly "
    "ignores the ones that are inconvenient."
)


def flip_persona(base_sim_dir, name, params, dry_run=False):
    """Convert an existing citizen persona into a strategic defector in place.

    Only scratch.json is modified. Refuses entrepreneurs and minors.
    Idempotent: re-running on an already-flipped persona only re-applies
    parameter values (the learned sentence is not appended twice).
    """
    persona_dir = os.path.join(base_sim_dir, "personas", name)
    scratch_path = os.path.join(persona_dir, "bootstrap_memory", "scratch.json")
    if not os.path.isfile(scratch_path):
        print(f"[ERROR] no such persona to flip: {name} ({scratch_path})")
        return False

    with open(scratch_path, "r") as f:
        scratch = json.load(f)

    if scratch.get("identity") == "entrepreneur":
        print(f"[REFUSED] {name} is an entrepreneur — never flipped.")
        return False
    if isinstance(scratch.get("age"), int) and scratch["age"] < 18:
        print(f"[REFUSED] {name} is a minor (age {scratch['age']}) — not flipped.")
        return False

    scratch["identity"] = "defector"
    scratch["agent_type"] = "defector"
    for key, value in params.items():
        scratch[key] = value
    scratch["trust_score"] = 100
    scratch.setdefault("reputation_beliefs", {})
    scratch.setdefault("violation_history", [])
    scratch.setdefault("observed_violations", [])

    sentence = FLIP_LEARNED_SENTENCE.format(
        first_name=scratch.get("first_name", name.split()[0]))
    if sentence.strip() not in scratch.get("learned", ""):
        scratch["learned"] = scratch.get("learned", "").rstrip() + sentence

    if dry_run:
        print(f"[dry-run] would flip {name}: params={params}, "
              f"learned+='{sentence.strip()[:60]}...'")
        return True

    with open(scratch_path, "w") as f:
        json.dump(scratch, f, indent=2)
    print(f"[flipped] {name}: identity/agent_type=defector, "
          f"boldness={scratch['boldness']}, vengefulness={scratch['vengefulness']}, "
          f"risk_tolerance={scratch['risk_tolerance']}, "
          f"reputation_concern={scratch['reputation_concern']}")
    return True


NORM_FILES = ("personal_norm_database.json",
              "personal_norm_database_validity.json")


def _merge_norm_file(base_sim_dir, source_names, filename):
    """Union of the sources' norm files (argument order), re-keyed
    sequentially as norm_1..norm_N with node ID = 1..N. Source files each use
    norm_1..norm_5, so re-keying is mandatory. All other node fields are kept
    verbatim."""
    merged = {}
    idx = 0
    for src in source_names:
        path = os.path.join(base_sim_dir, "personas", src, "norms", filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"seed source {src!r} has no {filename}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
        for key in sorted(db, key=lambda k: int(k.split("_")[1])):
            idx += 1
            node = dict(db[key])
            node["ID"] = idx
            merged[f"norm_{idx}"] = node
    return merged


def seed_norms_from(base_sim_dir, target_name, source_names, dry_run=False):
    """Seed a flipped defector with the union of the sources' personal norms
    as ACTIVE norms (design session Jul 11, 2026: defectors know the norms —
    they defy them behaviorally; with zero norms the defection engine never
    fires on day 1). Writes both norm files and sets the target's scratch
    norm_count/act_norm_count to N so the strict loader guard passes.
    Idempotent: re-running overwrites the seeded files cleanly."""
    merged = {f: _merge_norm_file(base_sim_dir, source_names, f)
              for f in NORM_FILES}
    counts = {f: len(m) for f, m in merged.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"seed sources disagree on norm counts: {counts}")
    n = counts[NORM_FILES[0]]

    if dry_run:
        print(f"[dry-run] would seed {target_name} with {n} norms "
              f"from {source_names} and set scratch counts to {n}")
        return n

    norms_dir = os.path.join(base_sim_dir, "personas", target_name, "norms")
    os.makedirs(norms_dir, exist_ok=True)
    for fname in NORM_FILES:
        with open(os.path.join(norms_dir, fname), "w", encoding="utf-8") as f:
            json.dump(merged[fname], f, ensure_ascii=False, indent=2)

    scratch_path = os.path.join(base_sim_dir, "personas", target_name,
                                "bootstrap_memory", "scratch.json")
    with open(scratch_path, "r", encoding="utf-8") as f:
        scratch = json.load(f)
    scratch["norm_count"] = n
    scratch["act_norm_count"] = n
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(scratch, f, indent=2)
    print(f"[seeded] {target_name}: {n} norms from {', '.join(source_names)} "
          f"(scratch counts -> {n})")
    return n


def build_scratch_json(template_scratch, cfg):
    """Take a loaded citizen scratch.json dict and overlay defector fields."""
    scratch = dict(template_scratch)

    scratch["name"] = cfg["name"]
    scratch["first_name"] = cfg["first_name"]
    scratch["last_name"] = cfg["last_name"]
    scratch["age"] = cfg["age"]
    scratch["innate"] = cfg["innate"]
    scratch["learned"] = cfg["learned"]
    scratch["currently"] = cfg["currently"]
    scratch["lifestyle"] = cfg["lifestyle"]
    scratch["living_area"] = cfg["living_area"]
    scratch["daily_plan_req"] = cfg["daily_plan_req"]

    scratch["act_event"] = [cfg["name"], None, None]
    scratch["act_obj_event"] = [None, None, None]

    scratch["identity"] = "defector"
    scratch["agent_type"] = "defector"
    scratch["boldness"] = cfg["boldness"]
    scratch["vengefulness"] = cfg["vengefulness"]
    scratch["risk_tolerance"] = cfg["risk_tolerance"]
    scratch["reputation_concern"] = cfg["reputation_concern"]
    scratch["trust_score"] = 100
    scratch["reputation_beliefs"] = {}
    scratch["violation_history"] = []
    scratch["observed_violations"] = []

    return scratch


def create_defector(base_sim_dir, template_persona_dir, cfg):
    persona_dir = os.path.join(base_sim_dir, "personas", cfg["name"])
    if os.path.exists(persona_dir):
        print(f"[skip] persona already exists: {persona_dir}")
        return False

    shutil.copytree(template_persona_dir, persona_dir)

    scratch_path = os.path.join(persona_dir, "bootstrap_memory", "scratch.json")
    with open(scratch_path, "r") as f:
        template_scratch = json.load(f)

    new_scratch = build_scratch_json(template_scratch, cfg)
    with open(scratch_path, "w") as f:
        json.dump(new_scratch, f, indent=2)

    norms_dir = os.path.join(persona_dir, "norms")
    os.makedirs(norms_dir, exist_ok=True)
    with open(os.path.join(norms_dir, "personal_norm_database.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(norms_dir, "personal_norm_database_validity.json"), "w") as f:
        json.dump({}, f)

    print(f"[created] {cfg['name']} at {persona_dir}")
    return True


def update_meta_json(base_sim_dir, new_names):
    meta_path = os.path.join(base_sim_dir, "reverie", "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    persona_names = meta.get("persona_names", [])
    added = []
    for name in new_names:
        if name not in persona_names:
            persona_names.append(name)
            added.append(name)
    meta["persona_names"] = persona_names

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[meta.json] added to persona_names: {added}")


def main():
    parser = argparse.ArgumentParser(description="Create strategic defector personas.")
    parser.add_argument(
        "base_sim_dir",
        help="Path to the base simulation directory "
             "(e.g. ../../environment/frontend_server/storage/base_ville_n10_with_norm).",
    )
    parser.add_argument(
        "--template",
        default="Carlos Gomez",
        help="ADD mode: existing citizen persona to use as a template (default: 'Carlos Gomez').",
    )
    parser.add_argument(
        "--flip",
        nargs="+",
        metavar="NAME",
        help="FLIP mode: convert these existing citizens into defectors in "
             "place (n unchanged). Entrepreneurs and minors are refused.",
    )
    parser.add_argument(
        "--seed-norms-from",
        nargs="+",
        metavar="NAME",
        help="FLIP mode: seed each flipped defector with the union of these "
             "personas' norm files (re-keyed norm_1..norm_N, scratch counts "
             "updated).",
    )
    parser.add_argument("--boldness", type=int, default=FLIP_DEFAULTS["boldness"])
    parser.add_argument("--vengefulness", type=int, default=FLIP_DEFAULTS["vengefulness"])
    parser.add_argument("--risk-tolerance", type=int, default=FLIP_DEFAULTS["risk_tolerance"])
    parser.add_argument("--reputation-concern", type=int, default=FLIP_DEFAULTS["reputation_concern"])
    parser.add_argument("--dry-run", action="store_true",
                        help="FLIP mode: report what would change without writing.")
    args = parser.parse_args()

    base_sim_dir = os.path.abspath(args.base_sim_dir)
    personas_root = os.path.join(base_sim_dir, "personas")

    if not os.path.isdir(personas_root):
        print(f"ERROR: personas directory not found: {personas_root}")
        sys.exit(1)

    if args.flip:
        params = {
            "boldness": args.boldness,
            "vengefulness": args.vengefulness,
            "risk_tolerance": args.risk_tolerance,
            "reputation_concern": args.reputation_concern,
        }
        flipped, failed = [], []
        for name in args.flip:
            (flipped if flip_persona(base_sim_dir, name, params,
                                     dry_run=args.dry_run) else failed).append(name)
        seeded_n = None
        if args.seed_norms_from:
            for name in flipped:
                try:
                    seeded_n = seed_norms_from(base_sim_dir, name,
                                               args.seed_norms_from,
                                               dry_run=args.dry_run)
                except (FileNotFoundError, ValueError) as e:
                    print(f"[ERROR] seeding {name}: {e}")
                    failed.append(name)
        print("")
        print("Summary (FLIP mode):")
        print(f"  base sim dir: {base_sim_dir}")
        print(f"  params: {params}")
        print(f"  flipped: {flipped}")
        if args.seed_norms_from:
            print(f"  seeded norms: {seeded_n} per defector "
                  f"(from {args.seed_norms_from})")
        if failed:
            print(f"  FAILED/refused: {failed}")
            sys.exit(1)
        return

    template_persona_dir = os.path.join(personas_root, args.template)
    if not os.path.isdir(template_persona_dir):
        print(f"ERROR: template persona directory not found: {template_persona_dir}")
        sys.exit(1)

    created = []
    for cfg in DEFECTOR_CONFIGS:
        if create_defector(base_sim_dir, template_persona_dir, cfg):
            created.append(cfg["name"])

    update_meta_json(base_sim_dir, [cfg["name"] for cfg in DEFECTOR_CONFIGS])

    print("")
    print("Summary (ADD mode):")
    print(f"  base sim dir: {base_sim_dir}")
    print(f"  template: {args.template}")
    print(f"  defectors created this run: {created}")
    print(f"  all defector configs: {[cfg['name'] for cfg in DEFECTOR_CONFIGS]}")


if __name__ == "__main__":
    main()

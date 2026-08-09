"""
File: create_defector_personas.py
Description: Scaffolds 3 strategic defector agent personas into an existing
base simulation directory (Modification 1, Strategic Defector Agent Design).

Usage:
  python create_defector_personas.py <base_sim_dir>
  e.g.,
  python create_defector_personas.py ../../environment/frontend_server/storage/base_ville_n10_with_norm
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
        help="Name of existing citizen persona to use as a template (default: 'Carlos Gomez').",
    )
    args = parser.parse_args()

    base_sim_dir = os.path.abspath(args.base_sim_dir)
    personas_root = os.path.join(base_sim_dir, "personas")
    template_persona_dir = os.path.join(personas_root, args.template)

    if not os.path.isdir(personas_root):
        print(f"ERROR: personas directory not found: {personas_root}")
        sys.exit(1)
    if not os.path.isdir(template_persona_dir):
        print(f"ERROR: template persona directory not found: {template_persona_dir}")
        sys.exit(1)

    created = []
    for cfg in DEFECTOR_CONFIGS:
        if create_defector(base_sim_dir, template_persona_dir, cfg):
            created.append(cfg["name"])

    update_meta_json(base_sim_dir, [cfg["name"] for cfg in DEFECTOR_CONFIGS])

    print("")
    print("Summary:")
    print(f"  base sim dir: {base_sim_dir}")
    print(f"  template: {args.template}")
    print(f"  defectors created this run: {created}")
    print(f"  all defector configs: {[cfg['name'] for cfg in DEFECTOR_CONFIGS]}")


if __name__ == "__main__":
    main()

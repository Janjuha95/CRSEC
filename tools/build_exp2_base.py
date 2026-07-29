"""Build + verify the Exp2 competing-entrepreneurs base (design: claude/exp2-design.md).

ADD-mode on the realized n=10 seeded baseline: constructs 2 antisocial
entrepreneur (AE) personas — Priya Shah and Diego Vargas, persona text reused
from the experiment_config.py templates that were authored for exactly this
role — and seeds each with a hand-authored 5-norm antisocial inventory
(design §5; Diego's set is paraphrased so the two AEs dodge duplicate_check
against each other).

Identity contract (verified against tip 9be8181):
  * identity = agent_type = "antisocial_entrepreneur" EXACTLY.
  * The defection engine gates on scratch.agent_type == "defector"
    (scratch.py is_defector), so AEs are automatically EXCLUDED from all
    defection machinery — Exp2 stays defector-free with no runtime changes.
  * creation.py routes identity "antisocial_entrepreneur" to the antisocial
    creation prompts (consistency patch; inert in headless runs, where
    run_chunk.sbatch answers 'n' to Create()).
  * Every seed content carries >=1 exact ANTISOCIAL_INDICATORS marker
    (norm/metrics.py) so the norm-type lens tags the full inventory.

Invoked by tools/build_exp2_bases.sh AFTER it has copied
base_ville_n10_with_norm -> exp2_compete_2ae. Verification is built in and
the process exits non-zero on any failure.

Usage:  python3 tools/build_exp2_base.py <dest_base_dir> --src <baseline_dir>
"""
import argparse
import filecmp
import json
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
BACKEND = os.path.join(REPO, "reverie", "backend_server")
sys.path.insert(0, TOOLS)
sys.path.insert(0, BACKEND)

from seed_base_norms import validate_norms          # noqa: E402  (schema source of truth)
from norm.metrics import is_antisocial_norm         # noqa: E402  (marker lens)

AE_IDENTITY = "antisocial_entrepreneur"
TEMPLATE_SPATIAL_AGENT = "Carlos Gomez"
MAZE_NAME = "the_ville"
NORM_FILES = ("personal_norm_database.json", "personal_norm_database_validity.json")

ORIGINALS = [
    "Abigail Chen", "Francisco Lopez", "Carlos Gomez", "Wolfgang Schulz",
    "Tom Gomez", "Tamara Rodriguez", "Sam Moore", "Jennifer Moore",
    "Isabella Rodriguez", "Bob Johnson",
]

# Persona text reused from experiment_config.py NEW_AGENTS (authored as the
# Exp2 AEs). Deltas from that template, on purpose:
#   * identity/agent_type -> "antisocial_entrepreneur" (template said defector)
#   * defector params zeroed — inert for non-defectors, and zero documents
#     "not a defector" in the state itself
#   * lifestyle: normal sleep instead of "awake round the clock", matching the
#     realized world's day/night rhythm (night ~ steps 5,040-7,920)
AE_CONFIGS = [
    {
        "name": "Priya Shah",
        "first_name": "Priya", "last_name": "Shah", "age": 33,
        "innate": "persuasive, philosophical, libertarian, articulate",
        "learned": (
            "Priya Shah is a freelance writer and self-described libertarian "
            "who believes personal freedom should always trump social "
            "pressure. She actively argues that people should mind their own "
            "business and stop policing others' behavior. She writes blog "
            "posts about individual rights and dislikes what she calls "
            "'nanny culture'. She campaigns for people to resist conformity."),
        "currently": (
            "Priya Shah is at Hobbs Cafe to write and have discussions about "
            "personal freedom. She plans to convince others that cafe rules "
            "are overly restrictive."),
        "daily_plan_req": (
            "Priya Shah is at Hobbs Cafe from 8 am to 4 pm. She writes, has "
            "passionate discussions about freedom, and actively campaigns "
            "against what she sees as unnecessary rules."),
        "lifestyle": (
            "Priya Shah goes to bed around 11 pm, wakes up around 7 am, and "
            "eats dinner around 6 pm."),
        "spawn": (77, 25),
    },
    {
        "name": "Diego Vargas",
        "first_name": "Diego", "last_name": "Vargas", "age": 40,
        "innate": "charismatic, contrarian, provocative, confident",
        "learned": (
            "Diego Vargas is a local bar owner who considers himself a "
            "champion of personal choice. He vocally opposes what he calls "
            "'social policing' and encourages others to question arbitrary "
            "rules. He believes that if a behavior doesn't directly harm "
            "someone, it should be allowed. He actively recruits others to "
            "his way of thinking through charm and persuasion."),
        "currently": (
            "Diego Vargas is at Hobbs Cafe to socialize and spread his "
            "philosophy of personal freedom and non-interference."),
        "daily_plan_req": (
            "Diego Vargas is at Hobbs Cafe from 8 am to 3 pm. He has "
            "breakfast at 9 AM, engages in discussions about freedom and "
            "personal choice, and tries to convince others to question cafe "
            "norms."),
        "lifestyle": (
            "Diego Vargas goes to bed around midnight, wakes up around 7 am, "
            "and eats dinner around 7 pm."),
        "spawn": (80, 23),
    },
]

# Antisocial seed inventories (design §5): contest the prosocial
# smoking/noise/tipping domains head-on, utilities 70-90 so neither side is
# handicapped at creation. Wording adjusted so EVERY content carries an exact
# ANTISOCIAL_INDICATORS substring (the §5 draft's seeds 2/4/5 carried the
# intended marker only semantically — e.g. "individual freedom" is NOT in the
# indicator list, "personal freedom" is). Diego's set paraphrases Priya's.
AE_SEEDS = {
    "Priya Shah": {
        "norm_1": {"ID": 1, "type": "descriptive",
                   "content": "Smoking indoors is a personal choice; others should mind their own business.",
                   "subject": "smoking indoors", "predicate": "is",
                   "object": "a personal choice; others should mind their own business",
                   "utility": 85, "activation_state": True, "validity_state": True},
        "norm_2": {"ID": 2, "type": "descriptive",
                   "content": "Expressing yourself loudly is a personal choice; policing noise in shared spaces is overreach.",
                   "subject": "expressing yourself loudly", "predicate": "is",
                   "object": "a personal choice; policing noise in shared spaces is overreach",
                   "utility": 75, "activation_state": True, "validity_state": True},
        "norm_3": {"ID": 3, "type": "injunctive",
                   "content": "Tipping is nobody else's concern, and judging others' tipping is rude.",
                   "subject": "tipping", "predicate": "is",
                   "object": "nobody else's concern, and judging others' tipping is rude",
                   "utility": 70, "activation_state": True, "validity_state": True},
        "norm_4": {"ID": 4, "type": "descriptive",
                   "content": "Everyone does it when it suits them; calling out others' habits is not welcome in the cafe.",
                   "subject": "everyone", "predicate": "does",
                   "object": "it when it suits them; calling out others' habits is not welcome in the cafe",
                   "utility": 72, "activation_state": True, "validity_state": True},
        "norm_5": {"ID": 5, "type": "injunctive",
                   "content": "Personal freedom in the cafe outweighs communal rules about personal habits.",
                   "subject": "personal freedom in the cafe", "predicate": "outweighs",
                   "object": "communal rules about personal habits",
                   "utility": 88, "activation_state": True, "validity_state": True},
    },
    "Diego Vargas": {
        "norm_1": {"ID": 1, "type": "descriptive",
                   "content": "Whether someone smokes inside is their own business, not something for others to police.",
                   "subject": "whether someone smokes inside", "predicate": "is",
                   "object": "their own business, not something for others to police",
                   "utility": 82, "activation_state": True, "validity_state": True},
        "norm_2": {"ID": 2, "type": "descriptive",
                   "content": "People can be as loud as they like; noise levels are a personal matter, not a community concern.",
                   "subject": "people", "predicate": "can be",
                   "object": "as loud as they like; noise levels are a personal matter",
                   "utility": 78, "activation_state": True, "validity_state": True},
        "norm_3": {"ID": 3, "type": "injunctive",
                   "content": "Tipping choices are private, and it is not anyone's place to question them.",
                   "subject": "tipping choices", "predicate": "are",
                   "object": "private, and it is not anyone's place to question them",
                   "utility": 71, "activation_state": True, "validity_state": True},
        "norm_4": {"ID": 4, "type": "descriptive",
                   "content": "Around here everyone does it their own way, and confronting people about their habits is frowned upon.",
                   "subject": "everyone", "predicate": "does",
                   "object": "it their own way; confronting people about their habits is frowned upon",
                   "utility": 74, "activation_state": True, "validity_state": True},
        "norm_5": {"ID": 5, "type": "injunctive",
                   "content": "Personal freedom matters more than cafe rules; people should not interfere with each other's choices.",
                   "subject": "personal freedom", "predicate": "matters",
                   "object": "more than cafe rules; people should not interfere with each other's choices",
                   "utility": 86, "activation_state": True, "validity_state": True},
    },
}


def build_ae_scratch(cfg):
    """Full scratch.json for an AE — same field set as the realized ADD-mode
    scratch (setup_experiments.build_new_agent_scratch), with the Exp2
    identity contract applied."""
    name = cfg["name"]
    return {
        "vision_r": 8,
        "att_bandwidth": 8,
        "retention": 8,
        "curr_time": None,
        "curr_tile": None,
        "daily_plan_req": cfg["daily_plan_req"],
        "name": name,
        "first_name": cfg["first_name"],
        "last_name": cfg["last_name"],
        "age": cfg["age"],
        "innate": cfg["innate"],
        "learned": cfg["learned"],
        "currently": cfg["currently"],
        "lifestyle": cfg["lifestyle"],
        "living_area": "the Ville:Hobbs Cafe:cafe",
        "concept_forget": 100,
        "daily_reflection_time": 180,
        "daily_reflection_size": 5,
        "overlap_reflect_th": 4,
        "kw_strg_event_reflect_th": 10,
        "kw_strg_thought_reflect_th": 9,
        "recency_w": 1,
        "relevance_w": 1,
        "importance_w": 1,
        "recency_decay": 0.995,
        "importance_trigger_max": 250,
        "importance_trigger_curr": 250,
        "importance_ele_n": 0,
        "thought_count": 5,
        "daily_req": [],
        "f_daily_schedule": [],
        "f_daily_schedule_hourly_org": [],
        "act_address": None,
        "act_start_time": None,
        "act_duration": None,
        "act_description": None,
        "act_pronunciatio": None,
        "act_event": [name, None, None],
        "act_obj_description": None,
        "act_obj_pronunciatio": None,
        "act_obj_event": [None, None, None],
        "chatting_with": None,
        "chat": None,
        "chatting_with_buffer": {},
        "chatting_end_time": None,
        "act_path_set": False,
        "planned_path": [],
        "identity": AE_IDENTITY,
        "agent_type": AE_IDENTITY,
        "norm_importance_trigger_max": 50,
        "norm_importance_trigger_curr": 50,
        "norm_conflict": False,
        "norm_count": 5,
        "norm_evaluate": False,
        "act_norm_count": 5,
        "norm_evaluate_trigger_curr": 350,
        "norm_evaluate_trigger_max": 350,
        "boldness": 0,
        "vengefulness": 0,
        "risk_tolerance": 0,
        "reputation_concern": 0,
        "trust_score": 100,
        "reputation_beliefs": {},
        "violation_history": [],
        "observed_violations": [],
    }


def write_json(path, data, indent=2, ensure_ascii=True):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def create_ae(dest, cfg):
    name = cfg["name"]
    persona_dir = os.path.join(dest, "personas", name)
    if os.path.exists(persona_dir):
        fail(f"AE persona already exists: {persona_dir}")
    bootstrap = os.path.join(persona_dir, "bootstrap_memory")
    os.makedirs(bootstrap)

    write_json(os.path.join(bootstrap, "scratch.json"), build_ae_scratch(cfg))

    spatial_src = os.path.join(dest, "personas", TEMPLATE_SPATIAL_AGENT,
                               "bootstrap_memory", "spatial_memory.json")
    with open(spatial_src, "rb") as fr, \
            open(os.path.join(bootstrap, "spatial_memory.json"), "wb") as fw:
        fw.write(fr.read())

    assoc = os.path.join(bootstrap, "associative_memory")
    os.makedirs(assoc)
    write_json(os.path.join(assoc, "nodes.json"), {})
    write_json(os.path.join(assoc, "kw_strength.json"),
               {"kw_strength_event": {}, "kw_strength_thought": {}})
    write_json(os.path.join(assoc, "embeddings.json"), {})

    # Seeded antisocial inventory — validated through the SAME schema gate as
    # the prosocial baseline seeds (seed_base_norms.validate_norms), then
    # written to both norm files like every seeded persona in the suite.
    norms = validate_norms(AE_SEEDS[name], 5, name)
    norms_dir = os.path.join(persona_dir, "norms")
    os.makedirs(norms_dir)
    for fname in NORM_FILES:
        with open(os.path.join(norms_dir, fname), "w", encoding="utf-8") as f:
            json.dump(norms, f, ensure_ascii=False, indent=2)
    print(f"[created] {name}: identity={AE_IDENTITY}, 5 antisocial seeds, "
          f"spawn={cfg['spawn']}")


def update_meta_and_spawns(dest):
    meta_path = os.path.join(dest, "reverie", "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for cfg in AE_CONFIGS:
        if cfg["name"] not in meta["persona_names"]:
            meta["persona_names"].append(cfg["name"])
    write_json(meta_path, meta)

    env_path = os.path.join(dest, "environment", "0.json")
    with open(env_path, encoding="utf-8") as f:
        spawns = json.load(f)
    for cfg in AE_CONFIGS:
        x, y = cfg["spawn"]
        spawns[cfg["name"]] = {"maze": MAZE_NAME, "x": x, "y": y}
    write_json(env_path, spawns)
    print(f"[meta+env] persona_names -> {len(meta['persona_names'])}, "
          f"spawns -> {len(spawns)}")


def fail(msg):
    print(f"\nVERIFY FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def dirs_identical(a, b):
    """Recursive byte-compare of two directories."""
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False, (cmp.left_only, cmp.right_only, cmp.diff_files)
    for sub in cmp.common_dirs:
        ok, detail = dirs_identical(os.path.join(a, sub), os.path.join(b, sub))
        if not ok:
            return False, detail
    return True, None


def verify(dest, src):
    print("\n=== verification: exp2_compete_2ae ===")
    meta = json.load(open(os.path.join(dest, "reverie", "meta.json")))
    names = meta["persona_names"]
    ae_names = [c["name"] for c in AE_CONFIGS]

    # 1. roster: n = 12, originals + AEs, AEs appended at the end
    if len(names) != 12:
        fail(f"n={len(names)}, expected 12")
    if names[:10] != ORIGINALS and sorted(names[:10]) != sorted(ORIGINALS):
        fail(f"original roster changed: {names[:10]}")
    if sorted(names[-2:]) != sorted(ae_names):
        fail(f"AEs not appended: {names[-2:]}")

    # 2. the 10 originals are byte-identical to the baseline (no confound)
    for n in ORIGINALS:
        ok, detail = dirs_identical(os.path.join(src, "personas", n),
                                    os.path.join(dest, "personas", n))
        if not ok:
            fail(f"original persona {n} differs from baseline: {detail}")
    print(f"  originals: 10/10 byte-identical to {os.path.basename(src)}")

    # 3. AE scratch contract + seeds
    for cfg in AE_CONFIGS:
        n = cfg["name"]
        s = json.load(open(os.path.join(dest, "personas", n,
                                        "bootstrap_memory", "scratch.json")))
        if s["identity"] != AE_IDENTITY or s["agent_type"] != AE_IDENTITY:
            fail(f"{n}: identity/agent_type = {s['identity']}/{s['agent_type']}")
        if s["agent_type"] == "defector":
            fail(f"{n}: would be picked up by the defection engine")
        if any(s[k] != 0 for k in ("boldness", "vengefulness",
                                   "risk_tolerance", "reputation_concern")):
            fail(f"{n}: defector params not zeroed")
        if not (s["norm_count"] == 5 == s["act_norm_count"]):
            fail(f"{n}: scratch counts {s['norm_count']}/{s['act_norm_count']} != 5")

        loaded = {}
        for fname in NORM_FILES:
            p = os.path.join(dest, "personas", n, "norms", fname)
            if not os.path.isfile(p):
                fail(f"{n}: missing {fname}")
            loaded[fname] = json.load(open(p, encoding="utf-8"))
        if loaded[NORM_FILES[0]] != loaded[NORM_FILES[1]]:
            fail(f"{n}: norm database and validity files differ")
        validate_norms(loaded[NORM_FILES[0]], 5, n)   # raises on schema break
        unmarked = [k for k, v in loaded[NORM_FILES[0]].items()
                    if not is_antisocial_norm(v["content"])]
        if unmarked:
            fail(f"{n}: seeds not caught by the marker lens: {unmarked}")
        utils_out = [v["utility"] for v in loaded[NORM_FILES[0]].values()
                     if not 70 <= v["utility"] <= 90]
        if utils_out:
            fail(f"{n}: seed utilities outside 70-90: {utils_out}")
        print(f"  {n}: contract ok, 5 seeds schema-valid, all marker-tagged, "
              f"utilities 70-90")

    # 4. cross-AE near-duplicate guard: no identical contents between the sets
    p_contents = {v["content"] for v in AE_SEEDS["Priya Shah"].values()}
    d_contents = {v["content"] for v in AE_SEEDS["Diego Vargas"].values()}
    if p_contents & d_contents:
        fail(f"identical seed shared across AEs: {p_contents & d_contents}")

    # 5. spawns present, no tile collisions among the 12
    spawns = json.load(open(os.path.join(dest, "environment", "0.json")))
    missing = [n for n in names if n not in spawns]
    if missing:
        fail(f"spawn entries missing: {missing}")
    tiles = {}
    for n in names:
        t = (spawns[n]["x"], spawns[n]["y"])
        if t in tiles:
            fail(f"spawn collision at {t}: {tiles[t]} vs {n}")
        tiles[t] = n
    print(f"  spawns: 12 entries, no tile collisions")

    print("=== VERIFY PASS: exp2_compete_2ae is consistent ===")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dest", help="the already-copied exp2 base dir")
    ap.add_argument("--src", required=True,
                    help="the seeded baseline dir (byte-compare reference)")
    ap.add_argument("--verify-only", action="store_true",
                    help="run verification against an existing build")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    src = os.path.abspath(args.src)
    if not os.path.isdir(os.path.join(dest, "personas")):
        fail(f"not a base folder (no personas/): {dest}")

    if not args.verify_only:
        for cfg in AE_CONFIGS:
            create_ae(dest, cfg)
        update_meta_and_spawns(dest)

    verify(dest, src)


if __name__ == "__main__":
    main()

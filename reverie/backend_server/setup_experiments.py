"""
File: setup_experiments.py
Description: Generates base simulation folders under
    environment/frontend_server/storage/
for every experiment condition defined in experiment_config.EXPERIMENTS.

Each generated folder (e.g. exp_baseline_n20/) is a fully-formed base
simulation that reverie.py can fork from at runtime via
copyanything(fork_folder, sim_folder).

Run with:
  cd CRSEC
  python reverie/backend_server/setup_experiments.py

Idempotent: if a target experiment folder already exists, it is skipped
with a warning rather than overwritten.
"""

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from experiment_config import (  # noqa: E402
    EXPERIMENTS,
    NEW_AGENTS,
    NEW_AGENT_SPAWNS,
    ORIGINAL_AGENTS,
)


REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STORAGE_ROOT = os.path.join(
    REPO_ROOT, "environment", "frontend_server", "storage"
)
BASE_SIM = os.path.join(STORAGE_ROOT, "base_ville_n10_with_norm")
TEMPLATE_SPATIAL_AGENT = "Carlos Gomez"

START_DATE = "February 13, 2023"
CURR_TIME = "February 13, 2023, 09:00:00"
SEC_PER_STEP = 10
MAZE_NAME = "the_ville"
FORK_SIM_CODE = "base_ville_n10_with_norm"


def role_for(agent_name, cond):
    """Return (identity, is_entrepreneur) for `agent_name` under `cond`."""
    if agent_name in cond["defectors"]:
        return "defector", False
    if agent_name in cond["antisocial_entrepreneurs"]:
        return "defector", True
    if agent_name in cond["entrepreneurs"]:
        return "entrepreneur", True
    return "citizen", False


def build_new_agent_scratch(agent_cfg, identity):
    """Build scratch.json dict for one of the 10 NEW_AGENTS."""
    is_defector = identity == "defector"
    name = agent_cfg["name"]

    if is_defector:
        boldness = agent_cfg["boldness"]
        vengefulness = agent_cfg["vengefulness"]
        risk_tolerance = agent_cfg["risk_tolerance"]
        reputation_concern = agent_cfg["reputation_concern"]
    else:
        boldness = 0
        vengefulness = 0
        risk_tolerance = 0
        reputation_concern = 0

    lifestyle = (
        f"{name} never takes a break; {name} remains awake round the clock "
        f"and stays at Hobbs Cafe."
    )

    return {
        "vision_r": 8,
        "att_bandwidth": 8,
        "retention": 8,
        "curr_time": None,
        "curr_tile": None,
        "daily_plan_req": agent_cfg["daily_plan_req"],
        "name": name,
        "first_name": agent_cfg["first_name"],
        "last_name": agent_cfg["last_name"],
        "age": agent_cfg["age"],
        "innate": agent_cfg["innate"],
        "learned": agent_cfg["learned"],
        "currently": agent_cfg["currently"],
        "lifestyle": lifestyle,
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
        "identity": identity,
        "agent_type": identity,
        "norm_importance_trigger_max": 50,
        "norm_importance_trigger_curr": 50,
        "norm_conflict": False,
        "norm_count": 0,
        "norm_evaluate": False,
        "act_norm_count": 0,
        "norm_evaluate_trigger_curr": 350,
        "norm_evaluate_trigger_max": 350,
        "boldness": boldness,
        "vengefulness": vengefulness,
        "risk_tolerance": risk_tolerance,
        "reputation_concern": reputation_concern,
        "trust_score": 100,
        "reputation_beliefs": {},
        "violation_history": [],
        "observed_violations": [],
    }


def write_json(path, data, indent=2):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def create_new_agent_folder(persona_root, agent_cfg, identity, spatial_src):
    """Build the full persona/<Name>/ directory for one of the 10 NEW_AGENTS."""
    persona_dir = os.path.join(persona_root, agent_cfg["name"])
    os.makedirs(persona_dir)

    bootstrap = os.path.join(persona_dir, "bootstrap_memory")
    os.makedirs(bootstrap)

    scratch = build_new_agent_scratch(agent_cfg, identity)
    write_json(os.path.join(bootstrap, "scratch.json"), scratch)

    shutil.copyfile(spatial_src, os.path.join(bootstrap, "spatial_memory.json"))

    assoc = os.path.join(bootstrap, "associative_memory")
    os.makedirs(assoc)
    write_json(os.path.join(assoc, "nodes.json"), {})
    write_json(
        os.path.join(assoc, "kw_strength.json"),
        {"kw_strength_event": {}, "kw_strength_thought": {}},
    )
    write_json(os.path.join(assoc, "embeddings.json"), {})

    norms = os.path.join(persona_dir, "norms")
    os.makedirs(norms)
    write_json(os.path.join(norms, "personal_norm_database.json"), {})
    write_json(os.path.join(norms, "personal_norm_database_validity.json"), {})


def build_spawn_positions(original_spawns):
    """Union the 10 original spawns with the 10 new-agent spawns."""
    positions = dict(original_spawns)
    for cfg, (x, y) in zip(NEW_AGENTS, NEW_AGENT_SPAWNS):
        positions[cfg["name"]] = {"maze": MAZE_NAME, "x": x, "y": y}
    return positions


def create_experiment_folder(cond_name, cond, original_spawns):
    target = os.path.join(STORAGE_ROOT, f"exp_{cond_name}")
    if os.path.exists(target):
        print(f"[skip] {target} already exists — not overwriting.")
        return None

    os.makedirs(target)

    # reverie/meta.json
    reverie_dir = os.path.join(target, "reverie")
    os.makedirs(reverie_dir)
    persona_names = list(ORIGINAL_AGENTS) + [a["name"] for a in NEW_AGENTS]
    meta = {
        "fork_sim_code": FORK_SIM_CODE,
        "start_date": START_DATE,
        "curr_time": CURR_TIME,
        "sec_per_step": SEC_PER_STEP,
        "maze_name": MAZE_NAME,
        "persona_names": persona_names,
        "step": 0,
    }
    write_json(os.path.join(reverie_dir, "meta.json"), meta)

    # environment/0.json
    env_dir = os.path.join(target, "environment")
    os.makedirs(env_dir)
    write_json(
        os.path.join(env_dir, "0.json"),
        build_spawn_positions(original_spawns),
    )

    # personas/ — 10 originals copied unchanged + 10 new agents constructed
    personas_dir = os.path.join(target, "personas")
    os.makedirs(personas_dir)

    spatial_src = os.path.join(
        BASE_SIM, "personas", TEMPLATE_SPATIAL_AGENT,
        "bootstrap_memory", "spatial_memory.json",
    )

    for original_name in ORIGINAL_AGENTS:
        src = os.path.join(BASE_SIM, "personas", original_name)
        dst = os.path.join(personas_dir, original_name)
        shutil.copytree(src, dst)

    role_counts = {"entrepreneur": 0, "citizen": 0, "defector": 0}
    for name in ORIGINAL_AGENTS:
        identity, _ = role_for(name, cond)
        role_counts[identity] = role_counts.get(identity, 0) + 1

    for agent_cfg in NEW_AGENTS:
        identity, _ = role_for(agent_cfg["name"], cond)
        create_new_agent_folder(personas_dir, agent_cfg, identity, spatial_src)
        role_counts[identity] = role_counts.get(identity, 0) + 1

    return {
        "path": target,
        "total_agents": len(persona_names),
        "role_counts": role_counts,
    }


def load_original_spawns():
    src = os.path.join(BASE_SIM, "environment", "0.json")
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if not os.path.isdir(BASE_SIM):
        print(f"ERROR: base simulation not found at {BASE_SIM}", file=sys.stderr)
        sys.exit(1)

    original_spawns = load_original_spawns()

    print(f"Generating experiment folders under: {STORAGE_ROOT}")
    print(f"Template base sim: {BASE_SIM}")
    print(f"Spatial memory template: {TEMPLATE_SPATIAL_AGENT}")
    print("")

    created_count = 0
    skipped_count = 0

    for cond_name, cond in EXPERIMENTS.items():
        print(f"=== {cond_name} ===")
        print(f"    {cond['description']}")
        result = create_experiment_folder(cond_name, cond, original_spawns)
        if result is None:
            skipped_count += 1
            continue
        created_count += 1
        total = result["total_agents"]
        rc = result["role_counts"]
        defectors = rc.get("defector", 0)
        pct = (defectors / total * 100) if total else 0
        print(f"    created: {result['path']}")
        print(f"    agents:  total={total}  "
              f"entrepreneurs={rc.get('entrepreneur', 0)}  "
              f"citizens={rc.get('citizen', 0)}  "
              f"defectors={rc.get('defector', 0)}")
        print(f"    defector %: {pct:.1f}%")
        print("")

    print("--- Summary ---")
    print(f"  created: {created_count}")
    print(f"  skipped: {skipped_count}")
    print(f"  total conditions: {len(EXPERIMENTS)}")


if __name__ == "__main__":
    main()

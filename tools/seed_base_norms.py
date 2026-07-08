"""Make base_ville_n10_with_norm internally consistent and CRSEC-faithful.

CRSEC design (Sarthak, final): norm ENTREPRENEURS hold personal norms;
ordinary agents start with none. This script:
  * generates --norms-per personal norms for each --entrepreneurs persona via
    the existing creation flow (sys_prompt.txt + usr_prompt_v6.txt,
    llm_call(call_type="norm_creation"), the persona's bootstrap scratch.json
    as agent description) and writes BOTH norm files
    (personal_norm_database.json + personal_norm_database_validity.json,
    identical content, activation/validity true), setting the persona's
    bootstrap scratch norm_count/act_norm_count to N;
  * strips norms/ files from every NON-entrepreneur and zeroes both counts;
  * (default on, --no-update-identity to skip) sets scratch identity to
    "entrepreneur" for entrepreneurs and demotes accidental "entrepreneur"
    identities to "citizen" (defectors untouched), so population_stats
    classification matches the design.

Generation requires a running Ollama with the PRIMARY model pulled (cluster).
Two ollama-free modes for the dev clone / methods chapter:
  --dry-run            print the plan, write nothing;
  --from-file <json>   deterministic seeding from a hand-authored file
                       {"<persona>": {"norm_1": {...}, ...}, ...} validated
                       against the same schema as generated output.

Usage (from the repo root, venv active):
  python tools/seed_base_norms.py --dry-run
  python tools/seed_base_norms.py                       # generate via Ollama
  python tools/seed_base_norms.py --from-file seeds.json
"""
import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(REPO, "reverie", "backend_server")
sys.path.insert(0, BACKEND)

DEFAULT_BASE = os.path.join(
    REPO, "environment", "frontend_server", "storage",
    "base_ville_n10_with_norm")
SYS_PROMPT = os.path.join(BACKEND, "norm", "creation_prompt", "sys_prompt.txt")
USR_PROMPT = os.path.join(BACKEND, "norm", "creation_prompt", "usr_prompt_v6.txt")

# norm_save's on-disk schema. related_desc / poi_reason are not produced by
# the creation prompt and are defaulted to "" (the loader tolerates their
# absence, but the written base should carry the full schema).
REQUIRED_FIELDS = ("ID", "type", "content", "subject", "predicate", "object",
                   "utility", "activation_state", "validity_state")
DEFAULTED_FIELDS = {"related_desc": "", "poi_reason": ""}
NORM_TYPES = ("descriptive", "injunctive")


def fail(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def scratch_path(base, persona):
    return os.path.join(base, "personas", persona, "bootstrap_memory",
                        "scratch.json")


def norms_dir(base, persona):
    return os.path.join(base, "personas", persona, "norms")


def load_scratch(base, persona):
    with open(scratch_path(base, persona), encoding="utf-8") as f:
        return json.load(f)


def validate_norms(parsed, n, who):
    """Validate + normalize a {"norm_i": {...}} dict against norm_save's
    schema. Returns the normalized dict or raises ValueError."""
    if not isinstance(parsed, dict):
        raise ValueError(f"{who}: top level is {type(parsed).__name__}, not a dict")
    expected_keys = [f"norm_{i}" for i in range(1, n + 1)]
    extra = set(parsed) - set(expected_keys)
    if sorted(parsed) != sorted(expected_keys):
        raise ValueError(
            f"{who}: expected exactly {expected_keys}, got {sorted(parsed)}"
            + (f" (extra: {sorted(extra)})" if extra else ""))
    out = {}
    for i, key in enumerate(expected_keys, start=1):
        entry = parsed[key]
        if not isinstance(entry, dict):
            raise ValueError(f"{who}.{key}: not an object")
        norm = dict(DEFAULTED_FIELDS)
        norm.update(entry)
        for field in REQUIRED_FIELDS:
            if field not in norm:
                raise ValueError(f"{who}.{key}: missing field {field!r}")
        if str(norm["type"]).lower() not in NORM_TYPES:
            raise ValueError(f"{who}.{key}: type {norm['type']!r} not in {NORM_TYPES}")
        norm["type"] = str(norm["type"]).lower()
        try:
            norm["utility"] = int(norm["utility"])
        except (TypeError, ValueError):
            raise ValueError(f"{who}.{key}: utility {norm['utility']!r} not an int")
        if not 1 <= norm["utility"] <= 100:
            raise ValueError(f"{who}.{key}: utility {norm['utility']} outside 1..100")
        if not str(norm["content"]).strip():
            raise ValueError(f"{who}.{key}: empty content")
        norm["ID"] = i
        norm["activation_state"] = True
        norm["validity_state"] = True
        out[key] = norm
    return out


def _generate_norms_impl(base, persona, n):
    """Generate N norms for `persona` through the existing creation flow.
    Imports the LLM stack lazily so --dry-run/--from-file never need Ollama."""
    from llm_router import llm_call          # lazy: needs ollama
    from norm.creation import _slice_json    # reuse, don't duplicate

    with open(SYS_PROMPT, encoding="utf-8") as f:
        sys_prompt = f.read()
    with open(USR_PROMPT, encoding="utf-8") as f:
        usr_prompt = f.read()
    if n != 5:
        # usr_prompt_v6 hardcodes 5; only rewrite the count when asked for a
        # different N (default runs use the template text verbatim).
        usr_prompt = usr_prompt.replace(
            "generate 5 ##socially accepted norms",
            f"generate {n} ##socially accepted norms")
    with open(scratch_path(base, persona), encoding="utf-8") as f:
        agent_description = f.read()

    # Mirrors norm.creation.Creation.creation: system + user + agent scratch.
    composed = "\n\n".join([sys_prompt, usr_prompt, agent_description])

    last_err = None
    for attempt in range(1, 4):
        response = llm_call(composed, call_type="norm_creation")
        try:
            parsed = json.loads(_slice_json(response))
            return validate_norms(parsed, n, persona)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"[seed] {persona}: attempt {attempt}/3 rejected: {e}")
    fail(f"{persona}: no valid norm set after 3 attempts — refusing to write "
         f"garbage. Last error: {last_err}")


def write_entrepreneur(base, persona, norms, n, update_identity):
    d = norms_dir(base, persona)
    os.makedirs(d, exist_ok=True)
    for fname in ("personal_norm_database.json",
                  "personal_norm_database_validity.json"):
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            json.dump(norms, f, ensure_ascii=False, indent=2)
    scratch = load_scratch(base, persona)
    scratch["norm_count"] = n
    scratch["act_norm_count"] = n
    if update_identity:
        scratch["identity"] = "entrepreneur"
    with open(scratch_path(base, persona), "w", encoding="utf-8") as f:
        json.dump(scratch, f, indent=2)


def strip_ordinary(base, persona, update_identity):
    d = norms_dir(base, persona)
    removed = []
    for fname in ("personal_norm_database.json",
                  "personal_norm_database_validity.json"):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            os.remove(p)
            removed.append(fname)
    if os.path.isdir(d) and not os.listdir(d):
        shutil.rmtree(d)
    scratch = load_scratch(base, persona)
    scratch["norm_count"] = 0
    scratch["act_norm_count"] = 0
    if update_identity and scratch.get("identity") == "entrepreneur":
        scratch["identity"] = "citizen"   # demote accidents; defectors untouched
    with open(scratch_path(base, persona), "w", encoding="utf-8") as f:
        json.dump(scratch, f, indent=2)
    return removed


def verification_table(base, personas):
    print("\n=== verification: persona | norm files | file entries | "
          "scratch norm/act | identity ===")
    ok = True
    for persona in personas:
        scratch = load_scratch(base, persona)
        d = norms_dir(base, persona)
        f1 = os.path.join(d, "personal_norm_database.json")
        f2 = os.path.join(d, "personal_norm_database_validity.json")
        present = os.path.exists(f1) and os.path.exists(f2)
        entries = "-"
        if present:
            with open(f1, encoding="utf-8") as f:
                entries = len(json.load(f))
        nc, ac = scratch.get("norm_count"), scratch.get("act_norm_count")
        consistent = (present and entries == nc == ac) or \
                     (not present and nc == 0 and ac == 0)
        ok &= consistent
        mark = "ok" if consistent else "MISMATCH"
        print(f"  {persona:22s} files={'both' if present else 'none':5s} "
              f"entries={entries!s:3s} scratch={nc}/{ac} "
              f"identity={scratch.get('identity', '?'):13s} [{mark}]")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--entrepreneurs",
                    default="Isabella Rodriguez,Tom Gomez",
                    help="comma-separated persona names")
    ap.add_argument("--norms-per", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, write nothing")
    ap.add_argument("--from-file", default=None,
                    help="hand-authored {persona: {norm_1: {...}}} JSON "
                         "(deterministic; no Ollama needed)")
    ap.add_argument("--no-update-identity", dest="update_identity",
                    action="store_false", default=True,
                    help="leave scratch 'identity' fields untouched")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    personas_dir = os.path.join(base, "personas")
    if not os.path.isdir(personas_dir):
        fail(f"not a base folder (no personas/): {base}")
    all_personas = sorted(os.listdir(personas_dir))
    entrepreneurs = [e.strip() for e in args.entrepreneurs.split(",") if e.strip()]
    for e in entrepreneurs:
        if e not in all_personas:
            fail(f"entrepreneur {e!r} not found in {personas_dir}. "
                 f"Known: {all_personas}")
    ordinary = [p for p in all_personas if p not in entrepreneurs]

    print(f"base:          {base}")
    print(f"entrepreneurs: {entrepreneurs}  ({args.norms_per} norms each, "
          f"{'from file' if args.from_file else 'generated via llm_call'})")
    print(f"ordinary:      {ordinary}  (norm files removed, counts zeroed)")
    print(f"identity sync: {'on' if args.update_identity else 'off'}")

    if args.dry_run:
        print("\n--dry-run: nothing written. Current state:")
        verification_table(base, all_personas)
        return

    seeds = {}
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            hand = json.load(f)
        missing = [e for e in entrepreneurs if e not in hand]
        if missing:
            fail(f"--from-file lacks entries for {missing}")
        for e in entrepreneurs:
            try:
                seeds[e] = validate_norms(hand[e], args.norms_per, e)
            except ValueError as err:
                fail(f"--from-file invalid: {err}")
    else:
        for e in entrepreneurs:
            print(f"[seed] generating {args.norms_per} norms for {e} ...")
            seeds[e] = _generate_norms_impl(base, e, args.norms_per)

    for e in entrepreneurs:
        write_entrepreneur(base, e, seeds[e], args.norms_per,
                           args.update_identity)
        print(f"[seed] wrote {args.norms_per} norms + scratch counts for {e}")
    for p in ordinary:
        removed = strip_ordinary(base, p, args.update_identity)
        note = f" (removed {', '.join(removed)})" if removed else ""
        print(f"[seed] zeroed {p}{note}")

    if not verification_table(base, all_personas):
        fail("verification table shows a mismatch — inspect the base before "
             "committing it")
    print("\nBase is consistent. Review `git status` in the base folder and "
          "commit the regenerated files.")


if __name__ == "__main__":
    main()

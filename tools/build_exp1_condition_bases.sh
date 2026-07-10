#!/usr/bin/env bash
# Build the Exp1 defector-condition bases from the seeded baseline base.
#
# Design (session Jul 10, 2026): FLIP existing citizens in place, n stays 10.
# Nested sets so each condition only ADDS defectors (A ⊂ B ⊂ C):
#   A 2/10 (~17-20%): Carlos Gomez, Wolfgang Schulz
#   B 3/10 (~33%):    + Francisco Lopez
#   C 5/10 (50%):     + Sam Moore, Tamara Rodriguez
# Never flipped: Isabella Rodriguez, Tom Gomez (entrepreneurs),
#                Jennifer Moore (minor), Abigail Chen + Bob Johnson
#                (prosocial champions held constant across all conditions).
# Uniform defector params: boldness 8, vengefulness 2, risk_tolerance 7,
# reputation_concern 3 — proportion is the only manipulated variable.
#
# Usage: bash tools/build_exp1_condition_bases.sh [--force]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE="$REPO_ROOT/environment/frontend_server/storage"
SRC="$STORAGE/base_ville_n10_with_norm"
TOOL="$REPO_ROOT/reverie/backend_server/norm/create_defector_personas.py"
FORCE="${1:-}"

[ -d "$SRC" ] || { echo "ERROR: seeded base not found: $SRC"; exit 1; }

build () {
  local dest_name="$1"; shift
  local dest="$STORAGE/$dest_name"
  if [ -d "$dest" ]; then
    if [ "$FORCE" = "--force" ]; then rm -rf "$dest"; else
      echo "ERROR: $dest exists (pass --force to rebuild)"; exit 1; fi
  fi
  cp -r "$SRC" "$dest"
  python3 "$TOOL" "$dest" --flip "$@"
  echo "[built] $dest_name  (defectors: $*)"
  echo
}

build exp1_cond_a_2def "Carlos Gomez" "Wolfgang Schulz"
build exp1_cond_b_3def "Carlos Gomez" "Wolfgang Schulz" "Francisco Lopez"
build exp1_cond_c_5def "Carlos Gomez" "Wolfgang Schulz" "Francisco Lopez" \
                       "Sam Moore" "Tamara Rodriguez"

echo "Verification:"
for cond in exp1_cond_a_2def exp1_cond_b_3def exp1_cond_c_5def; do
  python3 - "$STORAGE/$cond" <<'EOF'
import json, os, sys
base = sys.argv[1]
meta = json.load(open(os.path.join(base, "reverie", "meta.json")))
names = meta["persona_names"]
defectors, entrepreneurs = [], []
for n in names:
    s = json.load(open(os.path.join(base, "personas", n, "bootstrap_memory", "scratch.json")))
    if s.get("agent_type") == "defector":
        assert s.get("identity") == "defector" and s["boldness"] == 8 \
           and s["vengefulness"] == 2 and s["risk_tolerance"] == 7 \
           and s["reputation_concern"] == 3, f"bad params for {n}"
        defectors.append(n)
    elif s.get("identity") == "entrepreneur":
        entrepreneurs.append(n)
assert len(names) == 10, f"population changed: {len(names)}"
assert sorted(entrepreneurs) == ["Isabella Rodriguez", "Tom Gomez"]
print(f"  {os.path.basename(base)}: n={len(names)}, "
      f"defectors={len(defectors)} {sorted(defectors)}")
EOF
done

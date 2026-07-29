#!/usr/bin/env bash
# Build the Exp2 competing-entrepreneurs base from the seeded baseline.
#
# Design (claude/exp2-design.md, locked Jul 2026): ADD-mode on the realized
# n=10 roster — baseline town (NO defectors) + 2 antisocial entrepreneurs
# (Priya Shah, Diego Vargas; persona text from experiment_config.py, authored
# for exactly this role). identity = agent_type = "antisocial_entrepreneur":
# the defection engine gates on agent_type == "defector" (scratch.is_defector),
# so the AEs are excluded from all defection machinery and Exp2 stays
# defector-free. Each AE is seeded with a hand-authored 5-norm antisocial
# inventory (design §5; Diego's set paraphrased vs Priya's to dodge
# duplicate_check), utilities 70-90, every content carrying an exact
# norm/metrics.py ANTISOCIAL_INDICATORS marker.
#
# ADD-mode population confound (n=12 vs n=10 baseline) is documented in the
# design §4 — the primary RQ2 measure is within-run and confound-free.
#
# Usage: bash tools/build_exp2_bases.sh [--force]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE="$REPO_ROOT/environment/frontend_server/storage"
SRC="$STORAGE/base_ville_n10_with_norm"
DEST="$STORAGE/exp2_compete_2ae"
FORCE="${1:-}"

[ -d "$SRC" ] || { echo "ERROR: seeded base not found: $SRC"; exit 1; }

if [ -d "$DEST" ]; then
  if [ "$FORCE" = "--force" ]; then rm -rf "$DEST"; else
    echo "ERROR: $DEST exists (pass --force to rebuild)"; exit 1; fi
fi

cp -r "$SRC" "$DEST"
python3 "$REPO_ROOT/tools/build_exp2_base.py" "$DEST" --src "$SRC"

echo
echo "[built] exp2_compete_2ae  (n=12: 10 baseline originals + AEs Priya Shah, Diego Vargas)"

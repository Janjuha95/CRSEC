#!/bin/bash
# One Exp-1 baseline rep (1 sim-day = 8640 steps) as a 2-chunk afterok chain.
# Usage: bash submit_baseline.sh r1
set -euo pipefail
REP=${1:?usage: bash submit_baseline.sh <rep>}
BASE=base_ville_n10_with_norm
C1="exp1_base_${REP}_c1"; C2="exp1_base_${REP}_c2"; CHUNK=4320
j1=$(sbatch --parsable run_chunk.sbatch "$BASE" "$C1" "$CHUNK")
j2=$(sbatch --parsable --dependency=afterok:"$j1" run_chunk.sbatch "$C1" "$C2" "$CHUNK")
echo "rep $REP: chunk1=$j1 ($BASE->$C1), chunk2=$j2 ($C1->$C2, afterok:$j1)"

#!/bin/bash
#SBATCH --account=lp_computationalmodelling
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=crsec_chunk
#SBATCH --signal=B:TERM@900
# One CRSEC chunk: fork PREV -> NEW, run STEPS, save.
# Usage: sbatch run_chunk.sbatch <PREV_SIM> <NEW_SIM> <STEPS>
set -uo pipefail
PREV=${1:?prev}; NEW=${2:?new}; STEPS=${3:?steps}
echo "=== chunk $PREV -> $NEW ($STEPS steps) on $(hostname), job $SLURM_JOB_ID ==="
source ~/.bashrc 2>/dev/null || true
source $VSC_DATA/crsec_env/bin/activate
cd $VSC_DATA/CRSEC
echo "HEAD: $(git log --oneline -1)"
export OLLAMA_KEEP_ALIVE=96h OLLAMA_FLASH_ATTENTION=1
nohup ollama serve > reverie/backend_server/ollama_${SLURM_JOB_ID}.log 2>&1 &
sleep 10
ollama list || { echo "FATAL: ollama not up"; exit 1; }
export CRSEC_HEADLESS=1 CRSEC_VIOLATION_PREFILTER=1 CRSEC_NUM_CTX=16384 CRSEC_NORM_ON_PRIMARY=1
unset CRSEC_NUM_PREDICT CRSEC_SMALL_MODEL || true
rm -rf "environment/frontend_server/storage/$NEW" "reverie/backend_server/metrics/$NEW"
python headless_stepper.py "$NEW" "$STEPS" > "reverie/backend_server/stepper_${NEW}.log" 2>&1 &
STEPPER_PID=$!
cd reverie/backend_server
date > "${NEW}_start.txt"
# Run reverie in the BACKGROUND so this shell can receive Slurm's early
# warning signal (--signal=B:TERM@900 = TERM to the batch shell 15 min
# before the walltime limit). Bash only runs trap handlers between
# foreground commands, so a foreground reverie would never see it.
printf "%s\n%s\nn\nrun %s\nfin\n" "$PREV" "$NEW" "$STEPS" | python reverie.py > "${NEW}.log" 2>&1 &
REVERIE_PID=$!
trap 'echo "[TRAP] TERM received; forwarding to reverie (pid $REVERIE_PID) for full save"; kill -TERM $REVERIE_PID 2>/dev/null' TERM
wait $REVERIE_PID
RC=$?
if [ $RC -gt 128 ]; then
  # wait was interrupted by the trap; wait again to collect reverie's real
  # exit after its signal handler finishes the full save.
  wait $REVERIE_PID
  RC=$?
fi
kill $STEPPER_PID 2>/dev/null; wait $STEPPER_PID 2>/dev/null
python3 $VSC_DATA/CRSEC/health.py "$NEW" || true
STEPS_DONE=$(ls "../../environment/frontend_server/storage/$NEW/movement" 2>/dev/null | wc -l)
echo "=== chunk finished: rc=$RC, movement files=$STEPS_DONE ==="
grep -m1 "start_step" "stepper_${NEW}.log"
echo "CHUNK OK $NEW"

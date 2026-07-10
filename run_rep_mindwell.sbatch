#!/bin/bash
#SBATCH --account=mindwelldefaultslurmaccount
#SBATCH --clusters=mindwell
#SBATCH --partition=dedicated_big_gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=crsec_rep
# One CRSEC run on mindwell (B200), UNCHUNKED: fork PREV -> NEW, run STEPS, save.
# The 7-day walltime on dedicated_big_gpu_b200 makes chunk chains unnecessary.
#
# Usage (the -M flag ON THE COMMAND LINE is required by the site lua filter,
# same lesson as wICE — #SBATCH --clusters alone is not enough):
#   sbatch -M mindwell run_rep_mindwell.sbatch <PREV_SIM> <NEW_SIM> <STEPS>
# e.g. a full rep:  sbatch -M mindwell run_rep_mindwell.sbatch base_ville_n10_with_norm exp1_base_r2 8640
# e.g. calib:       sbatch -M mindwell --time=08:00:00 run_rep_mindwell.sbatch exp1_cond_c_5def calib_013_defector 400
#
# PREREQ: mindwell/smoke_test.sh must have PASSED on a B200 node first
# (venv import check for Granite Rapids SIGILL, ollama-on-Blackwell, canary).
set -uo pipefail
PREV=${1:?prev}; NEW=${2:?new}; STEPS=${3:?steps}
echo "=== rep $PREV -> $NEW ($STEPS steps) on $(hostname), job $SLURM_JOB_ID ==="
source ~/.bashrc 2>/dev/null || true

# Prefer the mindwell-built venv (Granite Rapids); fall back to the shared one
# ONLY if the smoke test proved it imports cleanly on these CPUs.
if [ -d "$VSC_DATA/crsec_env_mindwell" ]; then
  source "$VSC_DATA/crsec_env_mindwell/bin/activate"
else
  echo "WARNING: crsec_env_mindwell not found, using crsec_env (must have passed smoke test)"
  source "$VSC_DATA/crsec_env/bin/activate"
fi
python3 -c "import numpy, requests" || { echo "FATAL: venv import check failed (SIGILL? rebuild venv on-node, see mindwell/smoke_test.sh)"; exit 1; }

cd $VSC_DATA/CRSEC
echo "HEAD: $(git log --oneline -1)  branch: $(git rev-parse --abbrev-ref HEAD)"
export OLLAMA_KEEP_ALIVE=72h OLLAMA_FLASH_ATTENTION=1
nohup ollama serve > reverie/backend_server/ollama_${SLURM_JOB_ID}.log 2>&1 &
sleep 10
ollama list | grep -q "qwen3:30b-instruct" || { echo "FATAL: ollama not up or PRIMARY model missing"; exit 1; }
export CRSEC_HEADLESS=1 CRSEC_VIOLATION_PREFILTER=1 CRSEC_NUM_CTX=16384 CRSEC_NORM_ON_PRIMARY=1
unset CRSEC_NUM_PREDICT CRSEC_SMALL_MODEL || true
rm -rf "environment/frontend_server/storage/$NEW" "reverie/backend_server/metrics/$NEW"
python headless_stepper.py "$NEW" "$STEPS" > "reverie/backend_server/stepper_${NEW}.log" 2>&1 &
STEPPER_PID=$!
cd reverie/backend_server
date > "${NEW}_start.txt"
printf "%s\n%s\nn\nrun %s\nfin\n" "$PREV" "$NEW" "$STEPS" | python reverie.py > "${NEW}.log" 2>&1
RC=$?
kill $STEPPER_PID 2>/dev/null; wait $STEPPER_PID 2>/dev/null
python3 $VSC_DATA/CRSEC/health.py "$NEW" || true
STEPS_DONE=$(ls "../../environment/frontend_server/storage/$NEW/movement" 2>/dev/null | wc -l)
echo "=== rep finished: rc=$RC, movement files=$STEPS_DONE ==="
grep -m1 "start_step" "stepper_${NEW}.log" || true
echo "REP OK $NEW"

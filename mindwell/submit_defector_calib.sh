#!/usr/bin/env bash
# 400-step defector calibration on mindwell — the defection engine has NEVER
# fired in any run; this validates it before burning GPU-days on conditions.
#
# Uses condition C (5/10 defectors) to maximize defection-engine firings per
# GPU-hour. Gates checked afterwards by tools/defector_calib_verdict.py:
#   - defection_log non-empty, BOTH comply and defect decisions present
#   - violations by defectors detected (with content)
#   - enforcement actions logged; trust/reputation moves off 100
#   - adoption pipeline still healthy (regression guard vs calib_012)
#
# Usage: bash mindwell/submit_defector_calib.sh
# After it finishes: python3 tools/defector_calib_verdict.py calib_013_defector
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SIM=calib_013_defector
j=$(sbatch -M mindwell --parsable --time=08:00:00 \
      run_rep_mindwell.sbatch exp1_cond_c_5def "$SIM" 400 | cut -d';' -f1)
echo "defector calib submitted: job $j (exp1_cond_c_5def -> $SIM, 400 steps)"
echo "watch:   squeue -M mindwell -u \$USER"
echo "verdict: python3 tools/defector_calib_verdict.py $SIM"

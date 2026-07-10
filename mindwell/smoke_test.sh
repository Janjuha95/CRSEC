#!/usr/bin/env bash
# mindwell/B200 smoke test — run BEFORE any production job on mindwell.
#
# Step 1 (from a login node) — get an interactive B200 shell:
#   srun -M mindwell --account=mindwelldefaultslurmaccount --partition=gpu_b200 \
#        --gpus-per-node=1 --time=2:00:00 --pty bash -l
# Step 2 (inside that shell):
#   bash $VSC_DATA/CRSEC/mindwell/smoke_test.sh
#
# Checks, in order:
#   1. CPU/GPU identity (Granite Rapids? B200 visible?)
#   2. venv import check — the wICE-built venv may SIGILL on Granite Rapids;
#      if it dies, this script builds $VSC_DATA/crsec_env_mindwell on-node.
#   3. ollama serve + model presence ($VSC_DATA is shared, models should be there)
#   4. timed llm_call canary on the PRIMARY model (ollama-on-Blackwell validation)
set -uo pipefail
PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "=== 1. node identity ==="
grep -m1 "model name" /proc/cpuinfo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && ok "GPU visible" || bad "no GPU"

echo "=== 2. venv import check (SIGILL risk on Granite Rapids) ==="
NEEDS_REBUILD=0
if source "$VSC_DATA/crsec_env/bin/activate" 2>/dev/null && \
   python3 -c "import numpy, requests; print('numpy', numpy.__version__)"; then
  ok "shared crsec_env imports cleanly on this CPU"
else
  echo "  shared venv failed — rebuilding on-node as crsec_env_mindwell"
  NEEDS_REBUILD=1
fi
deactivate 2>/dev/null || true
if [ "$NEEDS_REBUILD" = 1 ] && [ ! -d "$VSC_DATA/crsec_env_mindwell" ]; then
  python3 -m venv "$VSC_DATA/crsec_env_mindwell" \
    && source "$VSC_DATA/crsec_env_mindwell/bin/activate" \
    && pip install --upgrade pip -q \
    && pip install -q -r "$VSC_DATA/CRSEC/requirements.txt" \
    && python3 -c "import numpy, requests" \
    && ok "crsec_env_mindwell built and imports" \
    || bad "venv rebuild failed (no network on compute node? build from mindwell login node instead)"
elif [ -d "$VSC_DATA/crsec_env_mindwell" ]; then
  source "$VSC_DATA/crsec_env_mindwell/bin/activate" \
    && python3 -c "import numpy, requests" && ok "existing crsec_env_mindwell imports" \
    || bad "existing crsec_env_mindwell broken"
fi

echo "=== 3. ollama on Blackwell ==="
export OLLAMA_KEEP_ALIVE=24h OLLAMA_FLASH_ATTENTION=1
nohup ollama serve > /tmp/ollama_smoke_$$.log 2>&1 &
OLLAMA_PID=$!
sleep 10
if ollama list; then ok "ollama serve up"; else bad "ollama serve failed (see /tmp/ollama_smoke_$$.log)"; fi
ollama list | grep -q "qwen3:30b-instruct" && ok "PRIMARY qwen3:30b-instruct present" \
  || bad "PRIMARY model missing from \$VSC_DATA models"

echo "=== 4. timed llm_call canary (PRIMARY) ==="
cd "$VSC_DATA/CRSEC/reverie/backend_server"
export CRSEC_HEADLESS=1 CRSEC_NUM_CTX=16384 CRSEC_NORM_ON_PRIMARY=1
unset CRSEC_NUM_PREDICT CRSEC_SMALL_MODEL || true
python3 - <<'EOF' && ok "canary returned" || bad "canary failed"
import time, sys
from llm_router import llm_call
t0 = time.time()
out = llm_call("Reply with exactly one word: ready", call_type="smoke_canary")
dt = time.time() - t0
print(f"  canary: {dt:.1f}s -> {out[:80]!r}")
# B200 should comfortably beat the wICE A100 pace; warn if suspiciously slow.
sys.exit(0 if out and dt < 120 else 1)
EOF

kill $OLLAMA_PID 2>/dev/null
echo
echo "=== SMOKE TEST: $PASS passed, $FAIL failed ==="
[ "$FAIL" = 0 ] && echo "mindwell is GO — submit via run_rep_mindwell.sbatch" \
                || echo "mindwell NOT ready — fix failures before submitting anything"
exit "$FAIL"

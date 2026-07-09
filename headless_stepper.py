
import json, os, sys, time

sim = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
base = f"environment/frontend_server/storage/{sim}"
mv_dir, env_dir = f"{base}/movement", f"{base}/environment"
meta_file = f"{base}/reverie/meta.json"
t_start = time.time()

# Wait for the sim folder (reverie creates it on fork), then read the sim's
# CURRENT step so continuation forks resume where reverie does, not at 0.
start_step = None
while start_step is None:
    try:
        with open(meta_file) as f:
            start_step = int(json.load(f).get("step", 0))
    except Exception:
        time.sleep(0.5)
max_step = start_step + n_steps
print(f"[stepper] start_step={start_step} n_steps={n_steps} (until {max_step})", flush=True)

while not os.path.exists(f"{env_dir}/0.json"):
    time.sleep(0.5)
with open(f"{env_dir}/0.json") as f:
    template = json.load(f)
print(f"[stepper] watching {mv_dir}", flush=True)

step = start_step
while step < max_step:
    mv_file = f"{mv_dir}/{step}.json"
    # freshness guard: never consume movement files copied in by the fork —
    # only ones written after this stepper started (reverie launches after us)
    if not os.path.exists(mv_file) or os.path.getmtime(mv_file) < t_start:
        time.sleep(0.3); continue
    time.sleep(0.1)
    try:
        with open(mv_file) as f:
            mv = json.load(f)
    except Exception:
        time.sleep(0.2); continue
    nxt = {}
    for name, info in mv.get("persona", {}).items():
        x, y = info["movement"]
        ent = dict(template.get(name, {"maze": "the_ville"}))
        ent["x"], ent["y"] = x, y
        nxt[name] = ent
    with open(f"{env_dir}/{step+1}.json", "w") as f:
        json.dump(nxt, f, indent=2)
    print(f"[stepper] wrote environment/{step+1}.json", flush=True)
    step += 1
print("[stepper] done", flush=True)

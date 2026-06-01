import json, os, sys, time

sim = sys.argv[1]
max_step = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
base = f"environment/frontend_server/storage/{sim}"
mv_dir, env_dir = f"{base}/movement", f"{base}/environment"

# wait for the sim folder + seed env to exist (reverie creates it on fork)
while not os.path.exists(f"{env_dir}/0.json"):
    time.sleep(0.5)
with open(f"{env_dir}/0.json") as f:
    template = json.load(f)
print(f"[stepper] watching {mv_dir}", flush=True)

step = 0
while step < max_step:
    mv_file = f"{mv_dir}/{step}.json"
    if not os.path.exists(mv_file):
        time.sleep(0.3); continue
    time.sleep(0.1)  # let reverie finish writing
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

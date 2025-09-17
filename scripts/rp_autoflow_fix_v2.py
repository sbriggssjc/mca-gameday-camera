import json, sys, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("outdir")
p.add_argument("--vx-cut", type=float, default=0.60)  # higher => more "pass"
p.add_argument("--mag-cut", type=float, default=8.0)  # higher => fewer "pass"
p.add_argument("--vy-run", type=float, default=0.20)  # higher => fewer "run"
a = p.parse_args()

out = Path(a.outdir)
src = out/"plays.jsonl"
bak = out/"plays.autoflow_backup2.jsonl"
plays = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]

runs = passes = 0
for p in plays:
    f = p.get("auto_flow", {})
    vx = float(f.get("vx_med", 0)); vy = float(f.get("vy_med", 0))
    p95 = float(f.get("mag_p95", 0)); ang = float(f.get("ang_med", 0))

    # pass if pan is strongly horizontal OR lots of fast pixels
    is_pass = (abs(vx) >= a.vx_cut) or (p95 >= a.mag_cut)
    # run if vertical motion dominates or flow angle suggests upfield movement
    is_run  = (not is_pass) and ((abs(vx) <= 0.20 and abs(vy) >= a.vy_run) or (abs(ang) > 1.0 and abs(vy) > abs(vx)))
    if not (is_pass or is_run):
        is_run = abs(vy) > abs(vx)  # tie-breaker

    p["is_run"] = bool(is_run)
    p["is_pass"] = not p["is_run"]
    p["family"] = "run" if p["is_run"] else "pass"
    if p["is_run"]:
        p["run_dir"] = (p.get("direction") or "unknown").lower()

    runs += p["is_run"]; passes += (not p["is_run"])

bak.write_text("\n".join(json.dumps(p) for p in plays))
src.write_text("\n".join(json.dumps(p) for p in plays))
print(f"Re-tag complete. runs={runs} passes={passes}")
print(f"Backup: {bak}\nUpdated: {src}")

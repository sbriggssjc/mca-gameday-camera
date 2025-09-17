import json, sys
from pathlib import Path
out = Path(sys.argv[1])
src = out/"plays.jsonl"
bak = out/"plays.autoflow_backup.jsonl"
plays = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]

def classify(flow):
    vx = float(flow.get("vx_med", 0))
    vy = float(flow.get("vy_med", 0))
    p95 = float(flow.get("mag_p95", 0))
    ang = float(flow.get("ang_med", 0))

    # Simple, robust-ish rule-of-thumb:
    if p95 >= 8 or abs(vx) >= 0.6:
        return "pass"
    if abs(vx) <= 0.2 and abs(vy) >= 0.2:
        return "run"
    if abs(ang) > 1.0 and abs(vy) > abs(vx):
        return "run"
    return "run" if abs(vy) > abs(vx) else "pass"

runs = passes = 0
for p in plays:
    flow = p.get("auto_flow", {})
    rp = classify(flow)
    is_run = (rp == "run")
    p["is_run"] = is_run
    p["is_pass"] = not is_run
    # keep a simple family hint
    p["family"] = "run" if is_run else "pass"
    # set direction bucket for runs using existing "direction" if present
    if is_run:
        p["run_dir"] = (p.get("direction") or "unknown").lower()
    else:
        # keep pass_family as-is, but normalize direction bucket for quick stats
        p["pass_family"] = (p.get("pass_family") or "dropback").lower()
    runs += is_run
    passes += (not is_run)

bak.write_text("\n".join(json.dumps(p) for p in plays))
src.write_text("\n".join(json.dumps(p) for p in plays))
print(f"Re-tag complete. runs={runs} passes={passes}\nBackup: {bak}\nUpdated: {src}")

#!/usr/bin/env python3
import csv, sys, pathlib
out = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
qo = list(csv.DictReader((out/"quick_tendencies_offense.csv").open()))
qd = list(csv.DictReader((out/"quick_tendencies_defense.csv").open()))
def take(bucket, rows, side): return [r for r in rows if r["side"]==side and r["bucket"]==bucket]
def section(title, rows): return "### "+title+"\n" + "\n".join(f"- {r['value']}: {r['count']}" for r in rows) + "\n"
md = ["# Scouting Snapshot", "*(Special teams & excludes removed via audit.)*\n",
      "## Opponent Offense (their plays)",
      section("Run/Pass", take("rp", qo, "offense")),
      section("Direction (rp_dir)", take("rp_dir", qo, "offense")),
      "## Opponent Defense (what offenses did vs them)",
      section("Run/Pass", take("rp", qd, "defense")),
      section("Direction (rp_dir)", take("rp_dir", qd, "defense"))]
(out/"scouting_snapshot.md").write_text("\n".join(md))
print("[wrote]", out/"scouting_snapshot.md")

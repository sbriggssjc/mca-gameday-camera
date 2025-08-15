import json, csv, argparse, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--plays-dir", default="plays")
ap.add_argument("--out", default="summaries")
args = ap.parse_args()

plays_dir = pathlib.Path(args.plays_dir)
summary_dir = pathlib.Path(args.out)
summary_dir.mkdir(parents=True, exist_ok=True)

plays_rows = []
player_rows = []

for play_json in sorted(plays_dir.glob('*/play.json')):
    play_data = json.loads(play_json.read_text())
    grades_path = play_json.parent / 'grades.json'
    grades = []
    if grades_path.exists():
        grades = json.loads(grades_path.read_text())
    plays_rows.append({
        "play_id": play_data.get("play_id"),
        "formation": play_data.get("formation", {}).get("name"),
        "formation_conf": play_data.get("formation", {}).get("confidence"),
        "playcall": play_data.get("playcall", {}).get("name"),
        "play_conf": play_data.get("playcall", {}).get("confidence"),
        "yards": play_data.get("outcome", {}).get("yards"),
        "success": play_data.get("outcome", {}).get("success"),
        "explosive": play_data.get("outcome", {}).get("explosive"),
        "turnover": play_data.get("outcome", {}).get("turnover"),
        "penalty": play_data.get("outcome", {}).get("penalty"),
    })
    for g in grades:
        player_rows.append({
            "play_id": play_data.get("play_id"),
            "player_id": g.get("player_id"),
            "pos": g.get("pos"),
            "grade": g.get("grade"),
        })

with open(summary_dir / 'plays_index.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "play_id","formation","formation_conf","playcall","play_conf","yards","success","explosive","turnover","penalty"
    ])
    writer.writeheader()
    writer.writerows(plays_rows)

with open(summary_dir / 'player_grades.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.DictWriter(f, fieldnames=["play_id","player_id","pos","grade"])
    writer.writeheader()
    writer.writerows(player_rows)

print("Wrote", summary_dir / 'plays_index.csv', 'and', summary_dir / 'player_grades.csv')

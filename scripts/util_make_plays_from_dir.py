import json, os, pathlib, re, sys, datetime
clips_dir = pathlib.Path(sys.argv[1]).expanduser()
dst_dir   = pathlib.Path(sys.argv[2]).expanduser()
dst_dir.mkdir(parents=True, exist_ok=True)
p = dst_dir / 'plays.jsonl'
rx = re.compile(r'P?(\d{1,4})[_-]?(.+?)?\.(mp4|mov)$', re.I)

rows = []
for name in sorted(os.listdir(clips_dir)):
    if not rx.search(name): 
        continue
    m = rx.search(name)
    play_num = int(m.group(1)) if m and m.group(1) else None
    label = (m.group(2) or '').replace('_',' ').replace('-',' ').strip()
    rows.append({
        "src_file": str((clips_dir/name)),
        "clip_file": name,
        "play_number": play_num,
        "label_hint": label,
        "offense_defense": "offense",
        "team_phase": "offense",
        "our_team": "Bartlesville",
        "our_jersey_color": "lightblue",
        "our_pants_color": "navy",
        "opponent_name": "Lincoln Christian",
        "opponent_from_this_game": False,
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
    })

with p.open('w', encoding='utf-8') as w:
    for r in rows:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {p} rows: {len(rows)}")

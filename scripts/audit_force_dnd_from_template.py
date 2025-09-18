#!/usr/bin/env python3
import json, csv, re, sys, pathlib

out = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
plays_path = out/"plays.jsonl"
tmpl_path  = out/"audit"/"audit_template.csv"

ORDINAL = {'1':1,'2':2,'3':3,'4':4,'1st':1,'2nd':2,'3rd':3,'4th':4,
           'first':1,'second':2,'third':3,'fourth':4}
WORD_NUM = {
    'zero':0,'none':0,'no gain':0,'nogain':0,
    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
    'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
    'minus one':-1,'minus two':-2,'minus three':-3,'minus four':-4,'minus five':-5,
}
GOAL_WORDS = ('goal','g','gtg','goal to go','goal-to-go','goal-to go','goal to-go')

def as_int_or_word(s, allow_goal=False):
    if s is None: return None
    s0 = str(s).strip()
    if not s0: return None
    # digits
    m = re.search(r'-?\d+', s0)
    if m: return int(m.group(0))
    # normalize hyphens -> spaces
    s1 = s0.lower().replace('-', ' ')
    # goal-to-go -> 10
    if allow_goal and any(g in s1 for g in GOAL_WORDS): return 10
    # direct word map
    if s1 in WORD_NUM: return WORD_NUM[s1]
    # basic "twenty five"
    parts = s1.split()
    if len(parts)==2 and parts[0]=='twenty' and parts[1] in WORD_NUM:
        return 20 + WORD_NUM[parts[1]]
    return None

def clip_id_from_str(s):
    return _clip_id_from_pathish(s)
    m = re.search(r'(\d{1,4})', str(s))
    return int(m.group(1)) if m else None

def load_template(fn):
    with open(fn, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    # build map: clip# -> (down, to_go)
    m = {}
    for row in rows:
        # clip field often contains a full path ".../Clip 003.mp4"
        cid = clip_id_from_str(row.get('clip') or row.get('src') or row.get('name') or '')
        if cid is None: 
            continue
        # prefer *_fix then base
        d = row.get('down_fix') or row.get('down') or ''
        tg = row.get('distance_fix') or row.get('to_go') or row.get('distance') or row.get('yards_to_go') or ''
        d_parsed  = ORDINAL.get(str(d).strip().lower(), as_int_or_word(d))
        tg_parsed = as_int_or_word(tg, allow_goal=True)
        m[cid] = (d_parsed, tg_parsed)
    return m

def clip_id_from_play(p):
    for k in ('src','clip','name','file','id','idx','title'):
        v = p.get(k)
        if v is None: continue
        cid = clip_id_from_str(v)
        if cid is not None: return cid
    return None

def main():
    if not plays_path.exists():
        print(f"[err] missing {plays_path}")
        sys.exit(1)
    if not tmpl_path.exists():
        print(f"[err] missing {tmpl_path}")
        sys.exit(1)

    dmap = load_template(tmpl_path)
    plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]
    updates = have = 0
    for p in plays:
        cid = clip_id_from_play(p)
        if cid is None or cid not in dmap: 
            continue
        d, tg = dmap[cid]
        if d in (1,2,3,4) and tg is not None:
            have += 1
            # only write if missing/blank
            if not p.get('down'): 
                p['down'] = d; updates += 1
            if p.get('to_go') is None:
                p['to_go'] = int(max(0, tg)); updates += 1

    # write back if any changes
    if updates:
        bkp = plays_path.with_suffix('.audit_backup.jsonl')
        plays_path.replace(bkp)
        with plays_path.open('w') as f:
            for pl in plays:
                f.write(json.dumps(pl, ensure_ascii=False)+'\n')
    print(f"[force-dnd] rows_in_csv={len(dmap)}  matched_plays={have}  fields_written={updates}")

if __name__ == "__main__":
    main()

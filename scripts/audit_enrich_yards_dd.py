#!/usr/bin/env python3
import csv, json, sys, pathlib, re, os, glob

out = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
plays_path = out/"plays.jsonl"
audit_dir  = out/"audit"

def _clip_id_from_pathish(s):
    if not s: return None
    ss = str(s)
    m = re.search(r'clip\s*[-_ ]?\s*(\d{1,4})', ss, flags=re.I)
    if m: return int(m.group(1))
    m2 = list(re.finditer(r'(\d{1,4})', ss))
    if m2: return int(m2[-1].group(1))
    return None

def _clip_id_from_val(v): return _clip_id_from_pathish(v)

def _norm_hdr(k): return re.sub(r'\s+',' ', str(k).strip().lower().replace('_',' ').replace('-',' '))

ORDINAL_MAP = {'1':1,'2':2,'3':3,'4':4,'1st':1,'2nd':2,'3rd':3,'4th':4,'first':1,'second':2,'third':3,'fourth':4}
GOAL_WORDS = ('goal','g','gtg','goal-to-go','goal to go','goal-to go','goal to-go')

WORD_NUMS = {
    'zero':0,'none':0,'no gain':0,'nogain':0,
    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
    'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
    'minus one':-1,'minus two':-2,'minus three':-3,'minus four':-4,'minus five':-5,
}

def _as_int(x):
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    m = re.search(r'-?\d+', s)
    return int(m.group(0)) if m else None

def _as_float(x):
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group(0)) if m else None

def _as_number_any(x, allow_goal=False):
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if m:
        v = float(m.group(0))
        return int(v) if v.is_integer() else v
    s2 = s.lower().replace('-', ' ')
    if allow_goal and any(g in s2 for g in GOAL_WORDS): return 10
    if s2 in WORD_NUMS: return WORD_NUMS[s2]
    parts = s2.split()
    if len(parts)==2 and parts[0]=='twenty' and parts[1] in WORD_NUMS: return 20 + WORD_NUMS[parts[1]]
    return None

def clip_id_from_row(row):
    for k in ("clip","src","name","file","id","idx"):
        if k in row:
            cid = _clip_id_from_val(row[k])
            if cid is not None: return cid
    return None

def clip_id_from_play(p):
    for k in ("src","clip","name","file","id","idx","title"):
        v = p.get(k)
        if v is None: continue
        cid = _clip_id_from_val(v)
        if cid is not None: return cid
    for k,v in p.items():
        if isinstance(v,str) and "clip" in v.lower():
            cid = _clip_id_from_val(v)
            if cid is not None: return cid
    return None

def parse_down_exact(row):
    if not row: return None
    # prefer *_fix
    for k in row.keys():
        nk = _norm_hdr(k)
        if nk in ("down fix","dn fix","d fix"):
            v = _as_number_any(row.get(k))
            if v is not None: return max(1, min(4, int(v)))
    for k in row.keys():
        nk = _norm_hdr(k)
        if nk in ("down","dn","d"):
            vv = row.get(k)
            v = _as_number_any(vv)
            if v is not None: return max(1, min(4, int(v)))
            vv_str = str(vv or "").strip().lower()
            if vv_str in ORDINAL_MAP: return ORDINAL_MAP[vv_str]
    return None

def parse_togo_exact(row):
    if not row: return None
    for k in row.keys():
        nk = _norm_hdr(k)
        if nk in ("to go fix","distance fix","yards to go fix","ytg fix","togo fix"):
            v = _as_number_any(row.get(k), allow_goal=True)
            if v is not None: return max(0, int(v))
    for k in row.keys():
        nk = _norm_hdr(k)
        if nk in ("to go","distance","yards to go","ytg","togo","to-go"):
            v = _as_number_any(row.get(k), allow_goal=True)
            if v is not None: return max(0, int(v))
    return None

def parse_dn_togo_text(s):
    if not s or not str(s).strip(): return (None,None)
    sn = re.sub(r'\s+',' ', str(s)).strip().lower()
    m = re.search(r'\b(1st|2nd|3rd|4th|first|second|third|fourth|[1234])\b[^0-9a-zA-Z]{0,5}(?:and|&|-)?[^0-9a-zA-Z]{0,5}(\d+|goal|g|gtg)\b', sn)
    if m:
        down_raw, dist_raw = m.groups()
        down = ORDINAL_MAP.get(down_raw, _as_number_any(down_raw))
        to_go = 10 if dist_raw in ('goal','g','gtg') else _as_number_any(dist_raw)
        if down in (1,2,3,4) and to_go is not None: return (down, max(0,int(to_go)))
    m = re.search(r'\b(1st|2nd|3rd|4th|first|second|third|fourth|[1234])\b.*?\b(goal(?:\s*-\s*to\s*-\s*go)?|goal\s*to\s*go|gtg|g)\b', sn)
    if m:
        down = ORDINAL_MAP.get(m.group(1), None)
        if down in (1,2,3,4): return (down, 10)
    m = re.search(r'\b(1st|2nd|3rd|4th|[1234])\b[^0-9a-zA-Z]{0,2}(\d{1,2})\b', sn)
    if m:
        down = ORDINAL_MAP.get(m.group(1), _as_number_any(m.group(1)))
        to_go = _as_number_any(m.group(2))
        if down in (1,2,3,4) and to_go is not None: return (down, max(0,int(to_go)))
    return (None,None)

def parse_dn_togo_combo(row):
    if not row: return (None,None)
    for k,v in row.items():
        if isinstance(v,str) and v.strip():
            d,tg = parse_dn_togo_text(v)
            if d is not None or tg is not None: return d,tg
    return (None,None)

def parse_dn_togo_from_play(play):
    for k,v in play.items():
        if isinstance(v,str) and v.strip():
            d,tg = parse_dn_togo_text(v)
            if d is not None or tg is not None: return d,tg
    return (None,None)

def parse_yards(row):
    def norm(k): return _norm_hdr(k)
    keys_fix = {"yards gained fix","gained yards fix","result yards fix"}
    keys_any = {"yards gained","gained yards","yards","yds","gain","gained","result yards"}
    for k in (list(row.keys()) if row else []):
        nk = norm(k)
        if nk in keys_fix:
            v = _as_number_any(row.get(k))
            if v is not None: return v
    for k in (list(row.keys()) if row else []):
        nk = norm(k)
        if nk in keys_any:
            v = _as_number_any(row.get(k))
            if v is not None: return v
    if row:
        for k,v in row.items():
            if re.search(r'yard|gain', str(k), re.I):
                fv = _as_number_any(v)
                if fv is not None: return fv
    return None

def needs_update(curr, new):
    if new is None: return False
    if curr is None: return True
    if isinstance(curr, (int, float)) and float(curr) == 0.0: return True
    if isinstance(curr, str) and not curr.strip(): return True
    return False

def pick_best_csv(audit_dir):
    dnd_csv = os.environ.get("DND_CSV")
    if dnd_csv and pathlib.Path(dnd_csv).exists(): return pathlib.Path(dnd_csv)
    best = None; best_score = -1
    for fn in sorted(glob.glob(str(audit_dir / "*.csv"))):
        try:
            with open(fn, newline='') as f:
                r = csv.reader(f); hdr = next(r)
            hs = {_norm_hdr(h) for h in hdr}
            score = 0
            if any(h in hs for h in {"down","dn","d","down fix","dn fix","d fix"}): score += 2
            if any(h in hs for h in {"to go","distance","yards to go","ytg","togo","to go fix","distance fix","yards to go fix","ytg fix","togo fix"}): score += 2
            if any(h in hs for h in {"yards gained","gained yards","yards","yds","gain","gained","result yards","yards gained fix","gained yards fix","result yards fix"}): score += 1
            if score > best_score: best_score, best = score, pathlib.Path(fn)
        except Exception:
            pass
    return best

audit_csv = pick_best_csv(audit_dir)

def load_csv_rows(p):
    if not p: return {}
    with p.open(newline='') as f:
        rdr = csv.DictReader(f)
        rows = {}
        for row in rdr:
            cid = clip_id_from_row(row)
            if cid is not None:
                rows[cid] = row
        return rows

def load_plays(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def write_plays(p, plays):
    p_backup = p.with_suffix('.audit_backup.jsonl')
    p.replace(p_backup)
    with p.open('w') as f:
        for pl in plays:
            f.write(json.dumps(pl, ensure_ascii=False) + '\n')

def main():
    if not plays_path.exists():
        print(f"[warn] missing {plays_path}; nothing to enrich"); return
    rows = load_csv_rows(audit_csv)
    print(f"[csv] using {audit_csv}")
    plays = load_plays(plays_path)

    updates = matched = dn_exact_hits = row_combo_hits = play_combo_hits = 0
    for p in plays:
        cid = clip_id_from_play(p)
        row = rows.get(cid) if cid in rows else None
        if row is not None: matched += 1

        yg = parse_yards(row) if row else None

        d = tg = None
        if row:
            d  = parse_down_exact(row)
            tg = parse_togo_exact(row)
            if d is not None and tg is not None:
                dn_exact_hits += 1
            else:
                d2,tg2 = parse_dn_togo_combo(row)
                if d is None and d2 is not None: d = d2
                if tg is None and tg2 is not None: tg = tg2
                if d2 is not None or tg2 is not None: row_combo_hits += 1

        if d is None or tg is None:
            d3,tg3 = parse_dn_togo_from_play(p)
            if d is None and d3 is not None: d = d3
            if tg is None and tg3 is not None: tg = tg3
            if d3 is not None or tg3 is not None: play_combo_hits += 1

        if needs_update(p.get('down'), d):           p['down'] = d; updates += 1
        if needs_update(p.get('to_go'), tg):         p['to_go'] = tg; updates += 1
        if needs_update(p.get('yards_gained'), yg):  p['yards_gained'] = yg; updates += 1
        if needs_update(p.get('yards'), yg):         p['yards'] = yg; updates += 1
        if needs_update(p.get('yg'), yg):            p['yg'] = yg; updates += 1
        if needs_update(p.get('gained_yards'), yg):  p['gained_yards'] = yg; updates += 1
        if needs_update(p.get('gain'), yg):          p['gain'] = yg; updates += 1
        if needs_update(p.get('gained'), yg):        p['gained'] = yg; updates += 1
        if needs_update(p.get('result_yards'), yg):  p['result_yards'] = yg; updates += 1

    if updates: write_plays(plays_path, plays)
    print(f"[enriched] matched {matched} plays; applied {updates} field updates across {len(plays)} plays (dn_exact={dn_exact_hits}, row_combo={row_combo_hits}, play_combo={play_combo_hits})")

if __name__ == "__main__":
    main()

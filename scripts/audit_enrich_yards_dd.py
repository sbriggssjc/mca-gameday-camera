#!/usr/bin/env python3
import csv, json, sys, pathlib, re

out = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
plays_path = out/"plays.jsonl"
audit_dir  = out/"audit"

candidates = [
    audit_dir/"audit_template.csv",
    audit_dir/"audit_kept_debug.csv",
    audit_dir/"audit_kept.csv",
    audit_dir/"audit_disagreements.csv",
]
audit_csv = next((p for p in candidates if p.exists()), None)

def _norm_hdr(k):
    import re
    return re.sub(r'\s+',' ', str(k).strip().lower().replace('_',' ').replace('-',' '))

ORDINAL_MAP = {
    '1':1,'2':2,'3':3,'4':4,
    '1st':1,'2nd':2,'3rd':3,'4th':4,
    'first':1,'second':2,'third':3,'fourth':4,
}
GOAL_WORDS = ('goal','g','gtg','goal-to-go','goal to go','goal-to go','goal to-go')

WORD_NUMS = {
    'zero':0,'none':0,'no gain':0,'nogain':0,
    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
    'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
    'minus one':-1,'minus two':-2,'minus three':-3,'minus four':-4,'minus five':-5,
}
def _as_number_any(x, allow_goal=False):
    if x is None: return None
    s = str(x).strip().lower()
    # normalize spaces/hyphens
    s2 = re.sub(r'\s+', ' ', s.replace('-', ' '))
    # digits first
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if m:
        v = float(m.group(0))
        return int(v) if v.is_integer() else v
    # goal-to-go => 10 if allowed
    if allow_goal and any(g in s2 for g in GOAL_WORDS):
        return 10
    # direct lookup
    if s2 in WORD_NUMS: return WORD_NUMS[s2]
    # split pairs like "twenty five" (basic)
    parts = s2.split()
    if len(parts)==2 and parts[0] in ('twenty',) and parts[1] in WORD_NUMS:
        base = 20
        return base + WORD_NUMS[parts[1]]
    return None

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

def _clip_id_from_val(v):
    return _clip_id_from_pathish(v)
    m = re.search(r'(\d{1,4})', str(v))
    return int(m.group(1)) if m else None

def clip_id_from_row(row):
    for k in ("clip","src","name","file","id","idx"):
        if k in row:
            cid = _clip_id_from_val(row[k])
            if cid is not None: return cid
    return None

def clip_id_from_play(p):
    for k in ("src","clip","name","file","id","idx"):
        cid = _clip_id_from_val(p.get(k))
        if cid is not None: return cid
    return None

def parse_down_exact(row):
    if not row: return None
    # prefer *_fix
    for k in row.keys():
        nk = _norm_hdr(k)
        if nk in ("down fix","dn fix","d fix"):
            v = _as_int(row.get(k))
            if v is not None: return max(1, min(4, v))
        if nk in ("down","dn","d"):
            # accept word ordinals too (First/Second...)
            vv = row.get(k)
            v = _as_int(vv)
            if v is not None: return max(1, min(4, v))
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
    for k in row.keys():
        nk = _norm_hdr(k)
        if nk in ("to go fix","distance fix","yards to go fix","ytg fix","togo fix"):
            v = _as_int(row.get(k))
            if v is not None: return max(0, v)
        if nk in ("to go","distance","yards to go","ytg","togo","to-go"):
            v = _as_int(row.get(k))
            if v is not None: return max(0, v)
    return None

# Try to parse a combined "down & distance" string from ANY column
def parse_dn_togo_combo(row):
    def norm(s): return re.sub(r'\s+', ' ', s).strip().lower()

    # search all text fields for patterns like "1st & 10", "3rd and 7", "Second-5", "1st & Goal"
    for k,v in row.items():
        s = str(v or "")
        if not s.strip(): continue
        sn = norm(s)

        # Pattern A: <ordinal> [and|&|-] <number|goal>
        m = re.search(r'\b(1st|2nd|3rd|4th|first|second|third|fourth|[1234])\b[^0-9a-zA-Z]{0,5}(?:and|&|-)?[^0-9a-zA-Z]{0,5}(\d+|goal|g|gtg)\b', sn)
        if m:
            down_raw, dist_raw = m.groups()
            down = ORDINAL_MAP.get(down_raw, _as_int(down_raw))
            if dist_raw in GOAL_WORDS or dist_raw in ('goal','g','gtg'):
                # If only "Goal" given, assume 10 yards to go if no number is present
                to_go = 10
            else:
                to_go = _as_int(dist_raw)
            if down in (1,2,3,4) and to_go is not None:
                return down, max(0, to_go)

        # Pattern B: "<ordinal> & goal-to-go" without a number
        m = re.search(r'\b(1st|2nd|3rd|4th|first|second|third|fourth|[1234])\b.*?\b(goal(?:\s*-\s*to\s*-\s*go)?|goal\s*to\s*go|gtg|g)\b', sn)
        if m:
            down_raw = m.group(1)
            down = ORDINAL_MAP.get(down_raw, _as_int(down_raw))
            if down in (1,2,3,4):
                return down, 10

        # Pattern C: compact "3rd-7" or "2nd&5"
        m = re.search(r'\b(1st|2nd|3rd|4th|[1234])\b[^0-9a-zA-Z]{0,2}(\d{1,2})\b', sn)
        if m:
            down_raw, dist_num = m.groups()
            down = ORDINAL_MAP.get(down_raw, _as_int(down_raw))
            to_go = _as_int(dist_num)
            if down in (1,2,3,4) and to_go is not None:
                return down, max(0, to_go)

    return None, None

def parse_yards(row):
    def norm(k): return _norm_hdr(k)
    # Prefer common names; then any header mentioning yard/gain
    for k in ("yards_gained","yards","yds","gained_yards","gained","gain","result","result_yards"):
        v = _as_float(row.get(k))
        if v is not None:
            return v
    for k,v in row.items():
        if re.search(r'yard|gain', str(k), re.I):
            fv = _as_float(v)
            if fv is not None:
                return fv
    return None

def needs_update(curr, new):
    if new is None: return False
    if curr is None: return True
    if isinstance(curr, (int, float)) and float(curr) == 0.0: return True
    if isinstance(curr, str) and not curr.strip(): return True
    return False

def load_csv_rows(p):
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
        print(f"[warn] missing {plays_path}; nothing to enrich")
        return
    if not audit_csv:
        print(f"[warn] no audit CSV found in {audit_dir}; nothing to enrich")
        return

    print(f"[csv] using {audit_csv}")
    rows = load_csv_rows(audit_csv)
    plays = load_plays(plays_path)

    updates = matched = dn_hits = combo_hits = 0
    for p in plays:
        cid = clip_id_from_play(p)
        if cid is None or cid not in rows:
            continue
        matched += 1
        row = rows[cid]

        # yards
        yg = parse_yards(row)

        # down & to-go (try exact, then combo)
        d  = parse_down_exact(row)
        tg = parse_togo_exact(row)
        if d is None or tg is None:
            d2, tg2 = parse_dn_togo_combo(row)
            if d is None and d2 is not None: d = d2
            if tg is None and tg2 is not None: tg = tg2
            if d2 is not None or tg2 is not None: combo_hits += 1
        else:
            dn_hits += 1

        if needs_update(p.get('down'), d):           p['down'] = d; updates += 1
        if needs_update(p.get('to_go'), tg):         p['to_go'] = tg; updates += 1
        if needs_update(p.get('yards_gained'), yg):  p['yards_gained'] = yg; updates += 1
        # synonyms for downstream scripts
        if needs_update(p.get('yards'), yg):         p['yards'] = yg; updates += 1
        if needs_update(p.get('yg'), yg):            p['yg'] = yg; updates += 1

    if updates:
        write_plays(plays_path, plays)

    print(f"[enriched] matched {matched} plays; applied {updates} field updates across {len(plays)} plays (dn_exact={dn_hits}, dn_combo={combo_hits})")

if __name__ == "__main__":
    main()

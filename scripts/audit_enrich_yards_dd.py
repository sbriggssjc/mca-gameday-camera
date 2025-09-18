#!/usr/bin/env python3
import csv, json, sys, pathlib, re

out = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
plays_path = out/"plays.jsonl"
audit_dir  = out/"audit"

# Find a usable audit CSV (template or kept_debug etc.)
candidates = [
    audit_dir/"audit_template.csv",
    audit_dir/"audit_kept_debug.csv",
    audit_dir/"audit_kept.csv",
    audit_dir/"audit_disagreements.csv",
]
audit_csv = next((p for p in candidates if p.exists()), None)

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
    if v is None: return None
    m = re.search(r'(\d{1,4})', str(v))
    return int(m.group(1)) if m else None

def clip_id_from_row(row):
    for k in ("clip","src","name","file","id","idx"):
        if k in row:
            cid = _clip_id_from_val(row[k])
            if cid is not None:
                return cid
    return None

def clip_id_from_play(p):
    for k in ("src","clip","name","file","id","idx"):
        cid = _clip_id_from_val(p.get(k))
        if cid is not None:
            return cid
    return None

def parse_down(row):
    for k in ("down","dn","d"):
        v = _as_int(row.get(k))
        if v is not None:
            return max(1, min(4, v))
    return None

def parse_togo(row):
    for k in ("to_go","to-go","distance","yards_to_go","ytg","togo"):
        v = _as_int(row.get(k))
        if v is not None:
            return max(0, v)
    return None

def parse_yards(row):
    # Prefer common names; then any header mentioning yard/gain
    for k in ("yards_gained","yards","yds","gained","gain","result","result_yards"):
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

    rows = load_csv_rows(audit_csv)
    plays = load_plays(plays_path)

    updates = 0
    matched = 0
    for p in plays:
        cid = clip_id_from_play(p)
        if cid is None or cid not in rows:
            continue
        matched += 1
        row = rows[cid]
        d  = parse_down(row)
        tg = parse_togo(row)
        yg = parse_yards(row)

        if needs_update(p.get('down'), d):           p['down'] = d; updates += 1
        if needs_update(p.get('to_go'), tg):         p['to_go'] = tg; updates += 1
        if needs_update(p.get('yards_gained'), yg):  p['yards_gained'] = yg; updates += 1
        # Write common synonyms used by other scripts
        if needs_update(p.get('yards'), yg):         p['yards'] = yg; updates += 1
        if needs_update(p.get('yg'), yg):            p['yg'] = yg; updates += 1

    if updates:
        write_plays(plays_path, plays)

    print(f"[enriched] matched {matched} plays; applied {updates} field updates across {len(plays)} plays")

if __name__ == "__main__":
    main()

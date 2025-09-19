#!/usr/bin/env python3
"""
Export a single Excel workbook that includes:
- A complete catalog of plays (all attributes we can find), including the clip filename.
- Summary tables for OFFENSE and DEFENSE:
  * Outcomes by play (if a play name/call exists)
  * Play type (run/pass) distribution
  * Run direction breakdown
  * Down & distance breakdown
  * Yardage buckets

Inputs (auto-detected):
  OUT/plays.jsonl                     -> primary source
  OUT/audit/audit_template.csv        -> enrich with *_fix fields if present
  OUT/clips/                          -> used to try to map clip filenames (best-effort)
Output:
  OUT/export/opponent_all_plays.xlsx  -> all catalog + summaries in one workbook
"""

import os, json, math, re, sys, glob
from pathlib import Path

import pandas as pd

# -------- helpers --------
def get_out_path():
    out = os.environ.get("OUT") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not out:
        raise SystemExit("Provide OUT via env or as first CLI arg")
    return Path(out)

def safe_read_jsonl(p: Path) -> pd.DataFrame:
    rows = []
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    with p.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # keep going even if a line is malformed
                continue
    return pd.json_normalize(rows, sep=".")

def read_audit_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        return df
    except Exception:
        return None

def first_nonempty(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "", float("nan")) and pd.notna(v):
            return v
    return ""

def norm_side(v: str) -> str:
    s = (str(v or "")).strip().lower()
    if s in {"o","off","offense","offence"}: return "offense"
    if s in {"d","def","defense","defence"}: return "defense"
    return s

def distance_bucket(dist):
    # dist may be str or number
    try:
        d = float(dist)
    except Exception:
        return "unknown"
    if d <= 2: return "0-2"
    if d <= 5: return "3-5"
    if d <= 8: return "6-8"
    if d <= 12: return "9-12"
    return "13+"

def yard_bucket(y):
    try:
        v = float(y)
    except Exception:
        return "unknown"
    if v <= -3: return "<= -3"
    if v <= 0:  return "-2 to 0"
    if v <= 3:  return "1 to 3"
    if v <= 7:  return "4 to 7"
    return "8+"

def best_clip_for_index(out_dir: Path, idx):
    # Try to find something that looks like the per-play clip
    # Look for exact clip name stored in json first (will be merged below).
    # If empty, try to glob clips folder using index. This is best-effort.
    cdir = out_dir / "clips"
    if not cdir.exists():
        return ""
    patts = [
        f"*_{idx}_*.mp4",
        f"*_{idx}.mp4",
        f"{idx}_*.mp4",
        f"{idx}.mp4",
        f"*{idx}*.mp4",
    ]
    for pat in patts:
        files = sorted(glob.glob(str(cdir / pat)))
        if files:
            return Path(files[0]).name
    return ""

def to_bool(x):
    s = str(x or "").strip().lower()
    return s in {"1","true","yes","y"}

# -------- main --------
def main():
    out = get_out_path()
    plays_path = out / "plays.jsonl"
    audit_path = out / "audit" / "audit_template.csv"
    export_dir = out / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = export_dir / "opponent_all_plays.xlsx"

    # load plays + audit
    df = safe_read_jsonl(plays_path)
    audit = read_audit_csv(audit_path)

    # Try to identify index key
    # Priority: explicit "index" in plays.jsonl, else in audit, else create row number
    idx_col = None
    for cand in ["index","idx","row_index","row_id","play_index","play_id"]:
        if cand in df.columns:
            idx_col = cand
            break
    if idx_col is None:
        df = df.copy()
        df["index"] = range(len(df))
        idx_col = "index"

    # Merge audit fields (side_fix, rp_fix, dir_fix, st_fix, down_fix, distance_fix, gained_yards_fix, exclude, notes_fix)
    if audit is not None:
        # find audit index column
        a_idx = None
        for cand in audit.columns:
            if str(cand).lower() == "index":
                a_idx = cand
                break
        if a_idx is None:
            # synthesize
            audit = audit.copy()
            audit["index"] = range(len(audit))
            a_idx = "index"

        # Keep audit columns with meaningful info
        keep_cols = [c for c in audit.columns if any(k in str(c).lower() for k in [
            "side", "rp", "dir", "st_", "down", "distance", "gained_yards", "exclude", "notes", "auto_outcome", "phase"
        ]) or str(c).lower() == "index"]

        audit_small = audit[keep_cols].copy()
        df = df.merge(audit_small, left_on=idx_col, right_on=a_idx, how="left", suffixes=("",""))

    # Compose the side-of-ball we should use for Jenks and Metro
    # Prefer fields inserted by your fixer; fallback to audit if needed
    df["jenks_side"] = df.get("jenks_side", pd.Series(index=df.index, dtype="object"))
    # If jenks_side is empty, try side_fix/side_auto
    df["jenks_side"] = df["jenks_side"].where(df["jenks_side"].astype(str).str.len() > 0,
                                              df.get("side_fix", "").apply(norm_side))
    df["jenks_side"] = df["jenks_side"].where(df["jenks_side"].astype(str).str.len() > 0,
                                              df.get("side_auto", "").apply(norm_side))
    # Metro is complement if jenks_side is offense/defense
    df["metro_side"] = df["jenks_side"].map(lambda s: "defense" if s=="offense" else ("offense" if s=="defense" else ""))

    # Clip filename column (prefer clip from JSON; else best effort from /clips)
    df["clip_file"] = df.get("clip", "")
    missing_clip_mask = (df["clip_file"].isna()) | (df["clip_file"].astype(str).str.strip() == "") | (df["clip_file"] == "null")
    if missing_clip_mask.any():
        for i in df[missing_clip_mask].index:
            idx_val = df.at[i, idx_col]
            df.at[i, "clip_file"] = best_clip_for_index(out, idx_val)

    # Normalize key analysis fields with fallback to *_fix -> *_auto -> raw
    def coalesce(*cols):
        for c in cols:
            if c in df.columns:
                v = df[c]
                if v.notna().any():
                    return v
        return pd.Series([""]*len(df))

    df["phase_use"]       = coalesce("phase", "phase_fix", "phase_auto")
    df["play_type_use"]   = coalesce("rp_fix","rp_auto","play_type","type").astype(str).str.lower().replace({
        "r":"run","p":"pass","run/pass":"runpass"
    })
    df["dir_use"]         = coalesce("dir_fix","dir_auto","direction")
    df["down_use"]        = coalesce("down_fix","down").astype(str).str.extract(r"(\d+)")[0]
    df["distance_use"]    = coalesce("distance_fix","distance")
    df["gained_yards_use"]= coalesce("gained_yards_fix","gained_yards","yards_gained","yards")

    # Buckets for reporting
    df["distance_bucket"] = df["distance_use"].apply(distance_bucket)
    df["yards_bucket"]    = df["gained_yards_use"].apply(yard_bucket)

    # Outcome column if present; fallback to a simple derived outcome
    outcome = df.get("auto_outcome")
    if outcome is None or outcome.isna().all():
        # crude proxy from yards gained
        def derived_outcome(y):
            try:
                v = float(y)
            except Exception:
                return "unknown"
            if v <= -1: return "negative"
            if v <= 2:  return "short"
            if v <= 6:  return "medium"
            return "explosive"
        df["outcome_use"] = df["gained_yards_use"].apply(derived_outcome)
    else:
        df["outcome_use"] = outcome

    # OFFENSE/DEFENSE filters (Jenks perspective)
    off = df[df["jenks_side"] == "offense"].copy()
    deff = df[df["jenks_side"] == "defense"].copy()

    # ---- Summaries ----
    def value_counts_df(frame, by_col, name):
        if by_col not in frame.columns:
            return pd.DataFrame(columns=[by_col, "count"])
        g = frame[by_col].fillna("").astype(str)
        vc = g.value_counts(dropna=False).reset_index()
        vc.columns = [by_col, "count"]
        vc.insert(0, "_summary", name)
        return vc

    # Basic tables
    off_by_playtype   = value_counts_df(off, "play_type_use", "off_by_playtype")
    off_by_direction  = value_counts_df(off, "dir_use", "off_by_direction")
    off_by_down       = value_counts_df(off, "down_use", "off_by_down")
    off_by_distbucket = value_counts_df(off, "distance_bucket", "off_by_distance_bucket")
    off_by_outcome    = value_counts_df(off, "outcome_use", "off_by_outcome")
    off_by_yards      = value_counts_df(off, "yards_bucket", "off_by_yards_bucket")

    def_by_playtype   = value_counts_df(deff, "play_type_use", "def_by_playtype")
    def_by_direction  = value_counts_df(deff, "dir_use", "def_by_direction")
    def_by_down       = value_counts_df(deff, "down_use", "def_by_down")
    def_by_distbucket = value_counts_df(deff, "distance_bucket", "def_by_distance_bucket")
    def_by_outcome    = value_counts_df(deff, "outcome_use", "def_by_outcome")
    def_by_yards      = value_counts_df(deff, "yards_bucket", "def_by_yards_bucket")

    # If you track a play name/call, map it to a canonical column
    play_name_col = None
    for cand in ("play_name","call","play","concept","tag","name"):
        if cand in df.columns:
            play_name_col = cand
            break
    off_by_play = value_counts_df(off, play_name_col, "off_by_play") if play_name_col else pd.DataFrame()
    def_by_play = value_counts_df(deff, play_name_col, "def_by_play") if play_name_col else pd.DataFrame()

    # Build catalog (wide) — put important columns first
    front_cols = []
    for c in (idx_col, "clip_file", "jenks_side", "metro_side",
              "play_type_use","dir_use","down_use","distance_use","gained_yards_use",
              "distance_bucket","yards_bucket","outcome_use"):
        if c in df.columns: front_cols.append(c)

    # Move front_cols to the front, keep everything else after
    all_cols = front_cols + [c for c in df.columns if c not in front_cols]

    catalog = df[all_cols].copy()

    # ---- Write Excel ----
    # Prefer xlsxwriter; fallback to openpyxl
    engine = "xlsxwriter"
    try:
        with pd.ExcelWriter(xlsx_path, engine=engine) as xl:
            catalog.to_excel(xl, sheet_name="catalog", index=False)

            # Offense summaries
            off_by_playtype.to_excel(xl, sheet_name="summary_offense", index=False, startrow=0)
            off_by_direction.to_excel(xl, sheet_name="summary_offense", index=False, startrow=off_by_playtype.shape[0]+2)
            off_by_down.to_excel(xl, sheet_name="summary_offense", index=False, startrow=off_by_playtype.shape[0]+off_by_direction.shape[0]+4)
            off_by_distbucket.to_excel(xl, sheet_name="summary_offense", index=False, startrow=off_by_playtype.shape[0]+off_by_direction.shape[0]+off_by_down.shape[0]+6)
            off_by_outcome.to_excel(xl, sheet_name="summary_offense", index=False, startrow=off_by_playtype.shape[0]+off_by_direction.shape[0]+off_by_down.shape[0]+off_by_distbucket.shape[0]+8)
            off_by_yards.to_excel(xl, sheet_name="summary_offense", index=False, startrow=off_by_playtype.shape[0]+off_by_direction.shape[0]+off_by_down.shape[0]+off_by_distbucket.shape[0]+off_by_outcome.shape[0]+10)
            if not off_by_play.empty:
                off_by_play.to_excel(xl, sheet_name="summary_offense", index=False, startrow=off_by_playtype.shape[0]+off_by_direction.shape[0]+off_by_down.shape[0]+off_by_distbucket.shape[0]+off_by_outcome.shape[0]+off_by_yards.shape[0]+12)

            # Defense summaries
            def_by_playtype.to_excel(xl, sheet_name="summary_defense", index=False, startrow=0)
            def_by_direction.to_excel(xl, sheet_name="summary_defense", index=False, startrow=def_by_playtype.shape[0]+2)
            def_by_down.to_excel(xl, sheet_name="summary_defense", index=False, startrow=def_by_playtype.shape[0]+def_by_direction.shape[0]+4)
            def_by_distbucket.to_excel(xl, sheet_name="summary_defense", index=False, startrow=def_by_playtype.shape[0]+def_by_direction.shape[0]+def_by_down.shape[0]+6)
            def_by_outcome.to_excel(xl, sheet_name="summary_defense", index=False, startrow=def_by_playtype.shape[0]+def_by_direction.shape[0]+def_by_down.shape[0]+def_by_distbucket.shape[0]+8)
            def_by_yards.to_excel(xl, sheet_name="summary_defense", index=False, startrow=def_by_playtype.shape[0]+def_by_direction.shape[0]+def_by_down.shape[0]+def_by_distbucket.shape[0]+def_by_outcome.shape[0]+10)
            if not def_by_play.empty:
                def_by_play.to_excel(xl, sheet_name="summary_defense", index=False, startrow=def_by_playtype.shape[0]+def_by_direction.shape[0]+def_by_down.shape[0]+def_by_distbucket.shape[0]+def_by_outcome.shape[0]+def_by_yards.shape[0]+12)

    except Exception:
        # fallback engine
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
            catalog.to_excel(xl, sheet_name="catalog", index=False)
            off_by_playtype.to_excel(xl, sheet_name="summary_offense", index=False)
            def_by_playtype.to_excel(xl, sheet_name="summary_defense", index=False)

    print(f"[export] wrote: {xlsx_path}")

if __name__ == "__main__":
    main()

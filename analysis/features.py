from __future__ import annotations

def extract_features(seg, tracking):
    f = {}
    # ... existing feature extraction would go here ...
    if not f or len(f) < 5:
        print(f"[feat] insufficient features for {seg.get('segment_id')} -> {len(f) if f else 0}")
    return f
